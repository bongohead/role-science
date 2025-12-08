"""
Reversed engineered forward pass for GLM-4.x MoE
- Supports GLM-4.5, GLM-4.5-Air, GLM-4.6 (HF v5 modular glm4_moe)
- See https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm4_moe/modeling_glm4_moe.py
- This supports multiple device usage
"""
import torch
from transformers.masking_utils import create_causal_mask
from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMoE
from ._pretrained_helpers import _sort_gate_tensors, _move_device

@torch.no_grad()
def run_glm4moe_return_topk(model, input_ids, attention_mask, return_hidden_states: bool = False):
    """
    Params:
        @model: A model of class `Glm4MoeForCausalLM`.
        @input_ids: A (B, N) tensor of inputs IDs on the same device as `model`.
        @attention_mask: A (B, N) tensor of mask indicators on the same device as `model`.
        @return_hidden_states: Boolean; whether to return optional outputs below.

    Returns:
        A dictionary with keys:
        - `logits`: (B, N, V) LM outputs
        - `all_topk_experts`: List (len = # MoE layers) of (BN, topk) expert IDs tensors
        - `all_topk_weights`: List (len = # MoE layers) of (BN, topk) expert weight tensors
        - `all_pre_mlp_hidden_states`: (optional) List (len = # MoE layers) of (BN, D) pre-MLP activations
        - `all_router_logits: (optional) List (len = # MoE layers) of (BN, n_experts) router *logits*
        - `all_hidden_states`: (optional) List (len = # MoE layers) of (BN, D) post-layer activations
        - `all_expert_outputs`: (optional) List (len = # MoE layers) of (BN, topk, D) pre-weighting expert outputs
    """
    ##### Setup #####
    # Pick the device that holds the token‑embedding weights as our starting point.
    emb_device = model.model.embed_tokens.weight.device
    input_ids = _move_device(input_ids, emb_device)
    attention_mask = _move_device(attention_mask, emb_device)

    input_embeds = model.model.embed_tokens(input_ids) # (B, N, D)
    B, N, D = input_embeds.shape

    cache_position = torch.arange(N, device = emb_device)
    position_ids = cache_position.unsqueeze(0) # (1, N)
    causal_mask = create_causal_mask(model.model.config, input_embeds, attention_mask, cache_position, None, position_ids) # (B, 1, N, N)

    # Compute ROPE embs (copy to each GPU as needed)
    position_embeddings = model.model.rotary_emb(input_embeds, position_ids)
    cos_global, sin_global = position_embeddings

    hidden_state = input_embeds

    all_topk_experts = []
    all_topk_weights = []
    all_pre_mlp_hidden_states = []
    all_router_logits = []
    all_hidden_states = []
    all_expert_outputs = []

    ##### Transformer Layers #####
    for layer in model.model.layers:

        # Device of this layers parameters (assuming all same)
        layer_dev = next(layer.parameters()).device

        # Move working tensors to that GPU (if not already there)
        hidden_state = _move_device(hidden_state, layer_dev)
        causal_mask = _move_device(causal_mask, layer_dev)
        position_ids = _move_device(position_ids, layer_dev)
        cos = _move_device(cos_global, layer_dev)
        sin = _move_device(sin_global, layer_dev)
        pos_emb = (cos, sin)

        # SA 
        residual = hidden_state
        hidden_state = layer.input_layernorm(hidden_state)
        hidden_state, _ = layer.self_attn(
            hidden_states = hidden_state,
            attention_mask = causal_mask,
            position_ids = position_ids,
            cache_position = cache_position,
            position_embeddings = pos_emb,
        )
        hidden_state = residual + hidden_state

        # MoE
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)

        if isinstance(layer.mlp, Glm4MoeMoE):
            B_, S_, D_ = hidden_state.shape
            moe_hidden_state_flat = hidden_state.view(-1, D_) # (BN,D)

            # 1. Router logits (pre-sigmoid) from gate, exactly like Glm4MoeMoE.forward
            router_logits = layer.mlp.gate(hidden_state)  # (BN, n_routed_experts)

            # 2. HF routing logic: sigmoid + group selection + top-k within groups
            topk_ids, topk_weight = layer.mlp.route_tokens_to_experts(router_logits)  # both (BN, top_k)

            # 3. Manually replicate Glm4MoeNaiveMoe.forward to get per-expert outputs
            num_exp = layer.mlp.experts.num_experts # config.num_local_experts
            K = layer.mlp.top_k # config.num_experts_per_tok

            final_flat = torch.zeros_like(moe_hidden_state_flat)  # (BN, D_), dtype matches hidden_state

            if return_hidden_states:
                layer_expert_outputs = moe_hidden_state_flat.new_zeros((moe_hidden_state_flat.size(0), K, D_)) # (BN, K, D)
            expert_mask = torch.nn.functional.one_hot(topk_ids, num_classes = num_exp).permute(2, 1, 0) # (num_exp, K, BN)
            expert_hits = torch.greater(expert_mask.sum(dim = (-1, -2)), 0).nonzero()  # (num_hits, 1)

            for expert_idx in expert_hits:
                expert_idx = expert_idx[0]

                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])  # each (ntoks,)
                if token_idx.numel() == 0:
                    continue
                current_state = moe_hidden_state_flat[token_idx]  # (ntoks, D_)

                # Same computation as Glm4MoeNaiveMoe.forward, but upcast weights
                w_gate = layer.mlp.experts.gate_up_proj[expert_idx].to(current_state.dtype)
                gate, up = torch.nn.functional.linear(current_state, w_gate).chunk(2, dim = -1)
                current_expert_output = layer.mlp.experts.act_fn(gate) * up

                w_down = layer.mlp.experts.down_proj[expert_idx].to(current_expert_output.dtype)
                current_expert_output = torch.nn.functional.linear(current_expert_output, w_down)
                # This matches HF v5.0.0 but is bugged, used above to upcast instead
                # gate, up = torch.nn.functional.linear(current_state, layer.mlp.experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                # current_expert_output = layer.mlp.experts.act_fn(gate) * up 
                # current_expert_output = torch.nn.functional.linear(current_expert_output, layer.mlp.experts.down_proj[expert_idx])  # (ntoks, D_)
                current_hidden_states = current_expert_output * topk_weight[token_idx, top_k_pos, None]
                final_flat.index_add_(0, token_idx, current_hidden_states.to(final_flat.dtype))

                if return_hidden_states:
                    layer_expert_outputs[token_idx, top_k_pos] = current_expert_output.to(layer_expert_outputs.dtype)

            # 4. Shared experts path, as in Glm4MoeMoE.forward
            shared_out = layer.mlp.shared_experts(hidden_state)  # (B_, S_, D_)
            hidden_state = final_flat.view(B_, S_, D_).to(hidden_state.dtype) + shared_out

            # 5. Sort gate tensors by descending weight for analysis
            topk_ids_sorted, topk_weight_sorted, layer_expert_outputs_sorted = _sort_gate_tensors(
                topk_ids.detach(),
                topk_weight.detach(),
                layer_expert_outputs.detach() if return_hidden_states else None,
            )

            all_topk_experts.append(topk_ids_sorted.cpu())
            all_topk_weights.append(topk_weight_sorted.cpu().to(torch.float32))

            if return_hidden_states:
                all_pre_mlp_hidden_states.append(moe_hidden_state_flat.detach().cpu())
                all_router_logits.append(router_logits.detach().cpu())
                all_expert_outputs.append(layer_expert_outputs_sorted.cpu())

        else: # Dense layer
            hidden_state = layer.mlp(hidden_state)

        hidden_state = residual + hidden_state

        if return_hidden_states and isinstance(layer.mlp, Glm4MoeMoE):
            all_hidden_states.append(hidden_state.view(-1, D_).detach().cpu())

    ##### LM head  #####
    hidden_state = _move_device(hidden_state, model.model.norm.weight.device)
    hidden_state = model.model.norm(hidden_state)
    hidden_state = _move_device(hidden_state, model.lm_head.weight.device)
    logits = model.lm_head(hidden_state).to(input_ids.device)

    return {
        'logits': logits,
        'all_topk_experts': all_topk_experts,
        'all_topk_weights': all_topk_weights,
        'all_pre_mlp_hidden_states': all_pre_mlp_hidden_states,
        'all_router_logits': all_router_logits,
        'all_hidden_states': all_hidden_states,
        'all_expert_outputs': all_expert_outputs
    }



############## THE HF v5.0.0 implemenetation of GLM4 is broken and does not support FP8 weights correctly .################
from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeNaiveMoe

def patched_naive_moe_forward(self, hidden_states, top_k_index, top_k_weights):
    # NOTE: Monkeypatch for GLM-4.x MoE experts when using FP8 checkpoints (e.g. zai-org/GLM-4.5-Air-FP8)
    #
    # Context:
    # - In Transformers v5, GLM-4 MoE experts are implemented by `Glm4MoeNaiveMoe`.
    # - For FP8 models like GLM-4.5-Air-FP8, the expert weights
    #     self.gate_up_proj / self.down_proj
    #   are stored as float8 (Float8_e4m3fn).
    # - When you load the model with torch_dtype=torch.bfloat16, the activations are bfloat16
    #   but the expert weights remain float8. The stock v5 code then does:
    #
    #       gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
    #
    #   which is a BF16 (activations) x FP8 (weights) matmul, and PyTorch rightfully errors:
    #       "expected mat1 and mat2 to have the same dtype, but got: bfloat16 != float8_e4m3fn"
    #
    # - In v4 this worked because the internals were different and didn’t expose the raw FP8
    #   tensors directly to a plain F.linear call in this way.
    #
    # What this patch does:
    # - We override Glm4MoeNaiveMoe.forward to upcast the FP8 expert weights to the
    #   activation dtype (BF16 / FP16 / FP32) *right before* the matmul.
    # - Semantically, this treats FP8 as a compressed storage format and does the compute
    #   in a normal float dtype. That:
    #     * fixes the runtime dtype mismatch,
    #     * keeps the MoE routing / math identical apart from the upcast,
    #     * works for both FP8 and non-FP8 checkpoints (the .to(...) is a no-op for non-FP8).
    #
    # Important:
    # - This must be applied before you run any forward passes:
    #       Glm4MoeNaiveMoe.forward = patched_naive_moe_forward
    # - If you have a custom reverse-engineered forward (e.g. run_glm4moe_return_topk),
    #   you should mirror the same `.to(current_state.dtype)` casts there too so that
    #   your custom path matches the patched HF behavior.
    final_hidden_states = torch.zeros_like(hidden_states)

    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue

        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]  # (ntoks, D)

        # This is the key: upcast FP8 weights to the activation dtype
        w_gate = self.gate_up_proj[expert_idx].to(current_state.device, current_state.dtype)
        gate, up = torch.nn.functional.linear(current_state, w_gate).chunk(2, dim=-1)

        current_hidden_states = self.act_fn(gate) * up

        w_down = self.down_proj[expert_idx].to(current_state.device, current_state.dtype)
        current_hidden_states = torch.nn.functional.linear(current_hidden_states, w_down)

        current_hidden_states = (
            current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        )

        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )

    return final_hidden_states

