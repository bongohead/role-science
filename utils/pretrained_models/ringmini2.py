"""
Reversed engineered forward pass for Qwen
- Supports Ring-mini-2.0
- See https://huggingface.co/inclusionAI/Ring-mini-2.0/blob/main/modeling_bailing_moe_v2.py
"""
import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from ._pretrained_helpers import _sort_gate_tensors

@torch.no_grad()
def run_ringmini2_return_topk(model, input_ids, attention_mask, return_hidden_states=False):
    """
    Params:
        @model: A model of class `BailingMoeV2ForCausalLM` (Ring-Mini-2.0).
        @input_ids: A (B, N) tensor of input IDs on the same device as `model`.
        @attention_mask: A (B, N) tensor of mask indicators on the same device as `model`.
        @return_hidden_states: Boolean; whether to return hidden_states themselves.

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
    input_embeds = model.model.word_embeddings(input_ids)  # (B, N, D)
    B, N, D = input_embeds.shape

    position_ids = torch.arange(0, N, device=input_embeds.device).unsqueeze(0)  # (1, N)
    if getattr(model.model, '_use_flash_attention_2', False):  # FA2 path expects a 2D padding mask or None
        causal_mask = attention_mask if (attention_mask is not None and (attention_mask == 0).any()) else None
    elif getattr(model.model, '_use_sdpa', False):
        causal_mask = _prepare_4d_causal_attention_mask_for_sdpa(attention_mask, (B, N), input_embeds, 0)
    else:
        causal_mask = _prepare_4d_causal_attention_mask(attention_mask, (B, N), input_embeds, 0)
    position_embeddings = model.model.rotary_emb(input_embeds, position_ids)

    hidden_state = input_embeds

    all_topk_experts = []
    all_topk_weights = []
    all_pre_mlp_hidden_states = []
    all_router_logits = []
    all_hidden_states = []
    all_expert_outputs = []

    # Exclude optional MTP layers that come after main decoder - we'll slice them off
    num_mtp = getattr(model.model, 'num_nextn_predict_layers', 0) or 0
    layers = model.model.layers[:-num_mtp] if num_mtp > 0 else model.model.layers

    for layer in layers:
        # SA
        residual = hidden_state
        hidden_state = layer.input_layernorm(hidden_state)
        attn_out, _, _ = layer.attention(
            hidden_states=hidden_state,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        hidden_state = residual + attn_out  # residual add

        # MLP
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)

        # Determine if this is a sparse MoE layer (vs dense MLP)
        is_moe = hasattr(layer.mlp, 'gate') and hasattr(layer.mlp, 'experts')

        # Dense layer path (first_k_dense_replace early layers)
        if not is_moe:
            dense_out = layer.mlp(hidden_state)  # (B, N, D)
            hidden_state = residual + dense_out.to(residual.device)
            continue

        # Pre-MLP Snapshot
        if return_hidden_states:
            all_pre_mlp_hidden_states.append(hidden_state.view(-1, D).detach().cpu())

        ####### BailingMoeV2SparseMoeBlock  (parity path without moe_infer)
        moe_in = hidden_state
        BN = B * N
        moe_hidden_state = moe_in.view(-1, D)  # (BN, D)

        # Router - group-limited topk
        # - topk_idx: (BN, K) indices
        # - topk_weight: (BN, K), renormed & scaled (fp32)
        # - router_logits: (BN, E) pre-sigmoid
        topk_idx, topk_weight, router_logits = layer.mlp.gate(moe_in)
        K = topk_idx.shape[1]

        # Build per-token, per-rank expert outputs in model dtype, then do fp32 weight-and-sum
        bnkd = torch.empty((BN, K, D), dtype=moe_hidden_state.dtype, device=moe_hidden_state.device)

        # Also collect raw expert outputs for analysis if requested
        if return_hidden_states:
            layer_expert_outputs = torch.empty((BN, K, D), dtype=moe_hidden_state.dtype, device=moe_hidden_state.device)

        # Fill bnkd at (token, rank) positions for each expert (deterministic assignment)
        # (micro-perf: iterate only experts actually hit)
        for expert_id in torch.unique(topk_idx).tolist():
            mask_e = (topk_idx == expert_id)               # (BN, K)
            if not mask_e.any():
                continue
            token_idx, rank_idx = torch.where(mask_e)      # rows=tokens, cols=ranks
            expert_out = layer.mlp.experts[expert_id](moe_hidden_state[token_idx])  # (S, D), bf16
            bnkd[token_idx, rank_idx] = expert_out.to(bnkd.dtype)
            if return_hidden_states:
                layer_expert_outputs[token_idx, rank_idx] = expert_out.to(layer_expert_outputs.dtype)

        # Weight & reduce in fp32; cast back once (matches reference)
        final_hidden_states = (bnkd.to(topk_weight.dtype) * topk_weight.unsqueeze(-1)).sum(dim=1).to(bnkd.dtype)
        final_hidden_states = final_hidden_states.view(B, N, D)

        # Add back shared experts
        if getattr(layer.mlp, 'shared_experts', None) is not None:
            final_hidden_states = final_hidden_states + layer.mlp.shared_experts(moe_in)

        #######
        hidden_state = residual + final_hidden_states

        # Sort only for reporting (does not affect computation)
        if return_hidden_states:
            topk_idx, topk_weight, layer_expert_outputs = _sort_gate_tensors(
                topk_idx.detach(), topk_weight.detach(), layer_expert_outputs.detach()
            )
        else:
            topk_idx, topk_weight, _ = _sort_gate_tensors(topk_idx.detach(), topk_weight.detach())

        all_topk_experts.append(topk_idx.cpu())
        all_topk_weights.append(topk_weight.cpu().to(torch.float32))

        if return_hidden_states:
            all_router_logits.append(router_logits.detach().cpu())
            all_hidden_states.append(hidden_state.view(-1, D).detach().cpu())
            all_expert_outputs.append(layer_expert_outputs.cpu())

    hidden_state = model.model.norm(hidden_state)
    logits = model.lm_head(hidden_state).float()

    return {
        'logits': logits,
        'all_topk_experts': all_topk_experts,
        'all_topk_weights': all_topk_weights,
        'all_pre_mlp_hidden_states': all_pre_mlp_hidden_states,
        'all_router_logits': all_router_logits,
        'all_hidden_states': all_hidden_states,
        'all_expert_outputs': all_expert_outputs
    }
