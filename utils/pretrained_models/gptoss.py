"""
Reversed engineered forward pass for GPT-OSS models
- Supports GPT-OSS-20B and GPT-OSS-120B
- See https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py
- Seperate MXFP4 path: https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/mxfp4.py
- Note that `all_expert_outputs` returns null under MXFP4 as the Triton kernel doesn't expose them.
"""
import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

@torch.no_grad()
def run_gptoss_return_topk(model, input_ids, attention_mask, return_hidden_states: bool = False):
    """
    Params:
        @model: A model of class `GptOssForCausalLM`.
        @input_ids: A (B, N) tensor of input IDs on the same device as `model`.
        @attention_mask: A (B, N) tensor of mask indicators on the same device as `model`.
        @return_hidden_states: Boolean; whether to return hidden_states themselves.

    Returns:
        dict with keys (identical to your Qwen helper):
        - `logits`: (B, N, V) LM outputs
        - `all_topk_experts`: List (len = # MoE layers) of (BN, topk) expert IDs tensors
        - `all_topk_weights`: List (len = # MoE layers) of (BN, topk) expert weight tensors
        - `all_pre_mlp_hidden_states`: (optional) List (len = # MoE layers) of (BN, D) pre-MLP activations
        - `all_router_logits: (optional) List (len = # MoE layers) of (BN, n_experts) router *logits*
        - `all_hidden_states`: (optional) List (len = # MoE layers) of (BN, D) post-layer activations
        - `all_expert_outputs`: (optional) List (len = # MoE layers) of (BN, topk, D) pre-weighting expert outputs ****(None for MXFP4 layers)
    """
    input_embeds = model.model.embed_tokens(input_ids)
    B, N, D = input_embeds.shape
    
    cache_position = torch.arange(0, N, device = input_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    # Build both masks; each layer picks one via `attention_type`
    mask_kwargs = {'config': model.model.config, 'input_embeds': input_embeds, 'attention_mask': attention_mask, 'cache_position': cache_position, 'past_key_values': None}
    causal_mask_mapping = {
        'full_attention': create_causal_mask(**mask_kwargs),
        'sliding_attention': create_sliding_window_causal_mask(**mask_kwargs)
    }
    position_embeddings = model.model.rotary_emb(input_embeds, position_ids)

    hidden_state = input_embeds

    all_topk_experts = []
    all_topk_weights = []
    all_pre_mlp_hidden_states = []
    all_router_logits = []
    all_hidden_states = []
    all_expert_outputs = []

    for layer_ix, layer in enumerate(model.model.layers):
        # SA with sinks
        residual = hidden_state
        hidden_state = layer.input_layernorm(hidden_state)
        hidden_state, _ = layer.self_attn(hidden_states = hidden_state, attention_mask = causal_mask_mapping[layer.attention_type], position_ids = position_ids, position_embeddings = position_embeddings)
        hidden_state = residual + hidden_state
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)

        if return_hidden_states:
            all_pre_mlp_hidden_states.append(hidden_state.view(-1, hidden_state.shape[2]).detach().cpu())
        
        ############ MLP ############
        
        BN = B * N
        moe_hidden_state = hidden_state.view(BN, D)  # (BN, D)

        # --- Router: logits -> top-k ids/weights (softmax within top-k), and scatter to full E ---
        router = layer.mlp.router  # GptOssTopKRouter
        router_logits = torch.nn.functional.linear(moe_hidden_state, router.weight, router.bias) # (BN, E)
        top_vals, selected_experts = torch.topk(router_logits, router.top_k, dim = -1, sorted = True)  # (BN, K)
        routing_weights_topk = torch.softmax(top_vals, dim = 1) # (BN, K)

        is_mxfp4 = 'mxfp4' in layer.mlp.experts.__class__.__name__.lower()

        if is_mxfp4:
            # ---- MXFP4 fused path ----
            # 1) Get raw router logits (BN, E) from MLP
            # The patched MXFP4 mlp_forward returns (routed_out, router_logits).
            routed_out, router_logits = layer.mlp(hidden_state)
            if router_logits.dim() == 3: # Unsure about output structure
                router_logits = router_logits.view(BN, -1)

            # 2) Top-k ids/weights (sorted by descending logit); weights are softmaxed within top-k
            top_vals, selected_experts = torch.topk(router_logits, router.top_k, dim = -1, sorted = True)  # (BN, K)
            routing_weights_topk = torch.softmax(top_vals, dim = 1) # (BN, K)

            # 3) Residual
            hidden_state = residual + routed_out  # routed_out is already (B, N, D)

            # 4) Book-keeping
            all_topk_experts.append(selected_experts.detach().cpu())
            all_topk_weights.append(routing_weights_topk.detach().cpu().to(torch.float32))
            if return_hidden_states:
                all_router_logits.append(router_logits.detach().cpu())
                all_hidden_states.append(hidden_state.view(BN, D).detach().cpu())
                all_expert_outputs.append(None) # not available from the fused kernel

        else:
            # Router logits and top-k (sorted=True)
            router_logits = F.linear(moe_hidden_state, router.weight, router.bias) # (BN, E)
            top_vals, selected_experts = torch.topk(router_logits, router.top_k, dim = -1, sorted = True) # (BN, K)
            routing_weights_topk = torch.softmax(top_vals, dim = 1) # (BN, K)
            routing_scores_full = torch.zeros_like(router_logits).scatter_(1, selected_experts, routing_weights_topk)

            experts = layer.mlp.experts
            E = experts.num_experts
            I = experts.expert_dim # Expert MLP dim

            # gate_up: (BN, E, 2H) = (BN, D) @ (E, D, 2H)
            gate_up = torch.einsum('bd,edh->beh', moe_hidden_state, experts.gate_up_proj)
            gate_up = gate_up + experts.gate_up_proj_bias.unsqueeze(0) # (1, E, 2I)

            gate = gate_up[..., ::2] # (BN, E, I)
            up = gate_up[..., 1::2] # (BN, E, I)
            gate = gate.clamp(min = None, max = experts.limit)
            up = up.clamp(min = -experts.limit, max = experts.limit)
            glu = gate * torch.sigmoid(gate * experts.alpha)
            gated_output = (up + 1) * glu # (BN, E, I)

            # out_all: (BN, E, D) = (BN, E, I) @ (E, I, D) + bias
            out_all = torch.einsum('beh,ehd->bed', gated_output, experts.down_proj)
            out_all = out_all + experts.down_proj_bias.unsqueeze(0) # (BN, E, D)

            final_hidden_states = (out_all * routing_scores_full.unsqueeze(-1)).sum(dim = 1)  # (BN, D)
            
            if return_hidden_states:
                gather_idx = selected_experts.unsqueeze(-1).expand(BN, router.top_k, D)
                layer_expert_outputs = out_all.gather(dim=1, index=gather_idx)             # (BN, K, D)
                
            hidden_state = residual + final_hidden_states.view(B, N, D)

            all_topk_experts.append(selected_experts.detach().cpu())
            all_topk_weights.append(routing_weights_topk.detach().cpu().to(torch.float32))
            if return_hidden_states:
                all_router_logits.append(router_logits.detach().cpu()) # raw logits over E
                all_hidden_states.append(hidden_state.view(-1, D).detach().cpu()) # post-layer states
                all_expert_outputs.append(layer_expert_outputs.detach().cpu())

        ############ END MLP ############

    hidden_state = model.model.norm(hidden_state)
    logits = model.lm_head(hidden_state)

    return {
        "logits": logits,
        "all_topk_experts": all_topk_experts,
        "all_topk_weights": all_topk_weights,
        "all_pre_mlp_hidden_states": all_pre_mlp_hidden_states,
        "all_router_logits": all_router_logits,
        "all_hidden_states": all_hidden_states,
        "all_expert_outputs": all_expert_outputs,
    }
