import torch

@torch.no_grad()
def run_jamba_return_topk(model, input_ids, attention_mask=None, return_hidden_states: bool = False):
    """
    Reverse-engineered forward pass for JambaForCausalLM (HF v4.57.3-style code you pasted).

    Params:
        model: JambaForCausalLM
        input_ids: (B, N)
        attention_mask: (B, N) with 1=keep, 0=pad (optional)
        return_hidden_states: whether to return per-layer pre-FFN and post-layer activations (BN, D) on CPU

    Returns (MoE-compatible dict):
        - logits: (B, N, V)
        - all_topk_experts: list len = #layers, each is (B*N, top_k) LongTensor on CPU or None (dense FFN layers)
        - all_topk_weights: list len = #layers, each is (B*N, top_k) FloatTensor on CPU or None
        - all_router_logits: list len = #layers, each is (B*N, num_experts) FloatTensor on CPU or None
        - all_pre_mlp_hidden_states: optional list of (B*N, D) CPU tensors (pre-FFN norm output)
        - all_hidden_states: optional list of (B*N, D) CPU tensors (post-layer output)
        - all_expert_outputs: [] (kept for API compatibility)
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    base = model.model  # JambaModel

    # 1) Token embeddings
    hidden_states = base.embed_tokens(input_ids)  # (B, N, D)
    B, N, D = hidden_states.shape

    # 2) Positions / masks (mirror JambaModel.forward)
    cache_position = torch.arange(N, device=hidden_states.device)
    position_ids = cache_position.unsqueeze(0)  # (1, N) in the official code

    # Attention layers want the "causal_mask" (4D for eager/sdpa, None or 2D for FA2),
    # Mamba layers want the "mamba_mask" (2D or None).
    causal_mask = base._update_causal_mask(attention_mask, hidden_states, cache_position)
    mamba_mask = base._update_mamba_mask(attention_mask, cache_position)

    # 3) Collectors
    all_topk_experts = []
    all_topk_weights = []
    all_router_logits = []
    all_expert_outputs = []  # (kept for compatibility; not filled)

    all_pre_mlp_hidden_states = []
    all_hidden_states = []

    # 4) Unroll each layer (attention or mamba) + FFN (dense or MoE)
    for layer in base.layers:
        is_mamba_layer = hasattr(layer, "mamba")  # JambaMambaDecoderLayer vs JambaAttentionDecoderLayer
        layer_mask = mamba_mask if is_mamba_layer else causal_mask

        # ---- Operator block (Attention or Mamba) ----
        residual = hidden_states
        hs_norm = layer.input_layernorm(hidden_states)

        if is_mamba_layer:
            op_out = layer.mamba(
                hidden_states=hs_norm,
                cache_params=None,
                attention_mask=layer_mask,  # 2D or None
            )
            hidden_states = residual + op_out
        else:
            attn_out, _, _ = layer.self_attn(
                hidden_states=hs_norm,
                attention_mask=layer_mask,  # 4D mask OR None/2D depending on attn impl
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
            )
            hidden_states = residual + attn_out

        # ---- FFN block (Dense MLP or Sparse MoE) ----
        residual = hidden_states
        pre_ff = layer.pre_ff_layernorm(hidden_states)
        if return_hidden_states:
            all_pre_mlp_hidden_states.append(pre_ff.reshape(-1, D).detach().cpu())

        ff_out = layer.feed_forward(pre_ff)

        # MoE layers return (hidden_states, router_logits); dense returns just hidden_states
        if isinstance(ff_out, tuple):
            ff_hidden, router_logits = ff_out  # router_logits: (B*N, num_experts)

            # Recompute top-k the same way JambaSparseMoeBlock does
            routing_probs = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
            topk_w, topk_e = torch.topk(routing_probs, layer.feed_forward.top_k, dim=-1)
            topk_w = topk_w.to(pre_ff.dtype)

            all_topk_experts.append(topk_e.detach().cpu())
            all_topk_weights.append(topk_w.detach().cpu())
            all_router_logits.append(router_logits.detach().cpu())
        else:
            ff_hidden = ff_out
            all_topk_experts.append(None)
            all_topk_weights.append(None)
            all_router_logits.append(None)

        hidden_states = residual + ff_hidden

        if return_hidden_states:
            all_hidden_states.append(hidden_states.reshape(-1, D).detach().cpu())

    # 5) Final norm + LM head (mirror JambaModel + JambaForCausalLM)
    hidden_states = base.final_layernorm(hidden_states)
    logits = model.lm_head(hidden_states)  # (B, N, V)

    return {
        "logits": logits,
        "all_topk_experts": all_topk_experts,
        "all_topk_weights": all_topk_weights,
        "all_pre_mlp_hidden_states": all_pre_mlp_hidden_states,
        "all_router_logits": all_router_logits,
        "all_hidden_states": all_hidden_states,
        "all_expert_outputs": all_expert_outputs,
    }
