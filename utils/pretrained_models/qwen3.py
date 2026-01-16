import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

@torch.no_grad()
def run_qwen3_return_topk(model, input_ids, attention_mask, return_hidden_states: bool = False):
    """
    Reverse-engineered forward pass for dense Qwen3 (HF v4.57.3).

    Params:
        @model: Qwen3ForCausalLM
        @input_ids: (B, N)
        @attention_mask: (B, N) OR already-prepared dict with keys {"full_attention", "sliding_attention"} (4D masks)
        @return_hidden_states: if True, collects per-layer (BN, D) tensors

    Returns:
        Same MoE-compatible dictionary format you’ve been using.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Qwen3Model
    base = model.model

    # Move attention_mask to device (support either 2D or dict of masks)
    if isinstance(attention_mask, dict):
        causal_mask_mapping = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in attention_mask.items()
        }
        attention_mask_2d = None
    else:
        attention_mask_2d = attention_mask.to(device) if attention_mask is not None else None
        causal_mask_mapping = None

    # Token embeddings
    inputs_embeds = base.embed_tokens(input_ids)  # (B, N, D)
    B, N, D = inputs_embeds.shape

    # No-cache path
    past_key_values = None
    cache_position = torch.arange(0, N, device=inputs_embeds.device)
    position_ids = cache_position.unsqueeze(0)

    # Build the same causal_mask_mapping as Qwen3Model.forward does (if not already provided)
    if causal_mask_mapping is None:
        mask_kwargs = dict(
            config=base.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask_2d,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
        has_sliding_layers = getattr(base, "has_sliding_layers", ("sliding_attention" in base.config.layer_types))
        if has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    # Shared RoPE
    position_embeddings = base.rotary_emb(inputs_embeds, position_ids)  # (cos, sin)

    hidden_states = inputs_embeds

    # API-compat: dense => empty MoE fields
    all_topk_experts = []
    all_topk_weights = []
    all_router_logits = []
    all_expert_outputs = []
    all_pre_mlp_hidden_states = []
    all_hidden_states = []

    # Unroll decoder layers (matches Qwen3DecoderLayer.forward)
    for layer in base.layers[: base.config.num_hidden_layers]:
        attn_mask = causal_mask_mapping[layer.attention_type]

        # ---- Attention ----
        residual = hidden_states
        hs_norm = layer.input_layernorm(hidden_states)

        attn_out, _ = layer.self_attn(
            hidden_states=hs_norm,
            position_embeddings=position_embeddings,
            attention_mask=attn_mask,
            past_key_values=None,
            cache_position=cache_position,
        )
        hidden_states = residual + attn_out

        # ---- MLP ----
        residual = hidden_states
        pre_mlp = layer.post_attention_layernorm(hidden_states)
        if return_hidden_states:
            all_pre_mlp_hidden_states.append(pre_mlp.reshape(-1, D).detach().cpu())

        mlp_out = layer.mlp(pre_mlp)
        hidden_states = residual + mlp_out

        if return_hidden_states:
            all_hidden_states.append(hidden_states.reshape(-1, D).detach().cpu())

    # Final norm + LM head
    hidden_states = base.norm(hidden_states)
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
