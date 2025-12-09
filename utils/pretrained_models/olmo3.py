"""
Reverse engineered forward pass for Olmo3
- Supports Olmo3ForCausalLM models from transformers v5+
- Mirrors transformers.models.olmo3.modeling_olmo3.Olmo3ForCausalLM / Olmo3Model
- No KV cache, no labels, full logits.
- MoE-related outputs are returned as None (Olmo3 is *not* an MoE model).
"""

import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

@torch.no_grad()
def run_olmo3_return_topk(model, input_ids, attention_mask = None, return_hidden_states: bool = False):
    """
    Params:
        model: A model of class `Olmo3ForCausalLM`.
        input_ids: (B, N) tensor of token IDs on the same device as `model`.
        attention_mask: (B, N) attention mask tensor (1 = keep, 0 = pad) on the same device as `model`,
                        or None for no padding.
        return_hidden_states: If True, return per-layer hidden states.

    Returns:
        A dictionary with keys:
        - logits: (B, N, V) LM logits (no slicing, all positions).
        - all_topk_experts: None  (not applicable; Olmo3 is dense, not MoE)
        - all_topk_weights: None  (not applicable)
        - all_pre_mlp_hidden_states:
            * If return_hidden_states: list (len = #layers) of (B*N, D) tensors
              representing the input to the MLP for each layer.
            * Else: empty list.
        - all_router_logits: None (not applicable)
        - all_hidden_states:
            * If return_hidden_states: list (len = #layers) of (B*N, D) tensors
              representing the *output* of each decoder layer (post-FFN + residual).
            * Else: empty list.
        - all_expert_outputs: None (not applicable)
    """
    # ---- 1. Embeddings & basic shapes ----
    # `model` is Olmo3ForCausalLM; its decoder is `model.model` (Olmo3Model)
    decoder = model.model
    config = decoder.config

    input_embeds: torch.Tensor = decoder.embed_tokens(input_ids)  # (B, N, D)
    B, N, D = input_embeds.shape
    device = input_embeds.device

    # ---- 2. Positions / cache_position / position_ids (no cache) ----
    # Mirrors Olmo3Model.forward when past_key_values is None
    cache_position = torch.arange(N, device=device)  # [0, 1, ..., N-1]
    position_ids = cache_position.unsqueeze(0)       # (1, N)

    # ---- 3. Mask creation: full vs sliding attention (per HF code) ----
    # Olmo3Model builds *both* masks and picks per-layer via `layer.self_attn.attention_type`
    mask_kwargs = {
        "config": config,
        "input_embeds": input_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": None,   # no cache in this helper
        "position_ids": position_ids,
    }
    causal_mask_mapping = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }

    # ---- 4. Rotary embeddings (shared across layers) ----
    hidden_states = input_embeds
    position_embeddings = decoder.rotary_emb(hidden_states, position_ids)

    # ---- 5. Per-layer forward, mirroring Olmo3DecoderLayer ----
    all_pre_mlp_hidden_states = []
    all_hidden_states = []

    for layer in decoder.layers:
        # Select the correct attention mask for this layer
        attn_mask = causal_mask_mapping[layer.self_attn.attention_type]

        # --- Self Attention block ---
        residual = hidden_states
        hidden_states, _ = layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states  # post-attn residual

        # This is the input to the MLP for this layer
        if return_hidden_states:
            all_pre_mlp_hidden_states.append(
                hidden_states.reshape(B * N, D).detach().cpu()
            )

        # --- Feed-forward block ---
        residual = hidden_states
        hidden_states = layer.mlp(hidden_states)
        hidden_states = layer.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states  # post-FFN residual

        # This is the output of the layer
        if return_hidden_states:
            all_hidden_states.append(
                hidden_states.reshape(B * N, D).detach().cpu()
            )

    # ---- 6. Final RMSNorm + LM head (same as Olmo3ForCausalLM.forward) ----
    hidden_states = decoder.norm(hidden_states)          # (B, N, D)
    logits = model.lm_head(hidden_states)                # (B, N, V)

    # ---- 7. MoE-related outputs are None for Olmo3 ----
    return {
        "logits": logits,
        "all_topk_experts": None,
        "all_topk_weights": None,
        "all_pre_mlp_hidden_states": all_pre_mlp_hidden_states,
        "all_router_logits": None,
        "all_hidden_states": all_hidden_states,
        "all_expert_outputs": None,
    }
