"""
Reverse‑engineered forward pass for EssentialAI/rnj-1-instruct
(dense Gemma3ForCausalLM / Gemma3TextModel).

- Text‑only (no images).
- No hooks.
- Calls submodules directly (embedding -> masks/rope -> per-layer attention+MLP -> final norm -> lm_head).
- Returns the same dict keys as your MoE runners, with MoE-specific values as empty dummies.
"""

import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


@torch.no_grad()
def run_rnj1_return_topk(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    return_hidden_states: bool = False,
):
    """
    Params:
        @model:
            A HuggingFace `Gemma3ForCausalLM` (as used by EssentialAI/rnj-1-instruct).

        @input_ids:
            (B, N) token IDs on the same device as `model`.

        @attention_mask:
            Usually a (B, N) padding mask (1 = keep, 0 = pad).
            (HF Gemma3 also supports passing a dict mapping of masks, but typical usage is 2D.)

        @return_hidden_states:
            If True, returns per-layer tensors:
              - `all_pre_mlp_hidden_states`: list (len = #layers) of (B*N, D) tensors
                right before the MLP (after pre_feedforward_layernorm).
              - `all_hidden_states`: list (len = #layers) of (B*N, D) tensors
                after the full layer (attn + mlp + residuals).

    Returns:
        A dict with keys:
          - `logits`: (B, N, V)
          - `all_topk_experts`: []             (dense dummy)
          - `all_topk_weights`: []             (dense dummy)
          - `all_pre_mlp_hidden_states`: list  (optional; empty if return_hidden_states=False)
          - `all_router_logits`: []            (dense dummy)
          - `all_hidden_states`: list          (optional; empty if return_hidden_states=False)
          - `all_expert_outputs`: []           (dense dummy)
    """
    # Base text model (Gemma3TextModel)
    text_model = model.model

    # 1) Token embeddings (Gemma3 uses scaled embeddings; this module does it internally)
    hidden_states = model.get_input_embeddings()(input_ids)  # (B, N, D)
    B, N, D = hidden_states.shape

    # 2) Position / cache indices (no kv-cache in this analysis forward)
    cache_position = torch.arange(0, N, device=hidden_states.device)
    position_ids = cache_position.unsqueeze(0)  # (1, N)

    # 3) Attention masks: Gemma3TextModel builds a dict mapping for (full_attention, sliding_attention)
    #    unless a dict is already provided.
    causal_mask_mapping = attention_mask
    if not isinstance(causal_mask_mapping, dict):
        mask_kwargs = dict(
            config=text_model.config,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }

    # 4) RoPE cos/sin per layer type (Gemma3 supports different RoPE per type)
    #    (Matches the official code pattern: fill a dict keyed by layer_type.)
    position_embeddings = {}
    for layer_type in text_model.config.layer_types:
        position_embeddings[layer_type] = text_model.rotary_emb(hidden_states, position_ids, layer_type)

    # 5) Layer-by-layer forward
    all_pre_mlp_hidden_states = []
    all_hidden_states = []

    for layer in text_model.layers[: text_model.config.num_hidden_layers]:
        layer_type = layer.attention_type  # "full_attention" or "sliding_attention"
        layer_attn_mask = causal_mask_mapping[layer_type]
        layer_pos_emb = position_embeddings[layer_type]

        # --- Attention block (matches Gemma3DecoderLayer.forward) ---
        residual = hidden_states
        normed = layer.input_layernorm(hidden_states)

        attn_out, _ = layer.self_attn(
            hidden_states=normed,
            position_embeddings=layer_pos_emb,
            attention_mask=layer_attn_mask,
            position_ids=position_ids,
            past_key_values=None,
            cache_position=cache_position,
        )
        attn_out = layer.post_attention_layernorm(attn_out)
        hidden_states = residual + attn_out

        # --- MLP block ---
        residual = hidden_states
        mlp_in = layer.pre_feedforward_layernorm(hidden_states)

        if return_hidden_states:
            all_pre_mlp_hidden_states.append(mlp_in.reshape(-1, mlp_in.shape[-1]).detach().cpu())

        mlp_out = layer.mlp(mlp_in)
        mlp_out = layer.post_feedforward_layernorm(mlp_out)
        hidden_states = residual + mlp_out

        if return_hidden_states:
            all_hidden_states.append(hidden_states.reshape(-1, hidden_states.shape[-1]).detach().cpu())

    # 6) Final norm
    hidden_states = text_model.norm(hidden_states)

    # 7) LM head + final logit softcapping (Gemma3ForCausalLM.forward behavior)
    logits = model.lm_head(hidden_states)  # (B, N, V)
    softcap = getattr(model.config, "final_logit_softcapping", None)
    if softcap is not None:
        logits = logits / softcap
        logits = torch.tanh(logits)
        logits = logits * softcap

    return {
        "logits": logits,
        "all_topk_experts": [],  # dense: dummy
        "all_topk_weights": [],  # dense: dummy
        "all_pre_mlp_hidden_states": all_pre_mlp_hidden_states,
        "all_router_logits": [],  # dense: dummy
        "all_hidden_states": all_hidden_states,
        "all_expert_outputs": [],  # dense: dummy
    }
