"""
Reversed engineered forward pass for Liquid FM 2.5 (Lfm2) - HF v4.57.3
- Dense hybrid (full-attention + short-conv) model
- Returns MoE-compatible fields as empty lists (API compatibility)
"""

import torch
from transformers.masking_utils import create_causal_mask

@torch.no_grad()
def run_lfm2_return_topk(
    model,
    input_ids,
    attention_mask,
    return_hidden_states: bool = False,
):
    """
    Params:
        @model: A model of class `Lfm2ForCausalLM`.
        @input_ids: (B, N) tensor of token IDs on the same device as `model`.
        @attention_mask: (B, N) tensor (1=keep, 0=pad) on the same device as `model`.
        @return_hidden_states: if True, collect per-layer (BN, D) tensors.

    Returns:
        A dict with keys:
        - `logits`: (B, N, V)
        - `all_topk_experts`: [] (dense/hybrid model: kept for API compatibility)
        - `all_topk_weights`: [] (dense/hybrid model: kept for API compatibility)
        - `all_pre_mlp_hidden_states`: List[(BN, D)] inputs to FFN (after ffn_norm), if requested
        - `all_router_logits`: [] (dense/hybrid model: kept for API compatibility)
        - `all_hidden_states`: List[(BN, D)] post-layer hidden states, if requested
        - `all_expert_outputs`: [] (dense/hybrid model: kept for API compatibility)
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device) if attention_mask is not None else None

    base_model = model.model  # Lfm2Model

    # 1) Token embeddings (matches Lfm2Model.forward)
    inputs_embeds = model.get_input_embeddings()(input_ids)  # (B, N, D)
    B, N, D = inputs_embeds.shape

    # 2) cache_position / position_ids (prefill path: no cache)
    past_key_values = None
    cache_position = torch.arange(0, N, device=inputs_embeds.device)
    position_ids = cache_position.unsqueeze(0)

    # 3) Causal mask (matches Lfm2Model.forward call signature)
    causal_mask = create_causal_mask(
        config=base_model.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    # 4) Shared RoPE embeddings (matches: position_embeddings = self.pos_emb(hidden_states, position_ids))
    hidden_states = inputs_embeds
    position_embeddings = base_model.pos_emb(hidden_states, position_ids)  # (cos, sin)

    # MoE-compatible outputs (empty for this model)
    all_topk_experts: list = []
    all_topk_weights: list = []
    all_router_logits: list = []
    all_expert_outputs: list = []

    all_pre_mlp_hidden_states: list = []
    all_hidden_states: list = []

    # 5) Unroll decoder layers (matches Lfm2DecoderLayer.forward logic)
    for layer in base_model.layers[: base_model.config.num_hidden_layers]:
        # Operator block (attention OR short-conv)
        residual = hidden_states
        op_in = layer.operator_norm(hidden_states)

        if layer.is_attention_layer:
            op_out, _ = layer.self_attn(
                hidden_states=op_in,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                past_key_values=None,
                cache_position=cache_position,
                # keep parity with signature; position_ids is unused by attention here but harmless
                position_ids=position_ids,
            )
        else:
            op_out = layer.conv(
                hidden_states=op_in,
                past_key_values=None,
                cache_position=cache_position,
                attention_mask=causal_mask,
            )

        hidden_states = residual + op_out

        # Feed-forward block
        ffn_in = layer.ffn_norm(hidden_states)
        if return_hidden_states:
            all_pre_mlp_hidden_states.append(ffn_in.reshape(-1, D).detach().cpu())

        ffn_out = layer.feed_forward(ffn_in)
        hidden_states = hidden_states + ffn_out

        if return_hidden_states:
            all_hidden_states.append(hidden_states.reshape(-1, D).detach().cpu())

    # 6) Final norm + LM head (matches Lfm2Model + Lfm2ForCausalLM behavior)
    hidden_states = base_model.embedding_norm(hidden_states)
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
