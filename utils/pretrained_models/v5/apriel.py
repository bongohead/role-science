"""
Reversed engineered forward pass for GLM-4.x MoE
- Supports Apriel-1.6-15B-Thinker (maybe others too)
- See https://huggingface.co/ServiceNow-AI/Apriel-1.6-15b-Thinker/tree/main
"""

import torch
from transformers.masking_utils import create_causal_mask

@torch.no_grad()
def run_apriel_return_topk(model, input_ids, attention_mask, pixel_values = None, image_sizes = None, return_hidden_states: bool = False,):
    """
    Reverse-engineered forward pass for Apriel-1.6-15b-Thinker (Llava + Mistral), without hooks.

    Params:
        @model: A model of class `LlavaForConditionalGeneration`.
        @input_ids: A (B, N) tensor of input IDs on the same device as `model`.
        @attention_mask: A (B, N) tensor of mask indicators on the same device as `model`.
        @pixel_values: (B, 3, H, W, ...) tensor of image pixels, as produced by the Pixtral processor; may be None (text only).
        @image_sizes: Per-image (H, W) tensor for multi-resolution; may be None.
        @return_hidden_states: Boolean; whether to return hidden_states themselves.

    Returns:
        A dictionary with keys:
        - `logits`: (B, N, V) LM outputs
        - `all_topk_experts`: [] (dense model: kept for API compatibility)
        - `all_topk_weights`: [] (dense model: kept for API compatibility)
        - `all_pre_mlp_hidden_states`: (optional) List (len = # layers) of (BN, D) pre-MLP activations
        - `all_router_logits: (optional)  [] (dense model: kept for API compatibility)
        - `all_hidden_states`: (optional) List (len = # layers) of (BN, D) post-layer activations
        - `all_expert_outputs`: (optional)  [] (dense model: kept for API compatibility)
    """
    device = next(model.parameters()).device

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    base_model = model.model            # LlavaModel
    text_model = base_model.language_model  # MistralModel

    # 1) Token embeddings (identical to LlavaModel + MistralModel path)
    input_embeds = model.get_input_embeddings()(input_ids)  # (B, N, D)
    B, N, D = input_embeds.shape

    # 2) Vision → projector → placeholder replacement (exactly like LlavaModel.forward)
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)

        image_features_list = base_model.get_image_features(
            pixel_values=pixel_values,
            # Use config defaults for feature layer & strategy
            vision_feature_layer = None,
            vision_feature_select_strategy = None,
            image_sizes=image_sizes,
        )
        # Concatenate per-image sequences → (total_image_tokens, D)
        image_features = torch.cat(image_features_list, dim = 0).to(
            device=input_embeds.device, dtype=input_embeds.dtype
        )

        special_image_mask = base_model.get_placeholder_mask(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            image_features=image_features,
        )
        input_embeds = input_embeds.masked_scatter(special_image_mask, image_features)

    # 3) Mistral-style causal mask and RoPE (mirrors MistralModel.forward)
    past_key_values = None
    cache_position = torch.arange(0, N, device=input_embeds.device)  # same as past_seen_tokens=0 branch
    position_ids = cache_position.unsqueeze(0)                       # (1, N)

    causal_mask = create_causal_mask(
        config = text_model.config,
        input_embeds = input_embeds,
        attention_mask = attention_mask,
        cache_position = cache_position,
        past_key_values = past_key_values,
        position_ids = position_ids,
    )

    # Shared RoPE cos/sin for all layers
    cos, sin = text_model.rotary_emb(input_embeds, position_ids=position_ids)

    hidden_states = input_embeds

    all_topk_experts: list = []
    all_topk_weights: list = []
    all_router_logits: list = []
    all_expert_outputs: list = []
    all_pre_mlp_hidden_states: list = []
    all_hidden_states: list = []

    # 4) Unroll each MistralDecoderLayer exactly (but without hooks/checkpointing)
    for layer in text_model.layers[: text_model.config.num_hidden_layers]:
        # Self-attention block
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        attn_output, _ = layer.self_attn(
            hidden_states=hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=causal_mask,
            past_key_values=None,
            cache_position=cache_position,
            # Make sure FA2 / advanced backends see the same position_ids they would normally get
            position_ids=position_ids,
        )
        hidden_states = residual + attn_output

        # MLP block
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        if return_hidden_states:
            all_pre_mlp_hidden_states.append(hidden_states.reshape(-1, D).detach().cpu())
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if return_hidden_states:
            all_hidden_states.append(hidden_states.reshape(-1, D).detach().cpu())

    # 5) Final RMSNorm + LM head
    hidden_states = text_model.norm(hidden_states)
    logits = model.lm_head(hidden_states)  # (B, N, V), matches logits_to_keep=0 behavior

    return {
        'logits': logits,
        'all_topk_experts': all_topk_experts,
        'all_topk_weights': all_topk_weights,
        'all_pre_mlp_hidden_states': all_pre_mlp_hidden_states,
        'all_router_logits': all_router_logits,
        'all_hidden_states': all_hidden_states,
        'all_expert_outputs': all_expert_outputs
    }
