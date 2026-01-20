"""
Reversed engineered forward pass for dense GLM-4.6v V models
- Supports GLM-4.6V-Flash (Glm46VForConditionalGeneration) and GLM-4.x V (Glm4vForConditionalGeneration)
- See: modeling_glm46v.py and modeling_glm4v.py
"""
import torch
from transformers.masking_utils import create_causal_mask

@torch.no_grad()
def run_glm46v_return_topk(model, input_ids, attention_mask, pixel_values = None, pixel_values_videos = None, image_grid_thw = None, video_grid_thw = None, return_hidden_states: bool = False):
    """
    Params:
        @model: A model of class `Glm4vModel`.
        @input_ids: A (B, N) tensor of input IDs on the same device as `model`.
        @attention_mask: A (B, N) tensor of mask indicators on the same device as `model`.
        @pixel_values: (L_img, C, P, P, ...) tensor of image pixels, as produced by the GLM-4v processor; may be None (text only).
        @pixel_values_videos: (L_vid, C, P, P, ...) tensor of video pixels; may be None.
        @image_grid_thw: (num_images, 3) tensor with (T, H, W) feature-grid sizes per image; may be None.
        @video_grid_thw: (num_videos, 3) tensor with (T, H, W) feature-grid sizes per video; may be None.
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
    input_embeds = model.get_input_embeddings()(input_ids)
    B, N, D = input_embeds.shape

    base_model = model.model # Glm4vModel/Glm46VModel
    text_model = base_model.language_model  # Glm4vTextModel

    ### Images following Glm4vModel.forward: get_image_features -> placeholder mask -> masked_scatter
    if pixel_values is not None and pixel_values.numel() > 0:
        if image_grid_thw is None:
            raise ValueError("pixel_values provided but image_grid_thw is None.")
        pixel_values = pixel_values.to(model.device)
        image_grid_thw = image_grid_thw.to(model.device)

        # list/tuple of (tokens_per_image, D)
        image_embeds_list = base_model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds_list, dim = 0).to(device = model.device, dtype = input_embeds.dtype)

        image_mask, _ = base_model.get_placeholder_mask(input_ids = input_ids, inputs_embeds = input_embeds, image_features = image_embeds)
        input_embeds = input_embeds.masked_scatter(image_mask, image_embeds)

    ### Videos
    if pixel_values_videos is not None and pixel_values_videos.numel() > 0:
        if video_grid_thw is None:
            raise ValueError("pixel_values_videos provided but video_grid_thw is None.")
        pixel_values_videos = pixel_values_videos.to(model.device)
        video_grid_thw = video_grid_thw.to(model.device)

        video_embeds_list = base_model.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds_list, dim=0).to(device = model.device, dtype = input_embeds.dtype)

        _, video_mask = base_model.get_placeholder_mask(input_ids = input_ids, inputs_embeds = input_embeds, video_features = video_embeds)
        input_embeds = input_embeds.masked_scatter(video_mask, video_embeds)


    # RoPE position ids (3D multimodal index)
    # Use the same helper as the official forward so text-only vs vision matches behavior; attention_mask for rope index is always 2D full-attention mask here.
    position_ids, rope_deltas = base_model.get_rope_index(
        input_ids = input_ids,
        image_grid_thw = image_grid_thw,
        video_grid_thw = video_grid_thw,
        attention_mask = attention_mask
    )
    # position_ids: (3, B, N), rope_deltas: (B, 1) - unused here (image caching) but kept for parity.

    # Same as Glm4vTextModel.forward: cache_position + causal mask
    cache_position = torch.arange(0, N, device = input_embeds.device)
    causal_mask = create_causal_mask(text_model.config, input_embeds, attention_mask, cache_position, None, position_ids = None)  # (B, 1, N, N) see src - no position_ids needs to be passed generally
    position_embeddings = text_model.rotary_emb(input_embeds, position_ids)
    cos, sin = position_embeddings

    hidden_state = input_embeds

    all_topk_experts = [] # dense model: kept for API compatibility
    all_topk_weights = [] # dense model: kept for API compatibility
    all_router_logits = [] # dense model: kept for API compatibility
    all_expert_outputs = [] # dense model: kept for API compatibility
    all_pre_mlp_hidden_states = []
    all_hidden_states = []

    for layer in text_model.layers:
        residual = hidden_state
        hidden_state = layer.input_layernorm(hidden_state)
        hidden_state, _ = layer.self_attn(
            hidden_states = hidden_state,
            position_embeddings = (cos, sin),
            attention_mask = causal_mask,
            position_ids = position_ids,
            past_key_values = None,
            use_cache = False,
            cache_position = cache_position
        )
        hidden_state = layer.post_self_attn_layernorm(hidden_state)
        hidden_state = residual + hidden_state
        
        ### MLP
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)
        if return_hidden_states:
            all_pre_mlp_hidden_states.append(hidden_state.reshape(-1, hidden_state.shape[2]).detach().cpu())
        hidden_state = layer.mlp(hidden_state)
        hidden_state = layer.post_mlp_layernorm(hidden_state)
        hidden_state = residual + hidden_state

        if return_hidden_states:
            all_hidden_states.append(hidden_state.reshape(-1, hidden_state.shape[2]).detach().cpu())

    # Final RMSNorm
    hidden_state = text_model.norm(hidden_state)
    logits = model.lm_head(hidden_state)  # (B, N, V)

    return {
        'logits': logits,
        'all_topk_experts': all_topk_experts,
        'all_topk_weights': all_topk_weights,
        'all_pre_mlp_hidden_states': all_pre_mlp_hidden_states,
        'all_router_logits': all_router_logits,
        'all_hidden_states': all_hidden_states,
        'all_expert_outputs': all_expert_outputs
    }
