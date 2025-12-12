"""
Reverse-engineered forward pass for dense Mistral3 VLMs
- Supports Devstral-Small-2-24B-Instruct
- Replicates a `Mistral3ForConditionalGeneration` with `Ministral3Model` text backbone (modeling_mistral3.py) +  Pixtral vision tower (`modeling_ministral3.py`)
"""
import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

@torch.no_grad()
def run_mistral3_return_topk(model, input_ids, attention_mask, pixel_values = None, image_sizes = None, return_hidden_states = False):
    """
    Params:
        @model: A model of class `Mistral3ForConditionalGeneration`with a `Ministral3Model` text backbone.
        @input_ids: A (B, N) tensor of inputs IDs on the same device as `model`. Should already include any `<image>` placeholder tokens that
            the processor inserts.
        @pixel_values: Optional image tensor as produced by the Mistral-3 / Pixtral processor, typically of shape (L_img, C, H, W).
            If provided, you must also pass `image_sizes`.
        @image_sizes: Optional tensor describing the original image sizes, as returned by the processor (e.g. `inputs.image_sizes`).
            Shape is usually (L_img, 2) with (height, width) for each image. Required if `pixel_values` is not None.
        @attention_mask: A (B, N) tensor of mask indicators on the same device as `model`.
        @return_hidden_states: Boolean; whether to return optional outputs below.

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
    # ------------------------------------------------------------------
    # 0. Basic handles
    # ------------------------------------------------------------------
    # Mistral3ForConditionalGeneration -> Mistral3Model -> Ministral3Model
    base_model = model.model              # Mistral3Model
    text_model = base_model.language_model  # Ministral3Model

    # Token embeddings: (B, N, D)
    input_embeds = model.get_input_embeddings()(input_ids)
    B, N, D = input_embeds.shape

    # ------------------------------------------------------------------
    # 1. Vision path: get_image_features + placeholder replacement
    #    (matches Mistral3Model.forward)
    # ------------------------------------------------------------------
    if pixel_values is not None and pixel_values.numel() > 0:
        if image_sizes is None:
            raise ValueError("`pixel_values` was provided but `image_sizes` is None.")

        pixel_values = pixel_values.to(input_embeds.device)

        # Returns a list/tuple of per‑image tensors (tokens_per_image_i, D_v)
        image_features_list = base_model.get_image_features(
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            vision_feature_layer=None,  # use config.vision_feature_layer
        )

        # Concatenate all image tokens and project to text hidden size
        image_features = torch.cat(image_features_list, dim=0).to(
            device=input_embeds.device,
            dtype=input_embeds.dtype,
        )

        # Find `<image>` placeholders and check feature length
        special_image_mask = base_model.get_placeholder_mask(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            image_features=image_features,
        )

        # Replace placeholder embeddings with vision features
        input_embeds = input_embeds.masked_scatter(special_image_mask, image_features)

    # ------------------------------------------------------------------
    # 2. Causal mask + RoPE indices (matches Ministral3Model.forward)
    #    We do a single full‑sequence pass, no kv cache.
    # ------------------------------------------------------------------
    # cache_position: [0, 1, ..., N-1]
    cache_position = torch.arange(0, N, device=input_embeds.device)

    # position_ids: (1, N)
    position_ids = cache_position.unsqueeze(0)

    # Standard Mistral‑style mask construction
    mask_fn = create_causal_mask if text_model.config.sliding_window is None else create_sliding_window_causal_mask
    causal_mask = mask_fn(
        config=text_model.config,
        input_embeds=input_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )

    # Rotary embeddings (cos, sin) computed once, reused across layers
    hidden_states = input_embeds
    position_embeddings = text_model.rotary_emb(hidden_states, position_ids=position_ids)

    # ------------------------------------------------------------------
    # 3. Decoder stack: explicit layer‑by‑layer forward
    # ------------------------------------------------------------------
    all_pre_mlp_hidden_states = []
    all_hidden_states = []

    for layer in text_model.layers[: text_model.config.num_hidden_layers]:
        # ---- Self‑attention block (Ministral3DecoderLayer.forward) ----
        residual = hidden_states
        normed = layer.input_layernorm(hidden_states)

        attn_output, _ = layer.self_attn(
            hidden_states=normed,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + attn_output

        # ---- MLP block ----
        residual = hidden_states
        mlp_input = layer.post_attention_layernorm(hidden_states)

        if return_hidden_states:
            # Save pre‑MLP activations as (B*N, D)
            all_pre_mlp_hidden_states.append(
                mlp_input.reshape(-1, mlp_input.shape[-1]).detach().cpu()
            )

        mlp_output = layer.mlp(mlp_input)
        hidden_states = residual + mlp_output

        if return_hidden_states:
            # Save post‑layer activations as (B*N, D)
            all_hidden_states.append(
                hidden_states.reshape(-1, hidden_states.shape[-1]).detach().cpu()
            )

    # Final RMSNorm (same as Ministral3Model.forward)
    hidden_states = text_model.norm(hidden_states)

    # ------------------------------------------------------------------
    # 4. LM head: logits over vocab
    # ------------------------------------------------------------------
    logits = model.lm_head(hidden_states)  # (B, N, V)

    return {
        "logits": logits,
        "all_topk_experts": [],          # dense model
        "all_topk_weights": [],          # dense model
        "all_pre_mlp_hidden_states": all_pre_mlp_hidden_states,
        "all_router_logits": [],         # dense model
        "all_hidden_states": all_hidden_states,
        "all_expert_outputs": [],        # dense model
    }
