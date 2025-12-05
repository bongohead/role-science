"""
Reverse-engineered forward pass for Qwen3VLMoE
- Supports text-only, plus optional vision/video inputs.
- See https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py
"""

import torch
from transformers.masking_utils import create_causal_mask
from ._pretrained_helpers import _sort_gate_tensors 

@torch.no_grad()
def run_qwen3vlmoe_return_topk(model, input_ids, attention_mask, pixel_values = None, pixel_values_videos = None, image_grid_thw = None, video_grid_thw = None, return_hidden_states: bool = False):
    """
    Params:
        @model: A model of class `Qwen3VLMoeForConditionalGeneration`.
        @input_ids: A (B, N) tensor of input IDs on the same device as `model`.
        @attention_mask: A (B, N) tensor of mask indicators on the same device as `model`.
        @pixel_values: (optional) image tensor as expected by Qwen3VLMoE; if None, text-only.
        @pixel_values_videos: (optional) video tensor as expected by Qwen3VLMoE; if None, no video.
        @image_grid_thw: (optional) (L_img, 3) tensor of T-H-W grids for each encoded image.
        @video_grid_thw: (optional) (L_vid, 3) tensor of T-H-W grids for each encoded video.
        @return_hidden_states: Boolean; whether to return optional outputs below.

    Returns:
        A dictionary with keys:
        - `logits`: (B, N, V) LM outputs
        - `all_topk_experts`: List (len = # MoE layers) of (BN, topk) expert ID tensors
        - `all_topk_weights`: List (len = # MoE layers) of (BN, topk) expert weight tensors
        - `all_pre_mlp_hidden_states`: (optional) List (len = # MoE layers) of (BN, D) pre-MLP activations
        - `all_router_logits`: (optional) List (len = # MoE layers) of (BN, n_experts) router *logits* (pre-softmax)
        - `all_hidden_states`: (optional) List (len = # MoE layers) of (BN, D) post-layer activations
        - `all_expert_outputs`: (optional) List (len = # MoE layers) of (BN, topk, D) pre-weighting expert outputs
    """
    # Unpack underlying components
    vl_model = model.model                    # Qwen3VLMoeModel
    text_model = vl_model.language_model      # Qwen3VLMoeTextModel

    # ---- 1. Text embeddings -------------------------------------------------
    inputs_embeds = text_model.embed_tokens(input_ids)  # (B, N, D)
    B, N, D = inputs_embeds.shape

    # ---- 2. Optional vision / video features + DeepStack setup --------------
    image_mask = None
    video_mask = None
    visual_pos_masks = None
    deepstack_visual_embeds = None

    # Images
    if pixel_values is not None:
        # (list_of_chunks, deepstack_image_embeds)
        image_embeds_list, deepstack_image_embeds = vl_model.get_image_features(pixel_values, image_grid_thw)
        # Flatten list of image embeddings into a single (total_image_tokens, D) tensor
        image_embeds = torch.cat(image_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

        # Mask of where image placeholder tokens are
        image_mask, _ = vl_model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        # Replace placeholder token embeddings with image embeddings
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    # Videos
    if pixel_values_videos is not None:
        video_embeds_list, deepstack_video_embeds = vl_model.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = vl_model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    # DeepStack visual masks/embeds (matches Qwen3VLMoeModel.forward)
    if image_mask is not None and video_mask is not None:
        image_mask_b = image_mask[..., 0]
        video_mask_b = video_mask[..., 0]
        visual_pos_masks = image_mask_b | video_mask_b
        deepstack_visual_embeds = []

        image_mask_joint = image_mask_b[visual_pos_masks]
        video_mask_joint = video_mask_b[visual_pos_masks]

        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            # (num_visual_positions, D)
            embed_joint = img_embed.new_zeros(
                visual_pos_masks.sum(), img_embed.shape[-1]
            ).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)

    elif image_mask is not None:
        visual_pos_masks = image_mask[..., 0]
        deepstack_visual_embeds = deepstack_image_embeds

    elif video_mask is not None:
        visual_pos_masks = video_mask[..., 0]
        deepstack_visual_embeds = deepstack_video_embeds

    # ---- 3. RoPE indices (multi-axis) ---------------------------------------
    # We only handle the "prefill" case (no cache), so we always call get_rope_index.
    position_ids, _ = vl_model.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
    )  # (3, B, N)

    cache_position = torch.arange(0, N, device=inputs_embeds.device)
    text_position_ids = position_ids[0]  # (B, N) text axis

    # ---- 4. Causal mask + shared rotary position embeddings -----------------
    causal_mask = create_causal_mask(
        config=text_model.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=text_position_ids,
    )

    hidden_state = inputs_embeds  # (B, N, D)
    position_embeddings = text_model.rotary_emb(hidden_state, position_ids=position_ids)

    # ---- 5. Per-layer decoding + MoE introspection --------------------------
    all_topk_experts = []
    all_topk_weights = []
    all_pre_mlp_hidden_states = []
    all_router_logits = []
    all_hidden_states = []
    all_expert_outputs = []

    for layer_idx, layer in enumerate(text_model.layers):
        # Self-attention (as in Qwen3VLMoeTextDecoderLayer.forward)
        residual = hidden_state
        hidden_state = layer.input_layernorm(hidden_state)
        hidden_state, _ = layer.self_attn(
            hidden_states=hidden_state,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_state = residual + hidden_state

        # MLP / MoE
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)

        # Check if this is a MoE layer
        is_moe_layer = hasattr(layer.mlp, "experts") and hasattr(layer.mlp, "gate")

        if is_moe_layer and return_hidden_states:
            # Pre-MLP activations (BN, D)
            all_pre_mlp_hidden_states.append(
                hidden_state.view(-1, hidden_state.shape[2]).detach().cpu()
            )

        if is_moe_layer:
            # ---- Qwen3VLMoeTextSparseMoeBlock manual re-implementation ----
            batch_size, seq_length, hidden_dim = hidden_state.shape
            hs_flat = hidden_state.view(-1, hidden_dim)  # (BN, D)

            # Gate: linear + softmax + top-k
            router_logits = torch.nn.functional.linear(
                hs_flat, layer.mlp.gate.weight
            )  # (BN, num_experts)
            router_probs = torch.nn.functional.softmax(
                router_logits, dim=-1, dtype=torch.float
            )
            routing_weights_top, selected_experts = torch.topk(
                router_probs,
                layer.mlp.top_k,
                dim=-1,
                sorted=True,
            )  # (BN, top_k) each

            # Renormalize over top-k
            routing_weights_top /= routing_weights_top.sum(dim=-1, keepdim=True)
            routing_weights_top = routing_weights_top.to(router_logits.dtype)  # keep f32
            # For combining experts we only need the top-k weights; no dense scatter needed.

            # Experts: apply per-expert MLP with top-k routing
            final_flat = torch.zeros_like(hs_flat)  # (BN, D)

            # One-hot over experts: (num_experts, top_k, BN)
            expert_mask = torch.nn.functional.one_hot(
                selected_experts,
                num_classes=layer.mlp.experts.num_experts,
            ).permute(2, 1, 0)
            expert_hits = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()  # (num_hits, 1)

            if return_hidden_states:
                layer_expert_outputs = torch.zeros(
                    hs_flat.size(0),
                    layer.mlp.top_k,
                    hidden_dim,
                    dtype=hs_flat.dtype,
                    device=hs_flat.device,
                )

            for expert_idx_tensor in expert_hits:
                expert_idx = expert_idx_tensor[0]

                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                # token_idx: which tokens chose this expert
                # top_k_pos: which top-k slot (0 = highest prob, etc)

                current_state = hs_flat[token_idx]  # (n_tokens_for_expert, D)

                # gate_up_proj: (num_experts, hidden_size, 2 * expert_dim)
                gate_up = current_state @ layer.mlp.experts.gate_up_proj[expert_idx]
                gate, up = gate_up.chunk(2, dim=-1)
                current_expert_output = layer.mlp.experts.act_fn(gate) * up
                # down_proj: (num_experts, expert_dim, hidden_size)
                current_expert_output = current_expert_output @ layer.mlp.experts.down_proj[expert_idx]

                # Weight by routing probability
                weighted = current_expert_output * routing_weights_top[token_idx, top_k_pos, None]
                final_flat.index_add_(0, token_idx, weighted.to(final_flat.dtype))

                if return_hidden_states:
                    layer_expert_outputs[token_idx, top_k_pos] = current_expert_output.to(
                        layer_expert_outputs.dtype
                    )

            final_hidden_states = final_flat.view(batch_size, seq_length, hidden_dim)
            hidden_state = residual + final_hidden_states

            # Collect MoE stats for this layer
            all_topk_experts.append(selected_experts.detach().cpu())  # (BN, top_k)
            all_topk_weights.append(
                routing_weights_top.detach().cpu().to(torch.float32)
            )

            if return_hidden_states:
                all_router_logits.append(router_logits.detach().cpu())
                all_expert_outputs.append(layer_expert_outputs.detach().cpu())

        else:
            # Dense MLP
            mlp_out = layer.mlp(hidden_state)
            hidden_state = residual + mlp_out

        # DeepStack visual integration (matches Qwen3VLMoeTextModel.forward)
        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            hidden_state = text_model._deepstack_process(
                hidden_state,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )

        # Post-layer activations (only if we’re recording them)
        if is_moe_layer and return_hidden_states:
            all_hidden_states.append(
                hidden_state.view(-1, hidden_state.shape[2]).detach().cpu()
            )

    # ---- 6. Final norm + LM head --------------------------------------------
    hidden_state = text_model.norm(hidden_state)
    logits = model.lm_head(hidden_state)  # (B, N, V)

    return {
        "logits": logits,
        "all_topk_experts": all_topk_experts,
        "all_topk_weights": all_topk_weights,
        "all_pre_mlp_hidden_states": all_pre_mlp_hidden_states,
        "all_router_logits": all_router_logits,
        "all_hidden_states": all_hidden_states,
        "all_expert_outputs": all_expert_outputs,
    }