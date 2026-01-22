"""
Reversed engineered forward pass for GLM-4.7-Flash
- Supports GLM-4.7-Flash
- See: modeling_glm4moelite.py
"""
import torch
from transformers.masking_utils import create_causal_mask
from ._pretrained_helpers import _sort_gate_tensors

@torch.no_grad()
def run_glm4moelite_return_topk(model, input_ids, attention_mask, return_hidden_states = False):
    """
    Params:
        @model: A model of class `Glm4MoeLite`.
        @input_ids: A (B, N) tensor of input IDs on the same device as `model`.
        @attention_mask: A (B, N) tensor of mask indicators on the same device as `model`.
        @return_hidden_states: Boolean; whether to return hidden_states themselves.

    Returns:
        A dictionary with keys:
        - `logits`: (B, N, V) LM outputs
        - `all_topk_experts`: List (len = # MoE layers) of (BN, topk) expert IDs tensors
        - `all_topk_weights`: List (len = # MoE layers) of (BN, topk) expert weight tensors
        - `all_pre_mlp_hidden_states`: (optional) List (len = # MoE layers) of (BN, D) pre-MLP activations
        - `all_router_logits: (optional) List (len = # MoE layers) of (BN, n_experts) router *logits*
        - `all_hidden_states`: (optional) List (len = # MoE layers) of (BN, D) post-layer activations
        - `all_expert_outputs`: (optional) List (len = # MoE layers) of (BN, topk, D) pre-weighting expert outputs
    """
    # Embeddings
    input_embeds = model.model.embed_tokens(input_ids)

    # Positions (no caching / no past_kv in this replica)
    cache_position = torch.arange(0, input_embeds.shape[1], device=input_embeds.device)
    position_ids = cache_position.unsqueeze(0)

    # Causal mask (HF helper)
    causal_mask = create_causal_mask(
        config=model.model.config,
        input_embeds=input_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )

    hidden_state = input_embeds

    # Shared RoPE embeddings across layers
    position_embeddings = model.model.rotary_emb(hidden_state, position_ids=position_ids)

    # Outputs (MoE layers only)
    all_moe_layer_ixs = []
    all_topk_experts = []
    all_topk_weights = []
    all_pre_mlp_hidden_states = []
    all_router_logits = []
    all_hidden_states = []
    all_expert_outputs = []

    for layer_ix, layer in enumerate(model.model.layers):
        # Self-attention block (matches Glm4MoeLiteDecoderLayer.forward structure)
        residual = hidden_state
        hidden_state = layer.input_layernorm(hidden_state)

        hidden_state, _ = layer.self_attn(
            hidden_states=hidden_state,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,        # accepted via **kwargs; ignored by attention forward
            past_key_values=None,
            use_cache=False,                  # accepted via **kwargs; ignored by attention forward
            cache_position=cache_position,
        )
        hidden_state = residual + hidden_state

        # MLP / MoE block
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)

        is_moe_layer = (
            hasattr(layer.mlp, "gate")
            and hasattr(layer.mlp, "route_tokens_to_experts")
            and hasattr(layer.mlp, "experts")
            and hasattr(layer.mlp, "shared_experts")
        )

        if is_moe_layer:
            # Record which transformer layer this corresponds to
            all_moe_layer_ixs.append(layer_ix)

            # Router inputs (pre-MLP hidden state)
            if return_hidden_states:
                all_pre_mlp_hidden_states.append(hidden_state.view(-1, hidden_state.shape[2]).detach().cpu())

            # Compute routing exactly like Glm4MoeLiteMoE.forward does (but keep the intermediates)
            router_logits = layer.mlp.gate(hidden_state)  # (B*N, n_routed_experts), fp32 from HF implementation
            topk_indices, topk_weights = layer.mlp.route_tokens_to_experts(router_logits)  # (B*N, topk), (B*N, topk)

            topk_ids, topk_weight, layer_expert_outputs = _sort_gate_tensors(
                topk_indices.detach(),
                topk_weights.detach(),
                None
            )

            all_topk_experts.append(topk_ids.detach().cpu())
            all_topk_weights.append(topk_weight.detach().cpu().to(torch.float32))

            if return_hidden_states:
                all_router_logits.append(router_logits.detach().cpu())
                all_expert_outputs.append(None)

            # Run MoE forward without recomputing the routing decisions
            # (this is exactly Glm4MoeLiteMoE.forward but using our topk tensors)
            residuals = hidden_state
            orig_shape = hidden_state.shape

            flat_hidden = hidden_state.view(-1, hidden_state.shape[-1])
            routed_out = layer.mlp.experts(flat_hidden, topk_indices, topk_weights).view(*orig_shape)
            shared_out = layer.mlp.shared_experts(residuals)

            hidden_state = routed_out + shared_out

        else:
            # Dense MLP layer
            hidden_state = layer.mlp(hidden_state)

        hidden_state = residual + hidden_state

        if return_hidden_states and is_moe_layer:
            all_hidden_states.append(hidden_state.view(-1, hidden_state.shape[2]).detach().cpu())

    # Final norm + LM head
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