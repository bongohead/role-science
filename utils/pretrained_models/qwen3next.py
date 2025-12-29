import torch
from transformers.masking_utils import create_causal_mask

@torch.no_grad()
def run_qwen3next_return_topk(model, input_ids, attention_mask, return_hidden_states=False):
    """
    Reversed engineered forward pass for Qwen3-Next (HF v4.57.3)
    - Matches the structure/IO of run_qwen3moe_return_topk
    - Handles both full_attention and linear_attention token mixers
    - Re-implements Qwen3NextSparseMoeBlock inline to expose top-k routing + weights
      (and includes the shared expert contribution for correctness)

    Params:
        @model: A model of class `Qwen3NextForCausalLM`.
        @input_ids: A B x N tensor of inputs IDs on the same device as `model`.
        @attention_mask: A B x N tensor of mask indicators on the same device as `model`.
        @return_hidden_states: Boolean; whether to return hidden_states themselves.

    Returns:
        A dictionary with keys (same as run_qwen3moe_return_topk):
        - `logits`: The standard B x N x V LM output
        - `all_topk_experts`: list over MoE layers, each (BN) x topk expert IDs
        - `all_topk_weights`: list over MoE layers, each (BN) x topk expert weights
        - `all_pre_mlp_hidden_states`: (if return_hidden_states) list over MoE layers, each (BN) x D
        - `all_router_logits`: (if return_hidden_states) list over MoE layers, each (BN) x n_experts
        - `all_hidden_states`: (if return_hidden_states) list over MoE layers, each (BN) x D (post-layer)
        - `all_expert_outputs`: (if return_hidden_states) list over MoE layers, each (BN) x topk x D (pre-weighting)
    """
    # Embeddings
    input_embeds = model.model.embed_tokens(input_ids)  # (B, N, D)

    # Positions (no caching)
    cache_position = torch.arange(0, input_embeds.shape[1], device=input_embeds.device)  # (N,)
    position_ids = cache_position.unsqueeze(0)  # (1, N)

    # Full-attention causal mask (matches Qwen3NextModel.forward in v4.57.3)
    causal_mask = create_causal_mask(
        config=model.config,
        input_embeds=input_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )

    # Linear-attention mask (matches Qwen3NextModel._update_linear_attn_mask)
    linear_attn_mask = attention_mask
    if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
        linear_attn_mask = None

    hidden_state = input_embeds

    # RoPE embeddings shared across layers (only used by full_attention layers)
    position_embeddings = model.model.rotary_emb(hidden_state, position_ids)

    # Outputs
    all_topk_experts = []
    all_topk_weights = []
    all_pre_mlp_hidden_states = []
    all_router_logits = []
    all_hidden_states = []
    all_expert_outputs = []

    for layer_ix, layer in enumerate(model.model.layers):
        # --------------------
        # Token mixer block
        # --------------------
        residual = hidden_state
        hidden_state = layer.input_layernorm(hidden_state)

        if layer.layer_type == "linear_attention":
            # linear attn takes a (B,N) mask or None (NOT the 4D causal mask)
            hidden_state = layer.linear_attn(
                hidden_states=hidden_state,
                cache_params=None,
                cache_position=cache_position,
                attention_mask=linear_attn_mask,
            )
        elif layer.layer_type == "full_attention":
            # full attention takes the 4D causal mask
            hidden_state, _ = layer.self_attn(
                hidden_states=hidden_state,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        else:
            raise ValueError(f"Unknown Qwen3Next layer_type={layer.layer_type!r} at layer {layer_ix}")

        hidden_state = residual + hidden_state

        # --------------------
        # MLP / MoE block (we inline MoE to expose routing)
        # --------------------
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)

        # Detect whether this layer is MoE. (Dense MLP layers do not have `.gate` / `.experts`.)
        is_moe = hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "experts") and hasattr(layer.mlp, "top_k")

        if not is_moe:
            # Dense MLP path (no routing info to record)
            hidden_state = layer.mlp(hidden_state)
            hidden_state = residual + hidden_state
            continue

        # Record pre-MLP hidden states (post-attn norm) for MoE layers only
        if return_hidden_states:
            all_pre_mlp_hidden_states.append(hidden_state.view(-1, hidden_state.shape[-1]).detach().cpu())

        batch_size, sequence_length, hidden_dim = hidden_state.shape
        moe_hidden_state = hidden_state.view(-1, hidden_dim)  # (BN, D)

        # Router logits + top-k
        router_logits = layer.mlp.gate(moe_hidden_state)  # (BN, E)

        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)  # fp32 softmax
        routing_weights, selected_experts = torch.topk(
            routing_weights, layer.mlp.top_k, dim=-1, sorted=True
        )  # (BN, K), (BN, K)

        if layer.mlp.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        routing_weights = routing_weights.to(moe_hidden_state.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=moe_hidden_state.dtype,
            device=moe_hidden_state.device,
        )  # (BN, D)

        # expert_mask: (E, K, BN)
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=layer.mlp.num_experts
        ).permute(2, 1, 0)

        if return_hidden_states:
            # (BN, K, D) pre-weighting expert outputs
            layer_expert_outputs = torch.zeros(
                (batch_size * sequence_length, layer.mlp.top_k, hidden_dim),
                dtype=moe_hidden_state.dtype,
                device=moe_hidden_state.device,
            )

        # Per-expert dispatch (same style as your Qwen3MoE function)
        for expert_idx in range(layer.mlp.num_experts):
            expert_layer = layer.mlp.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])  # idx in [0..K), top_x in [0..BN)

            current_state = moe_hidden_state[None, top_x].reshape(-1, hidden_dim)
            current_expert_output = expert_layer(current_state)  # (num_tokens_for_expert, D)

            current_hidden_states = current_expert_output * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(moe_hidden_state.dtype))

            if return_hidden_states:
                layer_expert_outputs[top_x, idx] = current_expert_output.to(layer_expert_outputs.dtype)

        # Shared expert branch (Qwen3Next-specific; required for correct forward output)
        shared_expert_output = layer.mlp.shared_expert(moe_hidden_state)  # (BN, D)
        shared_gate = torch.sigmoid(layer.mlp.shared_expert_gate(moe_hidden_state))  # (BN, 1)
        shared_expert_output = shared_gate * shared_expert_output  # broadcast to (BN, D)

        final_hidden_states = final_hidden_states + shared_expert_output.to(final_hidden_states.dtype)

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)  # (B, N, D)

        hidden_state = final_hidden_states
        hidden_state = residual + hidden_state

        # Record top-k routing info (MoE layers only)
        all_topk_experts.append(selected_experts.detach().cpu())
        all_topk_weights.append(routing_weights.detach().cpu().to(torch.float32))

        if return_hidden_states:
            all_router_logits.append(router_logits.detach().cpu())
            all_hidden_states.append(hidden_state.view(-1, hidden_dim).detach().cpu())
            all_expert_outputs.append(layer_expert_outputs.detach().cpu())

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
