"""
Reversed engineered forward pass for Qwen
- Supports all Qwen3-30B-A3B variants and Qwen3-235B-A22B
- See https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
"""
import torch
from transformers.masking_utils import create_causal_mask
from ._pretrained_helpers import _sort_gate_tensors

@torch.no_grad()
def run_qwen3moe_return_topk(model, input_ids, attention_mask, return_hidden_states = False):
    """
    Params:
        @model: A model of class `Qwen3MoeForCausalLM`.
        @input_ids: A (B, N) tensor of inputs IDs on the same device as `model`.
        @attention_mask: A (B, N) tensor of mask indicators on the same device as `model`.
        @return_hidden_states: Boolean; whether to return optional outputs below.

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
    input_embeds = model.model.embed_tokens(input_ids)
    B, N, D = input_embeds.shape

    cache_position = torch.arange(0, N, device = input_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = create_causal_mask(model.model.config, input_embeds, attention_mask, cache_position, None, position_ids) # Assm no sliding window.
    position_embeddings = model.model.rotary_emb(input_embeds, position_ids)

    hidden_state = input_embeds

    all_topk_experts = []
    all_topk_weights = []
    all_pre_mlp_hidden_states = []
    all_router_logits = []
    all_hidden_states = []
    all_expert_outputs = []

    for layer_ix, layer in enumerate(model.model.layers):
        # SA
        residual = hidden_state
        hidden_state = layer.input_layernorm(hidden_state)
        hidden_state, _ = layer.self_attn(hidden_states = hidden_state, attention_mask = causal_mask, position_ids = position_ids, position_embeddings = position_embeddings)
        hidden_state = residual + hidden_state
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)
        
        if return_hidden_states:
            all_pre_mlp_hidden_states.append(hidden_state.view(-1, hidden_state.shape[2]).detach().cpu())

        ####### Qwen3MoeSparseMoeBlock - below code replaces hidden_state = layer.mlp(hidden_state)
        batch_size, sequence_length, hidden_dim = hidden_state.shape
        hidden_state_reshaped = hidden_state.view(-1, hidden_dim) # (BN, D)

        ## Qwen3MoeTopKRouter
        router_logits = torch.nn.functional.linear(hidden_state_reshaped, layer.mlp.gate.weight) # <=> hidden_state_reshaped @ weight.T; out (BN, n_experts)
        router_probs = torch.nn.functional.softmax(router_logits, dtype = torch.float, dim = -1) # Softmax dim = which dim "computes to sum to 1"
        routing_weights, selected_experts = torch.topk(router_probs, layer.mlp.gate.top_k, dim = -1, sorted = True) # Both (BN, topk); weights of sel experts + sel expert ndices

        if layer.mlp.gate.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim = -1, keepdim = True)
        routing_weights = routing_weights.to(router_probs.dtype) # Keep f32

        ## Qwen3MoeExperts
        final_hidden_states = torch.zeros_like(hidden_state_reshaped)
        # One hot encode the selected experts to create an expert mask 
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes = layer.mlp.experts.num_experts).permute(2, 1, 0) # (n_experts, top_k, BN)
        expert_hits = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero() # (num_hits, 1)

        if return_hidden_states:
            layer_expert_outputs = torch.zeros((batch_size * sequence_length, layer.mlp.gate.top_k, hidden_dim), dtype = hidden_state_reshaped.dtype, device = hidden_state_reshaped.device) # BN x topk x D

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in expert_hits:
            expert_idx = expert_idx[0]

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx]) # token_idx = which toks selected expert; top_k_pos = which topk slot for those toks; both (ntoks, )
            current_state = hidden_state_reshaped[token_idx] # (ntoks, D)
            
            gate, up = torch.nn.functional.linear(current_state, layer.mlp.experts.gate_up_proj[expert_idx]).chunk(2, dim = -1)
            current_expert_output = layer.mlp.experts.act_fn(gate) * up
            current_expert_output = torch.nn.functional.linear(current_expert_output, layer.mlp.experts.down_proj[expert_idx])
            current_hidden_states = current_expert_output * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

            if return_hidden_states:
                layer_expert_outputs[token_idx, top_k_pos] = current_expert_output.to(layer_expert_outputs.dtype)

        final_hidden_states = (final_hidden_states).reshape(batch_size, sequence_length, hidden_dim)
        #######
        hidden_state = final_hidden_states
        hidden_state = residual + hidden_state

        all_topk_experts.append(selected_experts.detach().cpu())
        all_topk_weights.append(routing_weights.detach().cpu().to(torch.float32))

        if return_hidden_states:
            all_router_logits.append(router_logits.detach().cpu())
            all_hidden_states.append(hidden_state.view(-1, hidden_state.shape[2]).detach().cpu())
            all_expert_outputs.append(layer_expert_outputs.detach().cpu())

    hidden_state = model.model.norm(hidden_state)
    logits = model.lm_head(hidden_state)

    return {
        'logits': logits,
        'all_topk_experts': all_topk_experts,
        'all_topk_weights': all_topk_weights,
        'all_pre_mlp_hidden_states': all_pre_mlp_hidden_states,
        'all_router_logits': all_router_logits,
        'all_hidden_states': all_hidden_states,
        'all_expert_outputs': all_expert_outputs
    }
