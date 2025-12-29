"""
Reversed engineered forward pass for Nemotron-H hybrid models
- Supports nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 and nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
- Steps through NemotronHModel blocks (mamba / attention / mlp / moe) without hooks
- Matches the provided `modeling_nemotron_h.py` logic (including its block-level argument passing)
"""
import contextlib
import torch
import torch.nn.functional as F

def _nemotronh_router_forward(router, hidden_states_2d: torch.Tensor):
    """
    Replicates NemotronHTopkRouter.forward, but also returns router_logits.
    hidden_states_2d: (T, D)
    """
    # router_logits: (T, E) in float32
    router_logits = F.linear(hidden_states_2d.to(torch.float32), router.weight.to(torch.float32))
    scores = router_logits.sigmoid()

    # Same as NemotronHTopkRouter.get_topk_indices(scores)
    scores_for_choice = scores.view(-1, router.n_routed_experts) + router.e_score_correction_bias.unsqueeze(0)

    group_scores = (
        scores_for_choice.view(-1, router.n_group, router.n_routed_experts // router.n_group)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=router.topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, router.n_group, router.n_routed_experts // router.n_group)
        .reshape(-1, router.n_routed_experts)
    )

    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
    topk_indices = torch.topk(scores_for_choice, k=router.top_k, dim=-1, sorted=False)[1]  # (T, K)

    topk_weights = scores.gather(1, topk_indices)  # (T, K)
    if router.norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator
    topk_weights = topk_weights * router.routed_scaling_factor

    return router_logits, topk_indices, topk_weights


def _nemotronh_moe_forward_with_topk(moe_module, hidden_states_3d: torch.Tensor):
    """
    Replicates NemotronHMOE.forward, but also returns (router_logits, topk_indices, topk_weights).

    hidden_states_3d: (B, N, D)
    Returns:
      - out: (B, N, D)
      - router_logits: (B, N, E)
      - topk_indices: (B, N, K)
      - topk_weights: (B, N, K)
    """
    residuals = hidden_states_3d
    B, N, D = hidden_states_3d.shape

    hs_2d = hidden_states_3d.view(-1, D)  # (T, D) where T = B*N
    router_logits_2d, topk_indices_2d, topk_weights_2d = _nemotronh_router_forward(moe_module.gate, hs_2d)

    # Same as NemotronHMOE.forward
    moe_out_2d = moe_module.moe(hs_2d, topk_indices_2d, topk_weights_2d)  # (T, D)
    moe_out_3d = moe_out_2d.view(B, N, D)
    out = moe_out_3d + moe_module.shared_experts(residuals)

    router_logits = router_logits_2d.view(B, N, -1)
    topk_indices = topk_indices_2d.view(B, N, -1)
    topk_weights = topk_weights_2d.view(B, N, -1)

    return out, router_logits, topk_indices, topk_weights


@torch.no_grad()
def run_nemotron3_return_topk(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor = None,
    cache_params=None,
    cache_position: torch.Tensor = None,
    position_ids: torch.Tensor = None,
    return_hidden_states: bool = False,
):
    """
    Manual forward pass for NemotronH hybrid models (mamba/attention/mlp/moe), matching the provided HF code.

    Params:
        @model: A model of class `NemotronHForCausalLM` (expects `.backbone` and `.lm_head`).
        @input_ids: (B, N) tensor of input IDs on the same device as `model`.
        @attention_mask: (B, N) tensor mask (1 = keep, 0 = pad). If None, assumes all-ones.
        @cache_params: Optional `HybridMambaAttentionDynamicCache` (used by Mamba layers as in your code).
        @cache_position: Optional (N,) tensor of cache positions. If None, uses arange(N).
        @position_ids: Optional (B, N) tensor. If None, uses cache_position.unsqueeze(0) (kept for parity; not used by blocks here).
        @return_hidden_states: Whether to return per-layer flattened activations on CPU.

    Returns:
        A dictionary with keys:
        - `logits`: (B, N, V) float32 LM outputs (matches `NemotronHForCausalLM.forward` behavior)
        - `all_topk_experts`: List (len = #layers) of (B, N, K) Long tensors for MoE layers, else None
        - `all_topk_weights`: List (len = #layers) of (B, N, K) float tensors for MoE layers, else None
        - `all_router_logits`: List (len = #layers) of (B, N, E) float32 tensors for MoE layers, else None
        - `all_pre_mlp_hidden_states`: (optional) List (len = #layers) of (BN, D) normed pre-mixer activations (CPU)
        - `all_hidden_states`: (optional) List (len = #layers) of (BN, D) post-block activations (CPU)
        - `all_expert_outputs`: [] (kept for API compatibility; not populated here)
    """
    if not (hasattr(model, "backbone") and hasattr(model, "lm_head")):
        raise ValueError("Expected a NemotronHForCausalLM-like model with `.backbone` and `.lm_head`.")

    backbone = model.backbone
    lm_head = model.lm_head

    device = getattr(model, "device", next(model.parameters()).device)
    input_ids = input_ids.to(device)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = attention_mask.to(device)

    # Embeddings (matches NemotronHModel.forward)
    inputs_embeds = backbone.get_input_embeddings()(input_ids)
    hidden_states = inputs_embeds
    B, N, D = hidden_states.shape

    if cache_position is None:
        cache_position = torch.arange(N, device=hidden_states.device)
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # Masks (computed for parity with NemotronHModel.forward, even though your NemotronHBlock.forward
    # currently doesn't pass them into the mixers)
    _ = backbone._update_causal_mask(attention_mask, inputs_embeds, cache_position)
    _ = backbone._update_mamba_mask(attention_mask, cache_position)

    all_topk_experts = []
    all_topk_weights = []
    all_router_logits = []
    all_expert_outputs = []

    all_pre_mlp_hidden_states = []
    all_hidden_states = []

    for layer_idx, block in enumerate(backbone.layers):
        # Match NemotronHBlock.forward's CUDA stream guard
        if hidden_states.is_cuda:
            stream_ctx = torch.cuda.stream(torch.cuda.default_stream(hidden_states.device))
        else:
            stream_ctx = contextlib.nullcontext()

        with stream_ctx:
            residual = hidden_states

            # Norm (match: hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype)))
            hs = block.norm(hidden_states.to(dtype=block.norm.weight.dtype))

            if return_hidden_states:
                all_pre_mlp_hidden_states.append(hs.reshape(-1, hs.shape[-1]).detach().cpu())

            if block.residual_in_fp32:
                residual = residual.to(torch.float32)

            if block.block_type == "mamba":
                # Match your NemotronHBlock.forward: does NOT pass attention_mask into mamba mixer
                out = block.mixer(hs, cache_params=cache_params, cache_position=cache_position)

                all_topk_experts.append(None)
                all_topk_weights.append(None)
                all_router_logits.append(None)

            elif block.block_type == "attention":
                # Match your NemotronHBlock.forward: only passes cache_position
                attn_out = block.mixer(hs, cache_position=cache_position)
                out = attn_out[0]

                all_topk_experts.append(None)
                all_topk_weights.append(None)
                all_router_logits.append(None)

            elif block.block_type == "mlp":
                out = block.mixer(hs)

                all_topk_experts.append(None)
                all_topk_weights.append(None)
                all_router_logits.append(None)

            elif block.block_type == "moe":
                out, router_logits, topk_indices, topk_weights = _nemotronh_moe_forward_with_topk(block.mixer, hs)

                # Keep on-device by default (caller can move to CPU); mirrors typical “logits stay on device” style
                all_topk_experts.append(topk_indices)
                all_topk_weights.append(topk_weights)
                all_router_logits.append(router_logits)

            else:
                raise ValueError(f"Invalid block_type: {block.block_type}")

            hidden_states = residual + out

        if return_hidden_states:
            all_hidden_states.append(hidden_states.reshape(-1, hidden_states.shape[-1]).detach().cpu())

    # Final norm (matches NemotronHModel.forward)
    hidden_states = backbone.norm_f(hidden_states)

    # LM head (matches NemotronHForCausalLM.forward)
    logits = lm_head(hidden_states.to(lm_head.weight.dtype)).float()

    return {
        "logits": logits,
        "all_topk_experts": all_topk_experts,
        "all_topk_weights": all_topk_weights,
        "all_pre_mlp_hidden_states": all_pre_mlp_hidden_states,
        "all_router_logits": all_router_logits,
        "all_hidden_states": all_hidden_states,
        "all_expert_outputs": all_expert_outputs,
    }
