import torch

def _sort_gate_tensors(ids: torch.Tensor, weights: torch.Tensor, expert_outputs: torch.Tensor | None = None):
    """
    Sort the `topk` axis descending by `weights`, and apply the same permutation to expert IDs (and optional raw expert outputs). 
    - For Deepseek-based architectures, all functions that return top-k or activations must sort these, since 
      the MoE MUST be run with sorted = False.
    - GLM4.5 also runs with sorted = False by default.
    - For most other  models, MoEs can be run with sorted = True. However, it may be useful to call this on code that involves 
      ablation to re-sort the IDs, weights, and expert outputs according to the new topk order.
 
    Params:
        @ids: BN x k tensor of expert IDs
        @weights: BN x k tensor of gate probs
        @outputs BN x k x D tensor of raw expert outputs (optional)

    Returns:
        ids_sorted, weights_sorted, outputs_sorted (None if outputs is None)
    """
    weights_sorted, order = torch.sort(weights, dim = 1, descending = True)
    ids_sorted = torch.take_along_dim(ids, order, dim = 1)
    if expert_outputs is None:
        return ids_sorted, weights_sorted, None
    expert_outputs_sorted = torch.take_along_dim(
        expert_outputs, order.unsqueeze(-1), dim = 1
    )
    return ids_sorted, weights_sorted, expert_outputs_sorted

def _move_device(t, target_device):
    """
    Move tensor to dev only if necessary
    """
    return t if t is None or t.device == target_device else t.to(target_device, non_blocking = True)