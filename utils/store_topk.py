import pandas as pd
import torch

@torch.no_grad()
def convert_topk_to_df(input_ids: torch.Tensor, attention_mask: torch.Tensor, all_topk_experts: list[torch.Tensor], all_topk_weights: list[torch.Tensor] = None) -> pd.DataFrame:
    """
    Converts all_topk_experts into a pandas dataframe for later analysis.

    Params:
        @input_ids: A tensor of input IDs of size B x N
        @attention_mask: A tensor of 1 for real tokens, 0 for padding, of size B x N
        @all_topk_experts: A list or tuple of n_layers length, with each element a tensor size (BN, topk) containing the expert IDs selected
        @all_topk_weights: (Optional) A list or tuple of n_layers length, with each element a tensor size (BN, topk) containing the expert weights selected

    Returns:
        A dataframe at `sequence_ix` x `token_ix` x `layer_ix` x `topk_ix`, excluding masked tokens, with columns:
        - `sequence_ix`: The index of the batch sub-samples.
        - `token_ix`: The token index of a single sequence.
        - `layer_ix`: The layer index.
        - `token_id`: The token ID at that `sequence_ix` x `token_ix`.
        - `topk_ix`: The index of the expert within the top-k selection (1-based).
        - `expert`: The expert ID at that position.
        - `weight`: The weight of the expert at that position.

    Example:
        prompt = 'Hello'
        inputs = tokenizer(prompt, return_tensors = 'pt').to(main_device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            output = model(input_ids, attention_mask)

        convert_topk_to_df(input_ids, output['all_topk_experts'],)
    """
    # Shape validation for all_topk_weights
    if all_topk_weights is not None:
        # Check that the number of layers matches
        if len(all_topk_weights) != len(all_topk_experts):
            raise ValueError(f"all_topk_weights has {len(all_topk_weights)} layers but all_topk_experts has {len(all_topk_experts)} layers")
        # Check that the shape of each layer matches
        for layer_idx, (expert_tensor, weight_tensor) in enumerate(zip(all_topk_experts, all_topk_weights)):
            if expert_tensor.shape != weight_tensor.shape:
                raise ValueError(f"Shape mismatch at layer {layer_idx}: experts {expert_tensor.shape}, weights {weight_tensor.shape}")

    data = []
    
    B, N = input_ids.shape
    top_k = all_topk_experts[0].shape[1]
    
    # Flatten input_ids to match all_topk_experts shape B*N
    flat_input_ids = input_ids.reshape(-1).cpu().numpy()
    flat_attention = attention_mask.view(-1).cpu().numpy() 
    
    for layer_ix, layer_experts in enumerate(all_topk_experts):
        layer_experts_np = layer_experts.cpu().numpy()
        # If weights are provided, convert once for this layer
        if all_topk_weights is not None:
            layer_weights_np = all_topk_weights[layer_ix].cpu().numpy().round(2)  # shape [B*N, top_k]
        else:
            layer_weights_np = None
            
        # For each token position
        for token_pos in range(B * N):
            if flat_attention[token_pos] == 0:
                continue
            # Get batch and token indices
            sequence_ix = token_pos // N
            token_ix = token_pos % N
            token_id = flat_input_ids[token_pos]

            # Get experts for this token  - shape topk
            experts = layer_experts_np[token_pos]
            
            # Add each expert
            for k in range(top_k):
                row = {
                    "sequence_ix": sequence_ix,
                    "token_ix": token_ix,
                    "token_id": token_id,
                    "layer_ix": layer_ix,
                    "topk_ix": k + 1,  # 1-based indexing for topk_ix
                    "expert": experts[k]
                }  

                # Add weight if provided, otherwise use NaN
                if layer_weights_np is not None:
                    row["weight"] = layer_weights_np[token_pos, k]
                else:
                    row["weight"] = float("nan")
                    
                data.append(row)
    
    df = pd.DataFrame(data)
    return df