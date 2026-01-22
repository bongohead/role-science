"""
Helpers for probe training and usage
"""
import torch
from termcolor import colored
from transformers.loss.loss_utils import ForCausalLMLoss
import pandas as pd
from utils.store_outputs import convert_outputs_to_df_fast
from utils.dataset import ReconstructableTextDataset
from tqdm import tqdm
import cupy

@torch.no_grad()
def run_and_export_states(model, tokenizer, *, run_model_return_states, dl: ReconstructableTextDataset, layers_to_keep_acts: list[int]):
    """
    Run forward passes on given model and store the decomposed sample_df plus hidden states

    Params:
        @model: The model to run forward passes on via `run_model_return_states`. Should return a dict with keys `logits` and `all_pre_mlp_hidden_states`.
        @tokenizer: The tokenizer object corresponding to the model.
        @run_model_return_states: A function that runs the model and returns a dict with keys `logits` and `all_pre_mlp_hidden_states`.
        @dl: A ReconstructableTextDataset of which returns `input_ids`, `attention_mask`, `original_tokens`, and `prompt_ix`.
        @layers_to_keep_acts: A list of layer indices (0-indexed) for which to filter `all_pre_mlp_hidden_states` (see returned object description).

    Returns:
        A dict with keys:
        - `sample_df`: A sample (token)-level dataframe with corresponding input token ID, output token ID, and input token text (removes masked tokens)
        - `all_hs`: A tensor of size n_samples x layers_to_keep_acts x D return the hidden state for each retained layers. Each 
            n_sample corresponds to a row of sample_df.

    Example:
        test_outputs = run_and_export_states(model, tokenizer, train_dl, layers_to_keep_acts = list(range(model_n_layers)))
    """
    all_hidden_states = []
    sample_dfs = []

    for batch_ix, batch in tqdm(enumerate(dl), total = len(dl)):

        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        original_tokens = batch['original_tokens']
        prompt_indices = batch['prompt_ix']

        output = run_model_return_states(model, input_ids, attention_mask, return_hidden_states = True)

        # Check no bugs by validating output/perplexity
        if batch_ix == 0:
            loss = ForCausalLMLoss(
                output['logits'],
                torch.where(input_ids == tokenizer.pad_token_id, torch.tensor(-100), input_ids),
                output['logits'].size(-1)
            ).detach().cpu().item()
            for i in range(min(20, input_ids.size(0))):
                decoded_input = tokenizer.decode(input_ids[i, :], skip_special_tokens = False)
                next_token_id = torch.argmax(output['logits'][i, -1, :]).item()
                print('---------\n' + decoded_input + colored(tokenizer.decode([next_token_id], skip_special_tokens = False).replace('\n', '<lb>'), 'green'))
            print(f"PPL:", torch.exp(torch.tensor(loss)).item())
                
        original_tokens_df = pd.DataFrame(
            [(seq_i, tok_i, tok) for seq_i, tokens in enumerate(original_tokens) for tok_i, tok in enumerate(tokens)], 
            columns = ['sequence_ix', 'token_ix', 'token']
        )
                
        prompt_indices_df = pd.DataFrame(
            [(seq_i, seq_source) for seq_i, seq_source in enumerate(prompt_indices)], 
            columns = ['sequence_ix', 'prompt_ix']
        )
        
        # Create sample (token) level dataframe
        sample_df =\
            convert_outputs_to_df_fast(input_ids, attention_mask, output['logits'])\
            .merge(original_tokens_df, how = 'left', on = ['token_ix', 'sequence_ix'])\
            .merge(prompt_indices_df, how = 'left', on = ['sequence_ix'])\
            .assign(batch_ix = batch_ix)
        
        sample_dfs.append(sample_df)

        # Store pre-MLP hidden states - the fwd pass as n_layers list as BN x D, collapse to BN x n_layers x D, with BN filtering out masked items
        valid_pos = torch.where(attention_mask.cpu().view(-1) == 1) # Valid (BN, ) positions
        all_hidden_states.append(torch.stack(output['all_pre_mlp_hidden_states'], dim = 1)[valid_pos][:, layers_to_keep_acts, :])

    sample_df = pd.concat(sample_dfs, ignore_index = True).drop(columns = ['batch_ix', 'sequence_ix']) # Drop batch/seq_ix, since prompt_ix identifies
    all_hidden_states = torch.cat(all_hidden_states, dim = 0)

    return {
        'sample_df': sample_df,
        'all_hs': all_hidden_states
    }

def run_projections(valid_sample_df: pd.DataFrame, layer_hs: torch.Tensor, probe: dict) -> pd:
    """
    Run probe-level projections
    
    Params:
        @valid_sample_df: A sample-level df with columns `sample_ix` (1... T - 1), `sample_ix`.
            Can be shorter than full T - 1 due to pre-filters, as long as sample_ix is *indexed* corresponding to the full length of T.
        @layer_hs: A tensor of size T x D for the layer to project.
        @probe: The probe dict with keys `probe` (the trained model) and `role_space` (the roles list)
    
    Returns:
        A df at (sample_ix, target_role) level with cols `sample_ix`, `target_role`, `prob`
    """
    x_cp = cupy.asarray(layer_hs[valid_sample_df['sample_ix'].tolist(), :])
    y_cp = probe['probe'].predict_proba(x_cp).round(12)

    proj_results = pd.DataFrame(cupy.asnumpy(y_cp), columns = probe['role_space'])
    if len(proj_results) != len(valid_sample_df):
        raise Exception("Error!")

    role_df =\
        pd.concat([
            proj_results.reset_index(drop = True),
            valid_sample_df[['sample_ix']].reset_index(drop = True)
        ], axis = 1)\
        .melt(id_vars = ['sample_ix'], var_name = 'target_role', value_name = 'prob')\
        .reset_index(drop = True)\
        .assign(prob = lambda df: df['prob'].round(4))

    return role_df


def check_max_seq_len(tokenizer, strings: list[str]) -> int:
    """
    Check max token length across a list of strings
    """
    max_seqlen = int(tokenizer(strings, padding = True, truncation = False, return_tensors = 'pt')['attention_mask'].sum(dim = 1).max().item())
    return max_seqlen