import torch
from torch.utils.data import Dataset
from transformers.loss.loss_utils import ForCausalLMLoss
import pandas as pd 
import numpy as np
from tqdm import tqdm
import re
import gc

################### Dataset and dataloader helpers ###################
class ReconstructableTextDataset(Dataset):

    def __init__(self, raw_texts: list[str], tokenizer, max_length, **identifiers):
        """
        Creates a dataset object that contains the usual input_ids and attention_mask, but also returns a B-length list of the original tokens 
         in the same position as the input ids, as well as any optional identifiers. Returning the original tokens is important for BPE 
         tokenizers as otherwise it's difficult to reconstruct the correct string later! Supports fast tokenizers (full HFv5 compat).

        Params:
            @raw_texts: A list of samples of text dataset.
            @tokenizer: A HF tokenizer object.
            @ident_lists: Named lists such as q_indices = [...], sources = [...], each the same length as raw_texts. Will be identifiers. 
             These should contain useful identifiers that will be returned in the dataloader.

        Example:
            dl = DataLoader(
                ReconstructableTextDataset(['a', 'hello'], tokenizer, max_length = 768, q_indices = [0, 1]),
                batch_size = 2,
                shuffle = False,
                collate_fn = collate_fn
            )
        """
        if getattr(tokenizer, 'backend', None) == 'tokenizers': # For transfroerms v5, all tokenizer = backend supports fast toks
            supports_offset = True
        elif getattr(tokenizer, 'is_fast', False): # Backwards compat for transformers v4
            supports_offset = True
        else:
            supports_offset = False

        tokenized = tokenizer(
            raw_texts,
            add_special_tokens = False,
            max_length = max_length,
            padding = 'max_length',
            truncation = True,
            return_offsets_mapping = supports_offset,
            return_tensors = 'pt'
        )

        self.input_ids = tokenized['input_ids']
        self.attention_mask = tokenized['attention_mask']
        self.offset_mapping = tokenized.get('offset_mapping', None) # Set for fast tokenizers

        # Add additional mappings
        n = len(raw_texts)
        for k, v in identifiers.items():
            if len(v) != n:
                raise ValueError(f"Length mismatch for '{k}': {len(v)} ≠ {n}")
            setattr(self, k, v) # Sets identifiers as keys.
        self._ident_lists = identifiers  # Keep as dict for iteration
        
        self.tokenizer = tokenizer  
        self.original_tokens = self._get_original_tokens(raw_texts)

    def _get_original_tokens(self, texts):
        """
        Return the original tokens associated with each B x N position. This is important for reconstructing the original text when BPE tokenizers are used. They 
         are returned in form [[seq1tok1, seq1tok2, ...], [seq2tok1, seq2tok2, ...], ...].
        
        Params:
            @input_ids: A B x N tensor of input ids.
            @offset_mapping: A B x N x 2 tensor of offset mappings. Get from `tokenizer(..., return_offsets_mapping = True)`.

        Returns:
            A list of length B, each with length N, containing the corresponding original tokens corresponding to the token ID at the same position of input_ids.
        """
        all_token_substrings = []
        # General path for fast tokenizers
        if self.offset_mapping is not None:
            # Row count is unchanged—one row per token ID—only the `token` string changes.
            for i in range(self.input_ids.shape[0]):
                text = texts[i]
                offsets = [tuple(map(int, pair.tolist())) for pair in self.offset_mapping[i]]
                token_substrings = []
                last_end = 0  # furthest char index we've emitted
                for (start, end) in offsets:
                    if start == 0 and end == 0:  # padding
                        token_substrings.append("")
                        continue
                    # Only emit chars past what we've already emitted
                    emit_start = max(start, last_end)
                    if emit_start >= end:
                        token_substrings.append("")  # nothing new
                    else:
                        token_substrings.append(text[emit_start:end])
                    last_end = max(last_end, end)
                    
                all_token_substrings.append(token_substrings)
        else:
            raise RuntimeError('Unsupported tokenizer: not fast, cannot build offset_mapping equivalents.')

        return all_token_substrings

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'original_tokens': self.original_tokens[idx],
        }
        for k, v in self._ident_lists.items(): # Attach metadata
            item[k] = v[idx]
        return item
    
def stack_collate(batch):
    """
    Custom collate function; returns everything in a dataset as a list except tensors, which are stacked. Use for creating a dataloader from ReconstructableTextDataset.
    """
    stacked = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        stacked[k] = torch.stack(vals, dim = 0) if torch.is_tensor(vals[0]) else vals
        
    return stacked


################### Helpers for running forward passes and reshaping data ###################
@torch.no_grad()
def convert_outputs_to_df_fast(input_ids: torch.Tensor, attention_mask: torch.Tensor, output_logits: torch.Tensor) -> pd.DataFrame:
    """
    Create a sample (token) level dataframe from inputs ids and output logits. Skips positions where attention_mask == 0 (i.e., padding).

    Params:
        @input_ids: A tensor of input ids of size B x N
        @attention_mask: A tensor of 1 for real tokens, 0 for padding, of size B x N
        @output_logits: A B x N x V tensor of output logits

    Returns:
        A dataframe at `sequence_ix` x `token_ix` level, excluding masked tokens, with columns:
        - `sequence_ix`: Which sample in the batch.
        - `token_ix`: Token index within that sample.
        - `token_id`: The input token ID at that `sequence_ix` x `token_ix`.
        - `output_id`: Argmax of the output logits (the predicted token) associated with that `sequence_ix` x `token_ix`.
        - `output_prob`: The probability (softmax) of the output.

    Example:
        prompt = 'Hello'
        inputs = tokenizer(prompt, return_tensors = 'pt').to(main_device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            output = model(input_ids, attention_mask)

        convert_outputs_to_df_fast(input_ids, attention_mask, output['logits'])
    """
    logits = output_logits.to(torch.float32)
    B, N, V = logits.shape

    # Mask indices (boolean) and their coordinates
    attn = (attention_mask != 0)
    seq_ix, tok_ix = attn.nonzero(as_tuple = True)

    # Top-1 id and logit at each [B, N]
    top_logit, top_id = logits.max(dim = -1)

    # Compute top-1 probability without building full softmax; p_max = exp(max_logit - logsumexp(logits))
    lse = torch.logsumexp(logits, dim = -1)
    top_prob = (top_logit - lse).exp()

    # Slice only valid positions
    token_id = input_ids[attn]
    output_id = top_id[attn]
    output_prob = top_prob[attn]

    # Move small, final arrays to CPU once
    seq_ix = seq_ix.cpu().numpy()
    tok_ix = tok_ix.cpu().numpy()
    token_id = token_id.cpu().numpy()
    output_id = output_id.cpu().numpy()
    output_prob = output_prob.cpu().numpy().round(2)

    # Build DataFrame
    df = pd.DataFrame({
        "sequence_ix": seq_ix,
        "token_ix": tok_ix,
        "token_id": token_id,
        "output_id": output_id,
        "output_prob": output_prob,
    })
    return df



@torch.no_grad()
def run_and_export_states(model, tokenizer, *, run_model_return_states, dl: ReconstructableTextDataset, layers_to_keep_acts: list[int], extraction_key: str = 'all_pre_mlp_hidden_states'):
    """
    Run forward passes on given model and store the decomposed sample_df plus hidden states

    Params:
        @model: The model to run forward passes on via `run_model_return_states`. Should return a dict with keys `logits` and `all_pre_mlp_hidden_states`.
        @tokenizer: The tokenizer object corresponding to the model.
        @run_model_return_states: A function that runs the model and returns a dict with keys `logits` and `all_pre_mlp_hidden_states`.
        @dl: A ReconstructableTextDataset of which returns `input_ids`, `attention_mask`, `original_tokens`, and `prompt_ix`.
        @layers_to_keep_acts: A list of layer indices (0-indexed) for which to filter `all_pre_mlp_hidden_states` (see returned object description).
        @extract: The key in the output of `run_model_return_states` to extract hidden states from.

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
            loss = ForCausalLMLoss(output['logits'], torch.where(input_ids == tokenizer.pad_token_id, torch.tensor(-100), input_ids), output['logits'].size(-1)).detach().cpu().item()
            for i in range(min(20, input_ids.size(0))):
                decoded_input = tokenizer.decode(input_ids[i, :], skip_special_tokens = False)
                next_token_id = torch.argmax(output['logits'][i, -1, :]).item()
                print('---------\n' + decoded_input + tokenizer.decode([next_token_id], skip_special_tokens = False).replace('\n', '<lb>'))
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
        all_hidden_states.append(torch.stack(output[extraction_key], dim = 1)[valid_pos][:, layers_to_keep_acts, :])

    sample_df = pd.concat(sample_dfs, ignore_index = True).drop(columns = ['batch_ix', 'sequence_ix']) # Drop batch/seq_ix, since prompt_ix identifies
    all_hidden_states = torch.cat(all_hidden_states, dim = 0)

    return {
        'sample_df': sample_df,
        'all_hs': all_hidden_states
    }


################### Helper for flagging the role of each token in a sequence (or multiple sequences concatenated together) ###################
def label_gptoss_content_roles(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Label gpt-oss (Harmony) token streams with content-only roles and role segments.

    Returns:
        - role: one of {system, developer, user, assistant, cot, tool_call, tool} or None
        - is_content: bool
        - seg_ix: int or None
        - token_in_seg_ix: int or None
    """
    required = {"prompt_ix", "token_ix", "token"}
    missing = required - set(sample_df.columns)
    if missing:
        raise ValueError(f"sample_df missing required columns: {sorted(missing)}")

    d = (sample_df.sort_values(["prompt_ix", "token_ix"]).reset_index(drop = True).copy())

    # ---- Message segmentation + content-span detection (literal token matching) ----
    d["msg_seg_id"] = d.groupby("prompt_ix")["token"].transform(lambda s: (s == "<|start|>").cumsum())
    d["is_message"] = d["token"].eq("<|message|>")
    d["is_close"] = d["token"].isin(["<|end|>", "<|return|>", "<|call|>"])

    d["after_msg"] = d.groupby(["prompt_ix", "msg_seg_id"])["is_message"].cumsum().gt(0)
    d["before_end"] = d.groupby(["prompt_ix", "msg_seg_id"])["is_close"].cumsum().eq(0)

    # This is exactly the old in_content_span semantics.
    d["is_content"] = d["after_msg"] & d["before_end"] & ~d["is_message"]

    # ---- Header extraction and segment-role classification ----
    # Header tokens: tokens in [<|start|>, <|message|>) excluding <|start|>.
    d["is_header_tok"] = (d["msg_seg_id"] > 0) & (~d["after_msg"]) & (d["token"] != "<|start|>")

    def classify_header(tok_series: pd.Series):
        toks = tok_series.tolist()
        if not toks:
            return None

        toks_lc = [t.lower() for t in toks]
        header = "".join(toks_lc)
        header_ns = header.replace(" ", "")  # defensive, in case any tokens contain spaces

        # Tool output: functions.<tool_name> ...
        if header_ns.startswith("functions."):
            return "tool"

        # Tool call: assistant ... to=functions.<tool> ...
        if header_ns.startswith("assistant") and "to=functions" in header_ns:
            return "tool_call"

        # Assistant channels (no fallback)
        if header_ns.startswith("assistant") and "<|channel|>analysis" in header_ns:
            return "cot"
        if header_ns.startswith("assistant") and (
            "<|channel|>final" in header_ns or "<|channel|>commentary" in header_ns
        ):
            return "assistant"

        # System / user / developer
        if header_ns.startswith("user"):
            return "user"
        if header_ns.startswith("system"):
            return "system"
        if header_ns.startswith("developer"):
            return "developer"

        return None

    seg_roles = (
        d.loc[d["is_header_tok"]]
         .groupby(["prompt_ix", "msg_seg_id"])["token"]
         .agg(classify_header)
         .rename("segment_role")
    )
    d = d.merge(seg_roles, on=["prompt_ix", "msg_seg_id"], how="left")

    # Role is only assigned inside content spans.
    d["role"] = np.where(d["is_content"], d["segment_role"], None)

    # ---- seg_ix + token_in_seg_ix (only for labeled content tokens) ----
    is_labeled = d["is_content"] & d["role"].notna()

    prev_is_labeled = is_labeled.groupby(d["prompt_ix"]).shift(1, fill_value=False)
    prev_role = d.groupby("prompt_ix")["role"].shift(1)

    is_new_seg = is_labeled & (~prev_is_labeled | (d["role"] != prev_role))
    seg_counter = is_new_seg.groupby(d["prompt_ix"]).cumsum()  # 1,2,3,... at starts

    d["seg_ix"] = np.where(is_labeled, seg_counter - 1, None)

    d["token_in_seg_ix"] = None
    d.loc[is_labeled, "token_in_seg_ix"] = (
        d.loc[is_labeled].groupby(["prompt_ix", "seg_ix"]).cumcount()
    )

    # ---- Final cleanup / ordering ----
    out = d.drop(
        columns=["msg_seg_id", "is_message", "is_close", "after_msg", "before_end", "is_header_tok", "segment_role",],
        errors="ignore",
    )

    out = out.sort_values(["prompt_ix", "token_ix"]).reset_index(drop=True)
    return out


################### CHECK MEMORY ###################
def check_memory():
    """
    Check memory of all CUDA devices
    """
    device_count = torch.cuda.device_count()
    if device_count == 0:
        print("No CUDA devices found.")
        return

    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Allocated: {torch.cuda.memory_allocated(i)/1024/1024/1024:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(i)/1024/1024/1024:.2f} GB")
        print(f"  Total: {torch.cuda.get_device_properties(i).total_memory/1024/1024/1024:.2f} GB")
        print()

def clear_all_cuda_memory(verbose = True):
    """
    Clear all CUDA memory
    """
    # Ensure all CUDA operations are complete
    torch.cuda.synchronize()
    
    # Empty the cache on all devices
    for device_id in range(torch.cuda.device_count()):
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    
    # Clear references to any tensors and force garbage collection
    gc.collect()
    
    # Optionally, reset the CUDA context (commented out as it's more drastic and may not always be necessary)
    # for device_id in range(torch.cuda.device_count()):
    #     torch.cuda.reset()
    if verbose:
        print("All CUDA memory cleared on all devices.")
