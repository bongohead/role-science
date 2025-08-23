import torch
from torch.utils.data import Dataset
import tiktoken

class ReconstructableTextDataset(Dataset):

    def __init__(self, raw_texts: list[str], tokenizer, max_length, **identifiers):
        """
        Creates a dataset object that contains the usual input_ids and attention_mask, but also returns a B-length list of the original tokens 
         in the same position as the input ids, as well as any optional identifiers. Returning the original tokens is important for BPE 
         tokenizers as otherwise it's difficult to reconstruct the correct string later! Currently supports only fast tokenizers (most models), 
         or tiktoken-based tokenizers.

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
        tokenized = tokenizer(
            raw_texts,
            add_special_tokens = False,
            max_length = max_length,
            padding = 'max_length',
            truncation = True,
            return_offsets_mapping = getattr(tokenizer, 'is_fast', False), # True for fast tokenizers
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
            for i in range(0, self.input_ids.shape[0]):
                token_substrings = []
                for j in range(self.input_ids.shape[1]): 
                    start_char, end_char = self.offset_mapping[i][j].tolist()
                    if start_char == 0 and end_char == 0: # When pads, offset_mapping might be [0, 0], so let's store an empty string for those positions.
                        token_substrings.append("")
                    else:
                        token_substrings.append(texts[i][start_char:end_char])

                all_token_substrings.append(token_substrings)

        # Tiktoken tokenizers
        elif hasattr(self.tokenizer, 'model') and isinstance(self.tokenizer.model, tiktoken.Encoding):
            for seq_ids, text in zip(self.input_ids, texts):
                offsets = self._build_offsets_with_tiktoken(seq_ids.tolist())
                token_substrings = ['' if (s == 0 and e == 0) else text[s:e] for s, e in offsets]
                all_token_substrings.append(token_substrings)
        
        else:
            raise RuntimeError('Unsupported tokenizer: not fast and not tiktoken, cannot build offset_mapping equivalents.')

        return all_token_substrings

    # Tiktoken offsetter
    def _build_offsets_with_tiktoken(self, ids: list[int]):
        start = 0
        offsets = []
        for tid in ids:
            if tid == self.tokenizer.pad_token_id or tid in self.tokenizer.all_special_ids:
                offsets.append((0, 0)) # PAD/special -> empty span
                continue
            token_str = self.tokenizer.model.decode([tid])
            end = start + len(token_str)
            offsets.append((start, end))
            start = end
        return offsets

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
    Custom collate function; returns everything in a dataset as a list except tensors, which are stacked. 
    """
    stacked = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        stacked[k] = torch.stack(vals, dim = 0) if torch.is_tensor(vals[0]) else vals
        
    return stacked
