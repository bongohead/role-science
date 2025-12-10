"""
Helper functions to assign tokens to substrings
"""
import pandas as pd
import numpy as np
import bisect
from itertools import accumulate

def flag_message_types(sample_level_df: pd.DataFrame, base_messages: list[str], allow_ambiguous: bool = False) -> pd.DataFrame:
    """
    Take a token-level df with columns for prompt_ix, token; identify whether each token is one of base_messages.
    
    Description:
        High accuracy exact match version
        For each prompt_ix:
            1) Concatenate tokens (in token_ix order) -> full_text
            2) For each base_message t in base_messages:
                - find all exact occurrences of t in full_text
                - for each occurrence, mark all tokens whose character spans overlap t
            3) For each token:
                - if it belongs to exactly one base_message -> base_message_ix = that index, ambiguous=False
                - if it belongs to >1 base_messages       -> base_message_ix = None, ambiguous=True
                - else                                    -> base_message_ix = None, ambiguous=False
        Notes:
            - Tokens are concatenated *without* separators, i.e. "ab" + "cd" -> "abcd". A base_message like "bc" could match across token boundaries. 
              If you want to avoid that, ensure tokens include appropriate leading/trailing spaces or other delimiters.
            - If the base_message_type is "sure, here is " and the full string is "Sure here is dog", and the space is tokenized with " dog", 
              then " dog" will be included too.
            - Be careful to AVOID base_messages which are overlapping ("Sure, here is ", "Sure, here is a detailed answer:"): these means that 
              matches can be ambiguous, which raises a ValueError unless allow_ambiguous=True.

    Params:
        @sample_level_df: A token-level dataframe with columns for `prompt_ix`, `token_ix` (ordering within prompt), `token` (string token)
        @base_messages: A list of base messages to match tokens to
        @allow_ambiguous: If True, matches to the first base message when ambiguous

    Returns:
        The original sample_level_df, with and additional column `base_message_ix` equal to the index
        of the matching base_message (None for no match).
    """
    df = sample_level_df.sort_values(['prompt_ix', 'token_ix']).reset_index(drop=True)

    # Early exit if nothing to match
    if not base_messages:
        df['base_message_ix'] = None
        df['ambiguous'] = False
        df['base_message'] = None
        return df

    n_rows = len(df)
    memberships: list[set[int]] = [set() for _ in range(n_rows)]

    # Process each prompt separately
    for _, g in df.groupby('prompt_ix', sort=False):
        idx = g.index.to_numpy()
        tokens = g['token'].astype(str).tolist()

        # Build full text and token spans
        token_lens = [len(t) for t in tokens]
        token_ends = list(accumulate(token_lens))
        token_starts = [0] + token_ends[:-1]
        full_text = ''.join(tokens)
        last_token_end = token_ends[-1] if token_ends else 0

        # For each base_message, find all exact matches
        for bm_ix, bm in enumerate(base_messages):
            if not bm:
                continue  # skip empty templates

            bm_len = len(bm)
            start_pos = 0

            while (match_at := full_text.find(bm, start_pos)) != -1:
                match_end = match_at + bm_len

                # Optimization: if match starts beyond last token, nothing else to do
                if match_at >= last_token_end:
                    break

                # Find overlapping tokens via binary search
                first = bisect.bisect_right(token_ends, match_at)
                last = bisect.bisect_left(token_starts, match_end)

                for local_token_ix in range(first, last):
                    memberships[idx[local_token_ix]].add(bm_ix)

                # Move forward to find additional (possibly overlapping) matches
                start_pos = match_at + 1

    # Convert membership sets into output columns
    base_message_ix_col: list[int | None] = []

    for row_ix, s in enumerate(memberships):
        if len(s) > 1:
            if allow_ambiguous:
                # Assign to the first base_message by index (order in base_messages)
                base_message_ix_col.append(min(s))
            else:
                token = df.loc[row_ix, 'token']
                matched = [base_messages[i] for i in s]
                raise ValueError(
                    f"Ambiguous match at row {row_ix}, token {token!r}: "
                    f"matched {len(s)} base_messages: {matched}"
                )
        elif len(s) == 1:
            base_message_ix_col.append(next(iter(s)))
        else:
            base_message_ix_col.append(None)

    df['base_message_ix'] = base_message_ix_col
    df['base_message'] = [
        base_messages[i] if i is not None else None
        for i in base_message_ix_col
    ]

    return df


def flag_message_types_dep(sample_level_df, base_messages):
    """
    Take a token-level df with columns for prompt_ix, token; identify whether each token is one of base_messages.
    
    Params:
        @sample_level_df: A token-level dataframe with columns for `prompt_ix` and `token`
        @base_messages: A list of base messages to match tokens to

    Returns:
        The original sample_level_df, with and additional column `base_message_ix` equal to the index
        of the matching base_message (None for no match).
    """
    res = (
        sample_level_df\
        .sort_values(['prompt_ix', 'token_ix'])\
        .assign(
            _t1 = lambda d: d.groupby('prompt_ix')['token'].shift(-1),
            _t2 = lambda d: d.groupby('prompt_ix')['token'].shift(-2),
            _t3 = lambda d: d.groupby('prompt_ix')['token'].shift(-3),
            _t4 = lambda d: d.groupby('prompt_ix')['token'].shift(-4),
            _t5 = lambda d: d.groupby('prompt_ix')['token'].shift(-5),
            _t6 = lambda d: d.groupby('prompt_ix')['token'].shift(-6),
            _t7 = lambda d: d.groupby('prompt_ix')['token'].shift(-7),
            _t8 = lambda d: d.groupby('prompt_ix')['token'].shift(-8),
            _b1 = lambda d: d.groupby('prompt_ix')['token'].shift(1),
            _b2 = lambda d: d.groupby('prompt_ix')['token'].shift(2),
            _b3 = lambda d: d.groupby('prompt_ix')['token'].shift(3),
            _b4 = lambda d: d.groupby('prompt_ix')['token'].shift(4),
            _b5 = lambda d: d.groupby('prompt_ix')['token'].shift(5),
            _b6 = lambda d: d.groupby('prompt_ix')['token'].shift(6),
            _b7 = lambda d: d.groupby('prompt_ix')['token'].shift(7),
            _b8 = lambda d: d.groupby('prompt_ix')['token'].shift(8)
        )\
        .assign(
            has_roll = lambda d: d[['_t1','_t2','_t3','_t4','_t5', '_t6', '_t7', '_t8']].notna().all(axis = 1),
            has_back = lambda d: d[['_b1','_b2','_b3','_b4','_b5', '_b6', '_b7', '_b8']].notna().all(axis = 1),  
        )\
        .assign(
            tok_roll = lambda d: d['token'].fillna('') + d['_t1'].fillna('') + d['_t2'].fillna('') + d['_t3'].fillna('') + d['_t4'].fillna('') +
                d['_t5'].fillna('')  + d['_t6'].fillna('')  + d['_t7'].fillna('')  + d['_t8'].fillna(''),
            tok_back = lambda d: d['_b8'].fillna('') + d['_b7'].fillna('') + d['_b6'].fillna('') + d['_b5'].fillna('') + d['_b4'].fillna('') +
                d['_b3'].fillna('') + d['_b2'].fillna('') + d['_b1'].fillna('') + d['token'].fillna('')
        )\
        .pipe(lambda d: d.join(
            pd.concat(
                [
                    (
                        (d['has_roll'] & d['tok_roll'].apply(lambda s, t=t: s in t)) |
                        (d['has_back'] & d['tok_back'].apply(lambda s, t=t: s in t))
                        # d['tok_roll'].apply(lambda s, t=t: bool(s) and (s in t)) |
                        # d['tok_back'].apply(lambda s, t=t: bool(s) and (s in t))
                    ).rename(f'hit_p{i}')
                    for i, t in enumerate(base_messages)
                ],
                axis = 1
            )
        ))\
        .assign(
            base_message_ix = lambda d: np.select(
                [d[f'hit_p{i}'] for i in range(len(base_messages))],
                list(range(len(base_messages))),
                default = None
            ),
            ambiguous = lambda d: d[[f'hit_p{i}' for i in range(len(base_messages))]].sum(axis = 1) > 1,
            # base_name = lambda d: d['base_ix'].map({i: f"p{i}" for i in range(len(base_messages))})
        )\
        .drop(columns=['_t1','_t2','_t3','_t4', '_t5','_t6', '_t7', '_t8', '_b1','_b2','_b3','_b4','_b5', '_b6', '_b7', '_b8'])\
        .drop(columns = [f'hit_p{i}' for i in range(len(base_messages))])\
        .drop(columns = ['tok_roll', 'tok_back', 'ambiguous'])\
    )

    return res

