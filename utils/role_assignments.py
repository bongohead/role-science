"""
Helper functions to assign tokens to their roles
"""
import pandas as pd
import numpy as np

def label_gptoss_content_roles(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a token-levle df, labels each token with its role only within the content span. Makes no assumption about the number of messages in the sequence.
    
    Notes:
        - Segments are delimited by: <|start|> ... <|message|> CONTENT ... (<|end|> | <|return|> | <|call|>)
        - Roles are inferred from the header between <|start|> and <|message|>.
        - Tool-calls (assistant → functions) and tool outputs (functions.* → assistant) are supported.

    Params: 
        @sample_df: A dataframe with the following columns: prompt_ix, token_ix, token.
         Prompt_ix represents a global index of the sequence, equivalent to an index on (batch_ix, sequence_ix).

    Returns:
        The original df with new columns:
        - seg_id: segment id within prompt
        - in_content_span: bool
        - role: str in {system, user, developer, assistant-cot, assistant-final, assistant-commentary, tool} or NaN
    """
    res = (\
        sample_df
        .sort_values(['prompt_ix', 'token_ix'])

        # Segmenting and closers
        .assign(
            seg_id = lambda d: d.groupby('prompt_ix')['token'].transform(lambda s: (s == '<|start|>').cumsum()), # Each segment (message)
            is_message = lambda d: d['token'].eq('<|message|>'),
            is_close = lambda d: d['token'].isin(['<|end|>', '<|return|>', '<|call|>'])
        )
        .assign(
            after_msg = lambda d: d.groupby(['prompt_ix','seg_id'])['is_message'].cumsum().gt(0),
            before_end = lambda d: d.groupby(['prompt_ix','seg_id'])['is_close'].cumsum().eq(0),
            in_content_span = lambda d: d['after_msg'] & d['before_end'] & ~d['is_message']
        )

        # Header reconstruction (join tokens between <|start|> and <|message|>)
        .assign(token_hdr = lambda d: d['token'].where((d['seg_id'] > 0) & ~d['after_msg'] & ~d['token'].eq('<|start|>')))
        .pipe(lambda d: d.merge(
            d.groupby(['prompt_ix','seg_id'])['token_hdr'].agg(lambda s: ''.join(s.dropna().tolist())).rename('header'),
            on = ['prompt_ix','seg_id'], how = 'left'
        ))
        .drop(columns = ['token_hdr'])
        .assign(header = lambda d: d['header'].fillna('').str.lower())

        # Role classification from header
        .assign(
            segment_role = lambda d: np.select(
                [
                    d['header'].str.startswith('functions.'), # tool output
                    d['header'].str.startswith('assistant') & d['header'].str.contains('to=functions'), # tool call
                    d['header'].str.contains('<\\|channel\\|>analysis'),
                    d['header'].str.contains('<\\|channel\\|>final'),
                    d['header'].str.contains('<\\|channel\\|>commentary'),
                    d['header'].str.startswith('user'),
                    d['header'].str.startswith('system'),
                    d['header'].str.startswith('developer'),
                ],
                [
                    'tool',
                    'assistant-commentary', # Any assistant tool call
                    'assistant-cot',
                    'assistant-final',
                    'assistant-commentary', # Any non-tool commentary
                    'user',
                    'system',
                    'developer',
                ],
                default = None
            )
        )

        # Apply role ONLY inside the content span
        .assign(role = lambda d: np.where(d['in_content_span'], d['segment_role'], None))\
            
        # Optional cleanup of helpers
        .drop(columns = ['is_message', 'is_close', 'after_msg', 'before_end', 'header', 'segment_role'], errors = 'ignore')
        .reset_index(drop = True)
    )
    
    return res


def label_qwen3_content_roles(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Label tokens with Qwen3 roles, content-only and multi-message safe.

    Notes:
        - Message layout:
            <|im_start|>assistant\\n [<think>…</think>] [visible text] [zero+ <tool_call>…</tool_call>] <|im_end|>
            Tool results:
            <|im_start|>user\\n <tool_response>…</tool_response> [more …] <|im_end|>

    Params: 
        @sample_df: A dataframe with the following columns: prompt_ix, token_ix, token (and optionally batch_ix, sequence_ix for sample_ix).
         Prompt_ix represents a global index of the sequence, equivalent to an index on (batch_ix, sequence_ix).

    Returns:
        The original df with new columns:
        - seg_id: segment id within prompt
        - in_content_span: bool
        - role: str in {system, user, developer, assistant-cot, assistant-final, assistant-commentary, tool} or NaN
    """
    IM_START, IM_END = '<|im_start|>', '<|im_end|>'
    OPEN_THINK, CLOSE_THINK = '<think>', '</think>'
    OPEN_TCALL, CLOSE_TCALL = '<tool_call>', '</tool_call>'
    OPEN_TRESP, CLOSE_TRESP = '<tool_response>', '</tool_response>'

    # Newline Qwen3 tokenizers
    NL_PATTERN = r'[\nĊĉĈ]+'
    return (
        sample_df
        .sort_values(['prompt_ix', 'token_ix'])

        # Segment boundaries & basic markers
        .assign(
            seg_id = lambda d: d.groupby('prompt_ix')['token'].transform(lambda s: (s == IM_START).cumsum()),
            is_start = lambda d: d['token'].eq(IM_START),
            is_end = lambda d: d['token'].eq(IM_END),
            has_nl = lambda d: d['token'].str.contains(NL_PATTERN, regex=True),
        )

        # Header/body split: header ends at the first newline after <|im_start|>role
        .assign(
            nl_cum = lambda d: d.groupby(['prompt_ix','seg_id'])['has_nl'].cumsum(),
            is_first_nl = lambda d: d['has_nl'] & d['nl_cum'].eq(1),
            before_header = lambda d: d['nl_cum'].eq(0),

            # header tokens: everything after <|im_start|> up to (and including) the first newline
            is_header_token = lambda d: (d['seg_id'] > 0) & ~d['is_start'] & (d['before_header'] | d['is_first_nl']),

            before_end = lambda d: d.groupby(['prompt_ix','seg_id'])['is_end'].cumsum().eq(0),

            # Body tokens = inside a message, after header, before <|im_end|>
            in_body = lambda d: (d['seg_id'] > 0) & ~d['is_header_token'] & d['before_end'],
        )

        # Nested regions: <think>, <tool_call>, <tool_response>
        .assign(
            is_think_open = lambda d: d['token'].eq(OPEN_THINK),
            is_think_close = lambda d: d['token'].eq(CLOSE_THINK),
            is_tcall_open = lambda d: d['token'].eq(OPEN_TCALL),
            is_tcall_close = lambda d: d['token'].eq(CLOSE_TCALL),
            is_tresp_open = lambda d: d['token'].eq(OPEN_TRESP),
            is_tresp_close = lambda d: d['token'].eq(CLOSE_TRESP),
        )
        .assign(
            think_open_cum = lambda d: d.groupby(['prompt_ix','seg_id'])['is_think_open'].cumsum(),
            think_close_cum = lambda d: d.groupby(['prompt_ix','seg_id'])['is_think_close'].cumsum(),
            in_think = lambda d: d['think_open_cum'] > d['think_close_cum'],

            tcall_open_cum = lambda d: d.groupby(['prompt_ix','seg_id'])['is_tcall_open'].cumsum(),
            tcall_close_cum = lambda d: d.groupby(['prompt_ix','seg_id'])['is_tcall_close'].cumsum(),
            in_tool_call = lambda d: d['tcall_open_cum'] > d['tcall_close_cum'],

            tresp_open_cum = lambda d: d.groupby(['prompt_ix','seg_id'])['is_tresp_open'].cumsum(),
            tresp_close_cum = lambda d: d.groupby(['prompt_ix','seg_id'])['is_tresp_close'].cumsum(),
            in_tool_resp = lambda d: d['tresp_open_cum'] > d['tresp_close_cum'],
        )

        # Build the role header between <|im_start|> and the first newline (robust to newline sentinels)
        .assign(
            header_piece = lambda d: np.select(
                [
                    (d['seg_id'] > 0) & ~d['is_start'] & d['before_header'] & ~d['has_nl'],
                    (d['seg_id'] > 0) & ~d['is_start'] & d['is_first_nl'],
                ],
                [
                    d['token'],
                    d['token'].str.split('\n', n=1, regex=False).str[0],
                ],
                default=None
            )
        )
        .pipe(lambda d: d.merge(
            d.groupby(['prompt_ix','seg_id'])['header_piece']
                .agg(lambda s: ''.join([x for x in s.dropna().tolist()]))
                .rename('header_line'),
            on = ['prompt_ix','seg_id'], how = 'left'
        ))
        .drop(columns=['header_piece'])
        .assign(header_line = lambda d: d['header_line'].fillna('').str.lower())

        # Coarse segment kind from header
        .assign(
            seg_kind = lambda d: np.select(
                [
                    d['header_line'].str.startswith('assistant'),
                    d['header_line'].str.startswith('user'),
                    d['header_line'].str.startswith('system'),
                    d['header_line'].str.startswith('developer'),
                ],
                ['assistant', 'user', 'system', 'developer'],
                default = None
            )
        )

        # ----- Structural-whitespace filtering -----
        # 1. Any newline tokens INSIDE <think>…</think> are non-content.
        # 2. The entire RUN of newline tokens immediately AFTER </think> and BEFORE the first non-newline token in the same assistant 
        #  message is non-content.
        .assign(
            # First, a base tag mask (wrappers we never label)
            is_tag0 = lambda d: d[['is_start','is_end',
                                   'is_think_open','is_think_close',
                                   'is_tcall_open','is_tcall_close',
                                   'is_tresp_open','is_tresp_close']].any(axis=1),

            # Identify "post-think" zones and whether we've seen real (non-newline, non-tag) content in them yet
            post_think_run = lambda d: d.groupby(['prompt_ix','seg_id'])['is_think_close'].cumsum(),
            in_post_think = lambda d: (d['post_think_run'] > 0) & d['seg_kind'].eq('assistant') & d['in_body'],
            non_ws_visible = lambda d: d['in_post_think'] & ~d['has_nl'] & ~d['is_tag0'],  # first visible content after </think>
            seen_visible = lambda d: d.groupby(['prompt_ix','seg_id','post_think_run'])['non_ws_visible'].cumsum().gt(0),

            # Newlines inside think OR in the immediate gap after </think> (before first visible content)
            ws_in_think = lambda d: d['in_think'] & d['has_nl'],
            ws_after_think = lambda d: d['in_post_think'] & d['has_nl'] & ~d['seen_visible'],

            # Final tag mask includes structural whitespace
            is_tag = lambda d: d['is_tag0'] | d['ws_in_think'] | d['ws_after_think'],

            # Body minus wrappers => content span
            in_content_span = lambda d: d['in_body'] & ~d['is_tag'],
        )

        # Final role per token
        .assign(
            role = lambda d: np.select(
                [
                    (d['seg_kind'].eq('assistant')) & d['in_content_span'] & d['in_think'],
                    (d['seg_kind'].eq('assistant')) & d['in_content_span'] & d['in_tool_call'],
                    (d['seg_kind'].eq('assistant')) & d['in_content_span'] & ~d['in_think'] & ~d['in_tool_call'],

                    (d['seg_kind'].eq('user')) & d['in_content_span'] & d['in_tool_resp'],
                    (d['seg_kind'].eq('user')) & d['in_content_span'] & ~d['in_tool_resp'],

                    (d['seg_kind'].eq('system')) & d['in_content_span'],
                    (d['seg_kind'].eq('developer')) & d['in_content_span'],
                ],
                [
                    'assistant-cot',
                    'assistant-commentary',  # <tool_call> payload JSON
                    'assistant-final',
                    'tool',                  # <tool_response> payload JSON
                    'user',
                    'system',
                    'developer',
                ],
                default=None
            )
        )

        # Cleanup helpers
        .drop(
            columns = [
                'is_start','is_end','has_nl','nl_cum','is_first_nl','before_header','is_header_token','before_end','in_body',
                'is_think_open','is_think_close','is_tcall_open','is_tcall_close','is_tresp_open','is_tresp_close',
                'think_open_cum','think_close_cum','tcall_open_cum','tcall_close_cum','tresp_open_cum','tresp_close_cum',
                'header_line','seg_kind','is_tag0','post_think_run','in_post_think','non_ws_visible','seen_visible',
                'ws_in_think','ws_after_think'
            ],
            errors = 'ignore'
        )
        .reset_index(drop=True)
    )    

def label_glm4_content_roles(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Label tokens for GLM-4.5 / GLM-4.6 hybrid reasoning models, treating structural
    newlines the same way we do for Qwen3:

      * Newline after <|role|> sentinel (up to first newline) = header, non-content.
      * Newlines inside <think>...</think> = non-content.
      * Newlines in the immediate gap after </think> up to the first visible token
        in that 'post-think' run = non-content.
      * All other newlines are content.

    Assumes the official GLM-4.5/4.6 chat template, with single-token sentinels.

    Input:
        sample_df: must include columns
            - prompt_ix : global conversation index
            - token_ix  : position within that prompt
            - token     : decoded token string (single tokens)

    Output:
        original df +:
            - seg_id          : segment id within prompt (0 = prefix [gMASK]<sop>)
            - in_content_span : bool, True for non-wrapper tokens inside a segment
            - role            : one of {system, user, assistant-cot,
                                       assistant-final, assistant-commentary, tool}
                               or None for non-content / prefix.
    """

    # ---- Special tokens (assumed to be single tokens) ----
    SYSTEM     = '<|system|>'
    USER       = '<|user|>'
    ASSISTANT  = '<|assistant|>'
    OBS        = '<|observation|>'

    PREFIX_TOKENS = ['[gMASK]', '<sop>', '<eop>']

    THINK_OPEN, THINK_CLOSE   = '<think>', '</think>'
    TCALL_OPEN, TCALL_CLOSE   = '<tool_call>', '</tool_call>'
    TRESP_OPEN, TRESP_CLOSE   = '<tool_response>', '</tool_response>'
    ARGK_OPEN, ARGK_CLOSE     = '<arg_key>', '</arg_key>'
    ARGV_OPEN, ARGV_CLOSE     = '<arg_value>', '</arg_value>'
    NOTHINK                   = '/nothink'

    # Newline detection (same pattern you used for Qwen)
    NL_PATTERN = r'[\nĊĉĈ]+'

    df = sample_df.sort_values(['prompt_ix', 'token_ix']).copy()

    # ---- Segment boundaries: each GLM role sentinel starts a new segment ----
    df['is_seg_start'] = df['token'].isin([SYSTEM, USER, ASSISTANT, OBS])
    df['seg_id'] = df.groupby('prompt_ix')['is_seg_start'].cumsum()

    # seg_id == 0: prefix region (e.g. [gMASK]<sop>)

    # ---- Segment "kind": system / user / assistant / observation ----
    df['role_token'] = df['token'].where(df['is_seg_start'])
    df['seg_role_token'] = (
        df.groupby(['prompt_ix', 'seg_id'])['role_token']
          .transform('first')
          .fillna('')
    )

    df['seg_kind'] = np.select(
        [
            df['seg_role_token'].eq(ASSISTANT),
            df['seg_role_token'].eq(USER),
            df['seg_role_token'].eq(SYSTEM),
            df['seg_role_token'].eq(OBS),
        ],
        ['assistant', 'user', 'system', 'observation'],
        default=None,
    )

    # ---- Newline & "header" detection (Qwen-style) ----
    df['has_nl'] = df['token'].str.contains(NL_PATTERN, regex=True)

    by_seg = df.groupby(['prompt_ix', 'seg_id'], sort=False)

    df['nl_cum'] = by_seg['has_nl'].cumsum()
    df['is_first_nl'] = df['has_nl'] & df['nl_cum'].eq(1)
    df['before_header'] = df['nl_cum'].eq(0)

    # Header tokens: everything after the role sentinel up to (and including) the
    # first newline in that segment. For GLM this usually captures the single
    # '\n' right after <|role|>.
    df['is_header_token'] = (
        (df['seg_id'] > 0)
        & ~df['is_seg_start']
        & (df['before_header'] | df['is_first_nl'])
    )

    # ---- Tag markers for nested spans ----
    df['is_think_open'] = df['token'].eq(THINK_OPEN)
    df['is_think_close'] = df['token'].eq(THINK_CLOSE)

    df['is_tcall_open'] = df['token'].eq(TCALL_OPEN)
    df['is_tcall_close'] = df['token'].eq(TCALL_CLOSE)

    df['is_tresp_open'] = df['token'].eq(TRESP_OPEN)
    df['is_tresp_close'] = df['token'].eq(TRESP_CLOSE)

    df['is_argk_open'] = df['token'].eq(ARGK_OPEN)
    df['is_argk_close'] = df['token'].eq(ARGK_CLOSE)

    df['is_argv_open'] = df['token'].eq(ARGV_OPEN)
    df['is_argv_close'] = df['token'].eq(ARGV_CLOSE)

    df['is_nothink'] = df['token'].eq(NOTHINK)

    # ---- Nested span membership: <think>..., <tool_call>... ----
    df['think_open_cum'] = by_seg['is_think_open'].cumsum()
    df['think_close_cum'] = by_seg['is_think_close'].cumsum()
    df['in_think'] = df['think_open_cum'] > df['think_close_cum']

    df['tcall_open_cum'] = by_seg['is_tcall_open'].cumsum()
    df['tcall_close_cum'] = by_seg['is_tcall_close'].cumsum()
    df['in_tool_call'] = df['tcall_open_cum'] > df['tcall_close_cum']

    # ---- "Body" tokens: inside a segment, past its header ----
    df['in_body'] = (df['seg_id'] > 0) & ~df['is_header_token']

    # ---- Baseline "tag" mask (non-content wrappers & control tokens) ----
    df['is_prefix'] = df['token'].isin(PREFIX_TOKENS) & df['seg_id'].eq(0)
    df['is_role_sentinel'] = df['token'].isin([SYSTEM, USER, ASSISTANT, OBS])

    df['is_tag0'] = df[
        [
            'is_prefix',
            'is_role_sentinel',
            'is_think_open', 'is_think_close',
            'is_tcall_open', 'is_tcall_close',
            'is_tresp_open', 'is_tresp_close',
            'is_argk_open', 'is_argk_close',
            'is_argv_open', 'is_argv_close',
            'is_nothink',
        ]
    ].any(axis=1)

    # ---- Structural whitespace (Qwen-style) ----
    # 1) Newlines inside <think>...</think>
    df['post_think_run'] = by_seg['is_think_close'].cumsum()
    df['in_post_think'] = (
        (df['post_think_run'] > 0)
        & df['seg_kind'].eq('assistant')
        & df['in_body']
    )
    df['non_ws_visible'] = df['in_post_think'] & ~df['has_nl'] & ~df['is_tag0']
    df['seen_visible'] = (
        df.groupby(['prompt_ix', 'seg_id', 'post_think_run'])['non_ws_visible']
          .cumsum()
          .gt(0)
    )

    df['ws_in_think'] = df['in_think'] & df['has_nl']
    df['ws_after_think'] = df['in_post_think'] & df['has_nl'] & ~df['seen_visible']

    df['is_tag'] = df['is_tag0'] | df['ws_in_think'] | df['ws_after_think']

    # ---- Content span ----
    df['in_content_span'] = df['in_body'] & ~df['is_tag']

    # ---- Final role per token ----
    df['role'] = np.select(
        [
            # Assistant reasoning
            df['seg_kind'].eq('assistant') & df['in_content_span'] & df['in_think'],

            # Assistant tool call (function name + arg_key/arg_value/JSON)
            df['seg_kind'].eq('assistant') & df['in_content_span'] & df['in_tool_call'],

            # Assistant visible answer (not in think, not in tool_call)
            df['seg_kind'].eq('assistant') & df['in_content_span']
            & ~df['in_think'] & ~df['in_tool_call'],

            # User messages
            df['seg_kind'].eq('user') & df['in_content_span'],

            # System messages (includes tools preamble)
            df['seg_kind'].eq('system') & df['in_content_span'],

            # Tool outputs (observation segments)
            df['seg_kind'].eq('observation') & df['in_content_span'],
        ],
        [
            'assistant-cot',
            'assistant-commentary',
            'assistant-final',
            'user',
            'system',
            'tool',
        ],
        default=None,
    )

    # ---- Cleanup helper columns ----
    drop_cols = [
        'is_seg_start', 'role_token', 'seg_role_token',
        'has_nl', 'nl_cum', 'is_first_nl', 'before_header', 'is_header_token',
        'is_think_open', 'is_think_close',
        'is_tcall_open', 'is_tcall_close',
        'is_tresp_open', 'is_tresp_close',
        'is_argk_open', 'is_argk_close',
        'is_argv_open', 'is_argv_close',
        'is_nothink',
        'think_open_cum', 'think_close_cum',
        'tcall_open_cum', 'tcall_close_cum',
        'in_body', 'is_prefix', 'is_role_sentinel', 'is_tag0',
        'post_think_run', 'in_post_think', 'non_ws_visible', 'seen_visible',
        'ws_in_think', 'ws_after_think',
    ]
    df = df.drop(columns=drop_cols, errors='ignore')

    return df.reset_index(drop=True)



def label_content_roles(model_architecture, sample_df):
    """
    Takes a token-level df, labels each token with its role only within the content span. Makes no assumption about the number of messages in the sequence.
    
    Params: 
        @sample_df: A dataframe with the following columns: prompt_ix, token_ix, token.
         Prompt_ix represents a global index of the sequence, equivalent to an index on (batch_ix, sequence_ix).

    Returns:
        The original df with new columns:
        - seg_id: segment id within prompt
        - in_content_span: bool
        - role: str in {system, user, developer, assistant-cot, assistant-final, assistant-commentary, tool} or NaN
    """
    if model_architecture == 'gptoss':
        return label_gptoss_content_roles(sample_df)
    elif model_architecture == 'qwen3moe':
        return label_qwen3_content_roles(sample_df)
    elif model_architecture == 'glm4moe':
        return label_glm4_content_roles(sample_df)

    else:
        raise ValueError(f"Model prefix {model_architecture} not supported")