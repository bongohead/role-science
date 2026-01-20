"""
Helper functions to assign tokens to their roles - these are largely LLM generated; they're validated carefully by checking token counts.
"""
import pandas as pd
import numpy as np
import re

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

def label_qwen3_content_roles(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Labels Qwen3-30B-A3B token streams with *content-only* roles.

    Qwen3 template (summary):
      - ChatML envelope: <|im_start|>ROLE\\n ... <|im_end|>\\n
      - Assistant may include <think>...</think> (not guaranteed for all turns)
      - Assistant tool calls: <tool_call> ... </tool_call>  (JSON payload inside)
      - Tool outputs are embedded in a ChatML user wrapper using:
          <tool_response> ... </tool_response>
      - add_generation_prompt typically appends: <|im_start|>assistant\\n<think>\\n

    Content-only roles produced:
      {system, user, assistant, cot, tool_call, tool} or None.

    Tag tokens (role=None) include:
      - ChatML envelope tokens: <|im_start|>, <|im_end|>
      - Role header tokens (role text + header newline)
      - Wrapper tags (<think>, </think>, <tool_call>, </tool_call>, <tool_response>, </tool_response>)
        (scoped so tool_call wrappers in SYSTEM instructions are not active)
      - Pure newline token immediately after <|im_end|> (template-attached)

    Required input columns:
      - prompt_ix, token_ix, token

    Returns: original columns + role, is_content, seg_ix, token_in_seg_ix
    """
    required = {"prompt_ix", "token_ix", "token"}
    missing = required - set(sample_df.columns)
    if missing:
        raise ValueError(f"sample_df missing required columns: {sorted(missing)}")

    df = (
        sample_df.sort_values(["prompt_ix", "token_ix"])
        .reset_index(drop=True)
        .copy()
    )

    tok = df["token"].astype(str)

    # -----------------------------
    # Sentinels / wrappers
    # -----------------------------
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"

    # Common BOS markers (not guaranteed; safe to treat as control if present)
    COMMON_BOS = {"<s>", "<|begin_of_text|>", "<|bos|>"}

    THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
    THINK_EMPTY = "<think></think>"

    TCALL_OPEN, TCALL_CLOSE = "<tool_call>", "</tool_call>"
    TRESP_OPEN, TRESP_CLOSE = "<tool_response>", "</tool_response>"

    # Newline detection supports both literal newlines and byte-BPE newline glyphs.
    NL_ONLY_RE = re.compile(r"^[\n\rĊĉĈ]+$")
    NL_ANY_RE = re.compile(r"[\n\rĊĉĈ]")

    STRIP_CHARS = "\n\rĊĉĈ \t"
    tok_stripped = tok.str.strip(STRIP_CHARS)

    # prev/next token (within prompt)
    df["prev_token"] = df.groupby("prompt_ix", sort=False)["token"].shift(1)
    df["next_token"] = df.groupby("prompt_ix", sort=False)["token"].shift(-1)

    df["is_pure_nl"] = tok.str.fullmatch(NL_ONLY_RE, na=False)
    df["has_nl_char"] = tok.str.contains(NL_ANY_RE, na=False)

    # -----------------------------
    # Outer ChatML message parsing
    # -----------------------------
    df["is_im_start"] = tok.eq(IM_START)
    df["is_im_end"] = tok.eq(IM_END)
    df["is_common_bos"] = tok.isin(COMMON_BOS)

    # Message id increments on each <|im_start|>
    df["msg_id"] = df.groupby("prompt_ix", sort=False)["is_im_start"].cumsum()
    df["pos_in_msg"] = df.groupby(["prompt_ix", "msg_id"], sort=False).cumcount()

    # header_end_pos = first token position that CONTAINS a newline glyph
    header_end_pos = (
        df["pos_in_msg"]
        .where((df["msg_id"] > 0) & df["has_nl_char"])
        .groupby([df["prompt_ix"], df["msg_id"]], sort=False)
        .transform("min")
    ).fillna(np.inf)
    df["header_end_pos"] = header_end_pos

    # If the header-ending token contains newline AND also has more characters after it,
    # treat that token as body (merged header+body). This preserves your "merged => content" rule.
    df["header_end_token_mixed"] = (
        (df["msg_id"] > 0)
        & (df["pos_in_msg"] == df["header_end_pos"])
        & tok.str.contains(r"[\n\rĊĉĈ].+", regex=True, na=False)
    )

    # Track message end
    df["im_end_cum"] = df.groupby(["prompt_ix", "msg_id"], sort=False)["is_im_end"].cumsum()

    # Header tokens are always structural:
    # - tokens after <|im_start|> up through header_end_pos
    # - except the header_end token if it's mixed (then it's treated as body)
    df["is_header"] = (
        (df["msg_id"] > 0)
        & (df["pos_in_msg"] > 0)
        & (
            (df["pos_in_msg"] < df["header_end_pos"])
            | ((df["pos_in_msg"] == df["header_end_pos"]) & ~df["header_end_token_mixed"])
        )
    )

    # Body tokens are:
    # - after the header newline, OR the mixed header_end token itself
    # - and before <|im_end|> (if present)
    df["in_body"] = (
        (df["msg_id"] > 0)
        & (df["im_end_cum"].eq(0))
        & (
            (df["pos_in_msg"] > df["header_end_pos"])
            | ((df["pos_in_msg"] == df["header_end_pos"]) & df["header_end_token_mixed"])
        )
    )

    # Reconstruct outer_role by joining all header tokens up to header_end_pos,
    # then stripping everything after the first newline glyph.
    header_mask = (
        (df["msg_id"] > 0)
        & (df["pos_in_msg"] > 0)
        & (df["pos_in_msg"] <= df["header_end_pos"])
    )
    header_joined = (
        df.loc[header_mask]
        .groupby(["prompt_ix", "msg_id"], sort=False)["token"]
        .agg("".join)
    )
    outer_role = (
        header_joined.astype(str)
        .str.replace(r"[\n\rĊĉĈ].*$", "", regex=True)
        .str.strip()
    )

    # This join reassigns df; groupby objects must be created AFTER this line.
    df = df.join(outer_role.rename("outer_role"), on=["prompt_ix", "msg_id"])
    df["outer_role"] = df["outer_role"].fillna("")

    is_system_msg = df["outer_role"].eq("system")
    is_user_msg = df["outer_role"].eq("user")
    is_assistant_msg = df["outer_role"].eq("assistant")
    is_tool_msg = df["outer_role"].eq("tool")          # uncommon for Qwen3, but supported
    is_developer_msg = df["outer_role"].eq("developer")  # uncommon for Qwen3, but supported

    msg_g = df.groupby(["prompt_ix", "msg_id"], sort=False)

    # -----------------------------
    # Inner wrappers (SCOPED)
    # -----------------------------
    # Think wrappers are meaningful only inside assistant messages
    df["is_think_open"] = df["in_body"] & is_assistant_msg & tok_stripped.eq(THINK_OPEN)
    df["is_think_close"] = df["in_body"] & is_assistant_msg & tok_stripped.eq(THINK_CLOSE)
    df["is_think_empty"] = df["in_body"] & is_assistant_msg & tok_stripped.eq(THINK_EMPTY)

    df["think_open_cum"] = msg_g["is_think_open"].cumsum()
    df["think_close_cum"] = msg_g["is_think_close"].cumsum()
    df["in_think"] = df["think_open_cum"] > df["think_close_cum"]

    # Tool call wrappers are meaningful only inside assistant messages
    df["is_tcall_open"] = df["in_body"] & is_assistant_msg & tok_stripped.eq(TCALL_OPEN)
    df["is_tcall_close"] = df["in_body"] & is_assistant_msg & tok_stripped.eq(TCALL_CLOSE)

    df["tcall_open_cum"] = msg_g["is_tcall_open"].cumsum()
    df["tcall_close_cum"] = msg_g["is_tcall_close"].cumsum()
    df["in_tool_call"] = df["tcall_open_cum"] > df["tcall_close_cum"]

    # Tool response wrappers are meaningful inside user messages (tool runs are wrapped as user)
    df["is_tresp_open"] = df["in_body"] & is_user_msg & tok_stripped.eq(TRESP_OPEN)
    df["is_tresp_close"] = df["in_body"] & is_user_msg & tok_stripped.eq(TRESP_CLOSE)

    df["tresp_open_cum"] = msg_g["is_tresp_open"].cumsum()
    df["tresp_close_cum"] = msg_g["is_tresp_close"].cumsum()
    df["in_tool_response"] = df["tresp_open_cum"] > df["tresp_close_cum"]

    # -----------------------------
    # Tag / structural token mask
    # -----------------------------
    prev_tok = df["prev_token"].astype(str)
    next_tok = df["next_token"].astype(str)

    prev_stripped = prev_tok.str.strip(STRIP_CHARS)
    next_stripped = next_tok.str.strip(STRIP_CHARS)

    # Control / envelope tokens
    df["is_control"] = (
        (df["msg_id"].eq(0))          # before first <|im_start|>
        | df["is_common_bos"]
        | df["is_im_start"]
        | df["is_im_end"]
    )

    # Wrapper tags (scoped): treat these tokens as structural
    df["is_wrapper_tag"] = (
        df["is_header"]
        | df["is_im_start"] | df["is_im_end"]
        | df["is_think_open"] | df["is_think_close"] | df["is_think_empty"]
        | df["is_tcall_open"] | df["is_tcall_close"]
        | df["is_tresp_open"] | df["is_tresp_close"]
    )

    # Structural newlines (pure newline tokens only)
    # - after <|im_end|>\n always template-attached
    # - around wrapper tags is typically template-attached (safe only for pure-newline tokens)
    WRAPPER_STRS = {THINK_OPEN, THINK_CLOSE, THINK_EMPTY, TCALL_OPEN, TCALL_CLOSE, TRESP_OPEN, TRESP_CLOSE}

    df["is_struct_nl"] = df["is_pure_nl"] & (
        prev_tok.eq(IM_END)
        | prev_stripped.isin(WRAPPER_STRS)
        | next_stripped.isin(WRAPPER_STRS)
    )

    df["is_tag"] = df["is_control"] | df["is_wrapper_tag"] | df["is_struct_nl"]

    # Potential content tokens are body tokens that aren't tagged away
    df["potential_content"] = df["in_body"] & ~df["is_tag"]

    # -----------------------------
    # Content-only role assignment
    # -----------------------------
    role = np.array([None] * len(df), dtype=object)

    # system / developer
    role[(df["potential_content"] & is_system_msg).to_numpy()] = "system"
    role[(df["potential_content"] & is_developer_msg).to_numpy()] = "developer"

    # tool outer messages (rare here)
    role[(df["potential_content"] & is_tool_msg).to_numpy()] = "tool"

    # user: tool_response overrides to tool
    user_content = df["potential_content"] & is_user_msg
    role[(user_content & df["in_tool_response"]).to_numpy()] = "tool"
    role[(user_content & ~df["in_tool_response"]).to_numpy()] = "user"

    # assistant: tool_call > cot > assistant
    asst_content = df["potential_content"] & is_assistant_msg
    role[(asst_content & df["in_tool_call"]).to_numpy()] = "tool_call"
    role[(asst_content & ~df["in_tool_call"] & df["in_think"]).to_numpy()] = "cot"
    role[(asst_content & ~df["in_tool_call"] & ~df["in_think"]).to_numpy()] = "assistant"

    df["role"] = role
    df["is_content"] = df["role"].notna()

    # -----------------------------
    # seg_ix + token_in_seg_ix (content-only; tags break runs)
    # -----------------------------
    is_labeled = df["is_content"]

    prev_labeled = is_labeled.groupby(df["prompt_ix"], sort=False).shift(1, fill_value=False)
    prev_role = df.groupby("prompt_ix", sort=False)["role"].shift(1)

    is_new_seg = is_labeled & (~prev_labeled | (df["role"] != prev_role))
    seg_counter = is_new_seg.groupby(df["prompt_ix"], sort=False).cumsum()  # 1,2,3,...

    df["seg_ix"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[is_labeled, "seg_ix"] = (seg_counter[is_labeled] - 1).astype("Int64")

    df["token_in_seg_ix"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[is_labeled, "token_in_seg_ix"] = (
        df.loc[is_labeled]
        .groupby(["prompt_ix", "seg_ix"], sort=False)
        .cumcount()
        .astype("Int64")
    )

    # -----------------------------
    # Cleanup
    # -----------------------------
    drop_cols = [
        "prev_token", "next_token",
        "is_pure_nl", "has_nl_char",
        "is_im_start", "is_im_end", "is_common_bos",
        "msg_id", "pos_in_msg", "header_end_pos", "header_end_token_mixed", "im_end_cum",
        "is_header", "in_body", "outer_role",
        "is_think_open", "is_think_close", "is_think_empty",
        "think_open_cum", "think_close_cum", "in_think",
        "is_tcall_open", "is_tcall_close", "tcall_open_cum", "tcall_close_cum", "in_tool_call",
        "is_tresp_open", "is_tresp_close", "tresp_open_cum", "tresp_close_cum", "in_tool_response",
        "is_control", "is_wrapper_tag", "is_struct_nl", "is_tag",
        "potential_content",
    ]

    out = df.drop(columns=drop_cols, errors="ignore")
    out = out.sort_values(["prompt_ix", "token_ix"]).reset_index(drop=True)
    return out


def label_apriel_content_roles(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Label Apriel-1.6-15B-Thinker tokens with content-only roles.

    Returns:
        Original df + {role, is_content, seg_ix, token_in_seg_ix}.

    Description:
        - Begin sentinels (<|begin_*|>) and <|end|> may be split across multiple tokens; we detect them
          by searching the concatenated token text and mapping matches back to token indices.
        - Inside assistant segments:
            * <tool_calls>...</tool_calls> content => tool_call (wrappers are structural).
            * [BEGIN FINAL RESPONSE] splits cot (before) vs assistant (after); if absent, assume cot.
            * The injected prefix "Here are my reasoning steps:\\n" (if present) is structural.
            * Newline-only tokens immediately before/after structural markers (reasoning header, final marker,
              tool/thinking wrappers, <|end|>) are structural (non-content), unless merged with other text.
        - <tool_calls> and [BEGIN FINAL RESPONSE] are treated as structural only in assistant segments
          (they can appear as literal text in the system prompt).
    """
    required = {"prompt_ix", "token_ix", "token"}
    missing = required - set(sample_df.columns)
    if missing:
        raise ValueError(f"sample_df missing required columns: {sorted(missing)}")

    # ----- Literals to locate in the concatenated token stream -----
    BEGIN_PATTERNS = {
        "system": "<|begin_system|>",
        "user": "<|begin_user|>",
        "assistant": "<|begin_assistant|>",
        "tool": "<|begin_tool_result|>",
        "content": "<|begin_content|>",
    }
    END_PATTERN = "<|end|>"

    TOOL_OPEN, TOOL_CLOSE = "<tool_calls>", "</tool_calls>"
    THINK_OPEN, THINK_CLOSE = "<thinking>", "</thinking>"
    FINAL_MARK = "[BEGIN FINAL RESPONSE]"

    REASONING_HEADER = "Here are my reasoning steps:\n"

    # Newline-only tokens (covers real '\n' and common tokenizer glyphs)
    NL_ONLY_RE = re.compile(r"^[\nĊĉĈ]+$")

    def _norm_nl(s: str) -> str:
        # Normalize common newline glyphs to '\n'
        return s.replace("Ċ", "\n").replace("ĉ", "\n").replace("Ĉ", "\n")

    def _find_token_spans(text: str, starts: np.ndarray, ends: np.ndarray, literal: str):
        """
        Return list of (tok_start, tok_end, char_start, char_end) spans for a literal substring.
        Works even if the literal is split across many tokens.
        """
        spans = []
        for m in re.finditer(re.escape(literal), text):
            cs, ce = m.start(), m.end()
            # token start = first token whose end > cs
            tok_start = int(np.searchsorted(ends, cs, side="right"))
            # token end = last token whose start < ce
            tok_end = int(np.searchsorted(starts, ce, side="left") - 1)
            if 0 <= tok_start < len(starts) and tok_start <= tok_end < len(starts):
                spans.append((tok_start, tok_end, cs, ce))
        return spans

    def _process_prompt(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("token_ix").reset_index(drop=True)
        toks = g["token"].astype(str).tolist()
        n = len(toks)

        # Character offsets for mapping substring matches back to token indices
        lens = np.fromiter((len(t) for t in toks), dtype=int, count=n)
        starts = np.zeros(n, dtype=int)
        if n > 1:
            starts[1:] = np.cumsum(lens)[:-1]
        ends = starts + lens
        text = "".join(toks)

        # --- Find begin spans and end spans ---
        begin_spans = []
        for kind, pat in BEGIN_PATTERNS.items():
            for tok_s, tok_e, cs, _ in _find_token_spans(text, starts, ends, pat):
                begin_spans.append((tok_s, tok_e, kind, cs))
        begin_spans.sort(key=lambda x: (x[0], x[3], -(x[1] - x[0])))

        end_spans = _find_token_spans(text, starts, ends, END_PATTERN)

        # Build begin events (start -> (end, kind)) and begin-token mask
        begin_events = {}
        is_begin_tok = np.zeros(n, dtype=bool)
        for tok_s, tok_e, kind, _ in begin_spans:
            if tok_s in begin_events:
                continue
            begin_events[tok_s] = (tok_e, kind)
            is_begin_tok[tok_s:tok_e + 1] = True

        # Build end events (start -> end) and end-token mask
        end_events = {}
        is_end_tok = np.zeros(n, dtype=bool)
        for tok_s, tok_e, _, _ in end_spans:
            if tok_s in end_events:
                continue
            end_events[tok_s] = tok_e
            is_end_tok[tok_s:tok_e + 1] = True

        # Wrapper/marker spans (may also be split across tokens)
        tool_open_spans = _find_token_spans(text, starts, ends, TOOL_OPEN)
        tool_close_spans = _find_token_spans(text, starts, ends, TOOL_CLOSE)
        think_open_spans = _find_token_spans(text, starts, ends, THINK_OPEN)
        think_close_spans = _find_token_spans(text, starts, ends, THINK_CLOSE)
        final_spans = _find_token_spans(text, starts, ends, FINAL_MARK)

        # Overlap masks (token is part of wrapper/marker span)
        tool_open_tok = np.zeros(n, dtype=bool)
        tool_close_tok = np.zeros(n, dtype=bool)
        think_open_tok = np.zeros(n, dtype=bool)
        think_close_tok = np.zeros(n, dtype=bool)
        final_tok = np.zeros(n, dtype=bool)

        for s, e, _, _ in tool_open_spans:
            tool_open_tok[s:e + 1] = True
        for s, e, _, _ in tool_close_spans:
            tool_close_tok[s:e + 1] = True
        for s, e, _, _ in think_open_spans:
            think_open_tok[s:e + 1] = True
        for s, e, _, _ in think_close_spans:
            think_close_tok[s:e + 1] = True
        for s, e, _, _ in final_spans:
            final_tok[s:e + 1] = True

        # --- Scan to assign container kind + header tokens ---
        container = np.array([None] * n, dtype=object)  # system/user/assistant/tool/content/None
        msg_id = np.zeros(n, dtype=int)
        is_header_tok = np.zeros(n, dtype=bool)

        current_kind = None
        current_msg = 0

        header_phase = "done"  # 'in_begin', 'header', 'done'
        begin_end_idx = None
        pending_close_at = None

        for i in range(n):
            if i in begin_events:
                begin_end_idx, current_kind = begin_events[i]
                current_msg += 1
                header_phase = "in_begin"
                pending_close_at = None

            msg_id[i] = current_msg
            container[i] = current_kind

            if current_kind is not None:
                tok_norm = _norm_nl(toks[i])

                # Header = after begin tag span ends, up to first token containing '\n'
                if header_phase == "in_begin":
                    if begin_end_idx is not None and i == begin_end_idx:
                        if "\n" in tok_norm:
                            header_phase = "done"
                        else:
                            header_phase = "header"
                elif header_phase == "header":
                    is_header_tok[i] = True
                    if "\n" in tok_norm:
                        header_phase = "done"

                # Close assistant at <|end|> (after its span ends)
                if current_kind == "assistant" and i in end_events:
                    pending_close_at = end_events[i]
                if pending_close_at is not None and i == pending_close_at:
                    current_kind = None
                    header_phase = "done"
                    begin_end_idx = None
                    pending_close_at = None
            else:
                header_phase = "done"
                begin_end_idx = None
                pending_close_at = None

        in_segment = container != None
        in_body = in_segment & ~is_begin_tok & ~is_end_tok & ~is_header_tok

        # --- Newline-only detection ---
        is_nl_only = np.fromiter((NL_ONLY_RE.fullmatch(t) is not None for t in toks), dtype=bool, count=n)

        # next / prev non-newline-only token indices (within prompt)
        next_non_nl = np.full(n, -1, dtype=int)
        nxt = -1
        for i in range(n - 1, -1, -1):
            if not is_nl_only[i]:
                nxt = i
            next_non_nl[i] = nxt

        prev_non_nl = np.full(n, -1, dtype=int)
        prv = -1
        for i in range(n):
            if not is_nl_only[i]:
                prv = i
            prev_non_nl[i] = prv

        # --- Boundary newline runs: newline-only tokens immediately before begin/end token-spans ---
        is_boundary_nl = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_nl_only[i]:
                j = next_non_nl[i]
                if j != -1 and (is_begin_tok[j] or is_end_tok[j]):
                    is_boundary_nl[i] = True

        # Assistant-only structural handling (B)
        asst = (container == "assistant")

        is_tool_open = tool_open_tok & asst
        is_tool_close = tool_close_tok & asst
        is_think_open = think_open_tok & asst
        is_think_close = think_close_tok & asst
        is_final_mark = final_tok & asst

        # --- Compute per-assistant-message: in_tool_calls, in_thinking, after_final, seg_has_final ---
        in_tool_calls = np.zeros(n, dtype=bool)
        in_thinking = np.zeros(n, dtype=bool)
        after_final = np.zeros(n, dtype=bool)
        seg_has_final = np.zeros(n, dtype=bool)

        def _spans_in_msg(spans, msg):
            out = []
            for s, e, _, _ in spans:
                if msg_id[s] == msg and asst[s]:
                    out.append((s, e))
            return out

        asst_msgs = sorted(set(msg_id[asst]))
        for m in asst_msgs:
            idxs = np.where((msg_id == m) & asst)[0]
            if len(idxs) == 0:
                continue
            seg_start, seg_end = int(idxs[0]), int(idxs[-1])

            # tool_calls depth: open activates AFTER open-span end; close deactivates AT close-span start
            open_sp = _spans_in_msg(tool_open_spans, m)
            close_sp = _spans_in_msg(tool_close_spans, m)

            open_after = {}
            for s, e in open_sp:
                open_after[e] = open_after.get(e, 0) + 1
            close_at = {}
            for s, e in close_sp:
                close_at[s] = close_at.get(s, 0) + 1

            depth = 0
            for i in range(seg_start, seg_end + 1):
                if i in close_at:
                    depth = max(depth - close_at[i], 0)
                if depth > 0:
                    in_tool_calls[i] = True
                if i in open_after:
                    depth += open_after[i]

            # thinking depth
            open_sp = _spans_in_msg(think_open_spans, m)
            close_sp = _spans_in_msg(think_close_spans, m)

            open_after = {}
            for s, e in open_sp:
                open_after[e] = open_after.get(e, 0) + 1
            close_at = {}
            for s, e in close_sp:
                close_at[s] = close_at.get(s, 0) + 1

            depth = 0
            for i in range(seg_start, seg_end + 1):
                if i in close_at:
                    depth = max(depth - close_at[i], 0)
                if depth > 0:
                    in_thinking[i] = True
                if i in open_after:
                    depth += open_after[i]

            # final marker: if present, after_final starts AFTER the marker span ends
            finals = _spans_in_msg(final_spans, m)
            if finals:
                s0, e0 = min(finals, key=lambda x: x[0])
                seg_has_final[idxs] = True
                if e0 + 1 <= seg_end:
                    after_final[e0 + 1:seg_end + 1] = True

        # --- Reasoning header prefix (structural) inside assistant segments ---
        is_reason_header = np.zeros(n, dtype=bool)
        for m in asst_msgs:
            idxs = np.where((msg_id == m) & asst & in_body)[0]
            if len(idxs) == 0:
                continue

            toks_norm = [_norm_nl(toks[i]) for i in idxs]
            body_text = "".join(toks_norm)
            if not body_text.startswith(REASONING_HEADER):
                continue

            prefix_len = len(REASONING_HEADER)
            lens_local = np.fromiter((len(t) for t in toks_norm), dtype=int, count=len(toks_norm))
            starts_local = np.zeros(len(toks_norm), dtype=int)
            if len(toks_norm) > 1:
                starts_local[1:] = np.cumsum(lens_local)[:-1]

            # Mark any token whose start is within the prefix window
            for flag, ti in zip(starts_local < prefix_len, idxs):
                if flag:
                    is_reason_header[ti] = True
                else:
                    break

        # ---------------------------------------------------------------------
        # assistant-only structural whitespace adjacency
        # - newline-only tokens immediately BEFORE or AFTER:
        #     * reasoning header prefix
        #     * final marker
        #     * tool/thinking wrappers
        #     * <|end|>
        #   are structural (non-content).
        # ---------------------------------------------------------------------
        is_struct_anchor_asst = (
            is_final_mark | is_tool_open | is_tool_close | is_think_open | is_think_close | is_end_tok | is_reason_header
        )

        for i in range(n):
            if not (asst[i] and is_nl_only[i]):
                continue

            # Before-anchor: next non-nl token is an anchor
            j = next_non_nl[i]
            if j != -1 and is_struct_anchor_asst[j]:
                is_boundary_nl[i] = True
                continue

            # After-anchor: previous non-nl token is an anchor
            k = prev_non_nl[i]
            if k != -1 and is_struct_anchor_asst[k]:
                is_boundary_nl[i] = True
                continue

        # --- Tag mask and content mask ---
        # Note: tool/final/think wrappers are structural only in assistant segments.
        is_tag = (
            is_begin_tok
            | is_end_tok
            | is_header_tok
            | is_boundary_nl
            | (asst & (is_tool_open | is_tool_close | is_think_open | is_think_close | is_final_mark | is_reason_header))
        )

        is_content = in_body & ~is_tag

        # --- Role assignment (content-only) ---
        role = np.array([None] * n, dtype=object)

        role[((container == "system") | (container == "content")) & is_content] = "system"
        role[(container == "user") & is_content] = "user"
        role[(container == "tool") & is_content] = "tool"

        asst_content = asst & is_content
        role[asst_content & in_tool_calls] = "tool_call"

        remaining = asst_content & ~in_tool_calls
        role[remaining & in_thinking] = "cot"

        non_think = remaining & ~in_thinking
        # marker present => split; else all cot
        role[non_think & seg_has_final & ~after_final] = "cot"
        role[non_think & seg_has_final & after_final] = "assistant"
        role[non_think & ~seg_has_final] = "cot"

        # --- seg_ix + token_in_seg_ix (only for role!=None) ---
        is_labeled = role != None
        seg_ix = np.full(n, pd.NA, dtype=object)
        token_in_seg_ix = np.full(n, pd.NA, dtype=object)

        seg_counter = -1
        tok_counter = 0
        prev_role = None

        for i in range(n):
            if not is_labeled[i]:
                prev_role = None
                continue
            if prev_role != role[i]:
                seg_counter += 1
                tok_counter = 0
                prev_role = role[i]
            seg_ix[i] = seg_counter
            token_in_seg_ix[i] = tok_counter
            tok_counter += 1

        g["is_content"] = is_content.astype(bool)
        g["role"] = role
        g["seg_ix"] = pd.Series(seg_ix, dtype="Int64")
        g["token_in_seg_ix"] = pd.Series(token_in_seg_ix, dtype="Int64")
        return g

    out = (
        sample_df
        .sort_values(["prompt_ix", "token_ix"])
        .groupby("prompt_ix", sort=False, group_keys=False)
        .apply(_process_prompt)
        .sort_values(["prompt_ix", "token_ix"])
        .reset_index(drop=True)
    )
    return out

def label_olmo3_content_roles(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Label OLMo3-7B (Think + Instruct) token streams with content-only roles.

    Description:
        - ChatML framing: <|im_start|>role\\n ... (<|im_end|> or <|endoftext|>).
        - <functions>...</functions> and <function_calls>...</function_calls> are single tokens.
        - <think> / </think> are NOT single tokens; we detect them by substring scanning over the
            concatenated assistant segment text. A dangling "<think>" (no close) marks cot until
            the segment ends (supports olmo3-7b-think generation prompt).
        - Pure newline tokens immediately before <functions> in user/system segments and pure newline
            tokens immediately after </think> (until first non-newline token) are treated as structural
            (non-content). Newlines inside think content are preserved as cot content.

    Returns:
        - role: one of {system, user, assistant, cot, tool_call, tool} or None
        - is_content: bool
        - seg_ix: int or None
        - token_in_seg_ix: int or None
    """
    required = {"prompt_ix", "token_ix", "token"}
    missing = required - set(sample_df.columns)
    if missing:
        raise ValueError(f"sample_df missing required columns: {sorted(missing)}")

    # ---- Core ChatML tokens ----
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    EOS = "<|endoftext|>"

    # ---- Single-token wrappers (confirmed) ----
    OPEN_FUNCS, CLOSE_FUNCS = "<functions>", "</functions>"
    OPEN_FCALLS, CLOSE_FCALLS = "<function_calls>", "</function_calls>"

    # Think tags are NOT single tokens; we will scan for these substrings.
    THINK_OPEN_STR = "<think>"
    THINK_CLOSE_STR = "</think>"

    # Newline-only token detector (covers common newline glyphs too)
    NL_ONLY_RE = re.compile(r"^[\nĊĉĈ]+$")

    # For header parsing, normalize common newline glyphs to '\n'
    def _norm_nl_series(s: pd.Series) -> pd.Series:
        return (
            s.astype(str)
             .str.replace("Ċ", "\n", regex=False)
             .str.replace("ĉ", "\n", regex=False)
             .str.replace("Ĉ", "\n", regex=False)
        )

    df = (
        sample_df
        .sort_values(["prompt_ix", "token_ix"])
        .reset_index(drop=True)
        .copy()
    )

    token_norm = _norm_nl_series(df["token"])

    # ---- Segment by <|im_start|> ----
    df["seg_id"] = df.groupby("prompt_ix", sort=False)["token"].transform(
        lambda s: (s == IM_START).cumsum()
    )

    df["is_start"] = df["token"].eq(IM_START)
    df["is_end"] = df["token"].eq(IM_END)
    df["is_eos"] = df["token"].eq(EOS)

    # newline flags
    df["has_nl"] = token_norm.str.contains(r"[\n]+", regex=True) | df["token"].astype(str).str.contains(r"[ĊĉĈ]+", regex=True)
    df["is_pure_nl"] = df["token"].astype(str).str.fullmatch(NL_ONLY_RE, na=False)

    by_seg = df.groupby(["prompt_ix", "seg_id"], sort=False)

    # ---- Header/body split (tokens between <|im_start|> and first newline char are structural) ----
    df["nl_cum"] = by_seg["has_nl"].cumsum()
    df["is_first_nl"] = df["has_nl"] & df["nl_cum"].eq(1)
    df["before_header"] = df["nl_cum"].eq(0)

    df["is_header_token"] = (
        (df["seg_id"] > 0)
        & ~df["is_start"]
        & (df["before_header"] | df["is_first_nl"])
    )

    # ---- Close sentinel: <|im_end|> OR EOS ----
    df["is_close"] = df["is_end"] | df["is_eos"]
    df["before_end"] = by_seg["is_close"].cumsum().eq(0)

    # Body tokens = inside a message, after header, before close sentinel
    df["in_body"] = (df["seg_id"] > 0) & ~df["is_header_token"] & df["before_end"]

    # ---- Reconstruct header role string per segment ----
    # Grab tokens up to the first newline (split token that contains newline if needed).
    df["header_piece"] = np.select(
        [
            (df["seg_id"] > 0) & ~df["is_start"] & df["before_header"] & ~df["has_nl"],
            (df["seg_id"] > 0) & ~df["is_start"] & df["is_first_nl"],
        ],
        [
            df["token"].astype(str),
            token_norm.str.split("\n", n=1, regex=False).str[0],
        ],
        default=None,
    )

    # Concatenate header pieces per segment
    header_line = (
        df["header_piece"]
        .fillna("")
        .groupby([df["prompt_ix"], df["seg_id"]], sort=False)
        .transform("sum")
        .fillna("")
        .str.strip()
        .str.lower()
    )
    df = df.drop(columns=["header_piece"])
    df["header_line"] = header_line

    # Coarse segment kind
    df["seg_kind"] = np.select(
        [
            df["header_line"].str.startswith("assistant"),
            df["header_line"].str.startswith("user"),
            df["header_line"].str.startswith("system"),
            df["header_line"].str.startswith("environment"),
            df["header_line"].str.startswith("tool"),
        ],
        [
            "assistant",
            "user",
            "system",
            "environment",  # environment is tool output in these templates
            "environment",  # tool alias -> environment
        ],
        default=None,
    )

    is_assistant_seg = df["seg_kind"].eq("assistant")
    is_user_seg = df["seg_kind"].eq("user")
    is_system_seg = df["seg_kind"].eq("system")
    is_env_seg = df["seg_kind"].eq("environment")

    # ---- Single-token wrapper tags (scoped; avoid treating literal mentions as structure) ----
    # <functions> wrappers are structural in system/user segments only
    df["is_funcs_open"] = (is_system_seg | is_user_seg) & df["token"].eq(OPEN_FUNCS)
    df["is_funcs_close"] = (is_system_seg | is_user_seg) & df["token"].eq(CLOSE_FUNCS)

    # <function_calls> wrappers are structural in assistant segments only
    df["is_fcalls_open"] = is_assistant_seg & df["token"].eq(OPEN_FCALLS)
    df["is_fcalls_close"] = is_assistant_seg & df["token"].eq(CLOSE_FCALLS)

    # in_function_calls (assistant-only)
    df["fcalls_open_cum"]  = df.groupby(["prompt_ix", "seg_id"], sort=False)["is_fcalls_open"].cumsum()
    df["fcalls_close_cum"] = df.groupby(["prompt_ix", "seg_id"], sort=False)["is_fcalls_close"].cumsum()
    df["in_function_calls"] = df["fcalls_open_cum"] > df["fcalls_close_cum"]

    # ---- Structural newline before <functions> (template-inserted in user when functions appended) ----
    df["next_token"] = df.groupby("prompt_ix", sort=False)["token"].shift(-1)
    df["ws_before_functions"] = df["is_pure_nl"] & df["next_token"].eq(OPEN_FUNCS) & (is_user_seg | is_system_seg)

    # ---- Base tag mask (always structural) ----
    df["is_tag0"] = (
        df["is_start"] | df["is_end"] | df["is_eos"]
        | df["is_funcs_open"] | df["is_funcs_close"]
        | df["is_fcalls_open"] | df["is_fcalls_close"]
        | df["ws_before_functions"]
    )

    # ---- Think region detection (assistant-only; <think> is NOT single-token) ----
    def _mark_think_region(group: pd.DataFrame) -> pd.DataFrame:
        n = len(group)
        group["in_think"] = False
        group["is_think_tag"] = False
        group["ws_after_think_local"] = False

        if n == 0:
            return group
        if group["seg_kind"].iloc[0] != "assistant":
            return group

        tokens = group["token"].astype(str).tolist()
        text = "".join(tokens)

        # char offsets -> token indices
        lengths = np.fromiter((len(t) for t in tokens), dtype=int, count=n)
        starts = np.zeros(n, dtype=int)
        if n > 1:
            starts[1:] = np.cumsum(lengths)[:-1]
        ends = starts + lengths

        opens = list(re.finditer(re.escape(THINK_OPEN_STR), text))
        closes = list(re.finditer(re.escape(THINK_CLOSE_STR), text))

        in_think = np.zeros(n, dtype=bool)
        is_think_tag = np.zeros(n, dtype=bool)
        ws_after = np.zeros(n, dtype=bool)

        close_i = 0
        for om in opens:
            o_start, o_end = om.span()

            # Find the next close that starts after the open tag ends
            while close_i < len(closes) and closes[close_i].start() < o_end:
                close_i += 1

            if close_i < len(closes):
                cm = closes[close_i]
                close_i += 1
                c_start, c_end = cm.span()

                # Tag tokens = anything overlapping the open/close tag substrings
                open_tag = (starts < o_end) & (ends > o_start)
                close_tag = (starts < c_end) & (ends > c_start)
                is_think_tag |= open_tag | close_tag

                # Think content = tokens overlapping (o_end, c_start), excluding tag tokens
                content_mask = (ends > o_end) & (starts < c_start)
                content_mask &= ~is_think_tag
                in_think |= content_mask

                # Structural whitespace immediately after </think> (pure newlines until first non-newline non-tag)
                after_idxs = np.where(starts >= c_end)[0]
                for idx in after_idxs:
                    if group["is_tag0"].iloc[idx] or is_think_tag[idx]:
                        continue
                    if group["is_pure_nl"].iloc[idx]:
                        ws_after[idx] = True
                        continue
                    break
            else:
                # Dangling open: think continues to end of segment
                open_tag = (starts < o_end) & (ends > o_start)
                is_think_tag |= open_tag

                # Think content = all tokens after open tag end, excluding tag tokens
                content_mask = ends > o_end
                content_mask &= ~is_think_tag
                in_think |= content_mask

        group["in_think"] = in_think
        group["is_think_tag"] = is_think_tag
        group["ws_after_think_local"] = ws_after
        return group

    df = (
        df.groupby(["prompt_ix", "seg_id"], sort=False, group_keys=False)
          .apply(_mark_think_region)
    )

    # ---- Final tag mask ----
    df["is_tag"] = df["is_tag0"] | df["is_think_tag"] | df["ws_after_think_local"]

    # Content span = body tokens minus tags
    df["is_content"] = df["in_body"] & ~df["is_tag"]

    # ---- Role assignment (content-only) ----
    role = np.array([None] * len(df), dtype=object)

    # system/user/environment
    role[(is_system_seg & df["is_content"]).to_numpy()] = "system"
    role[(is_user_seg & df["is_content"]).to_numpy()] = "user"
    role[(is_env_seg & df["is_content"]).to_numpy()] = "tool"

    # assistant with precedence: tool_call > cot > assistant
    asst_content = is_assistant_seg & df["is_content"]
    role[(asst_content & df["in_function_calls"]).to_numpy()] = "tool_call"
    role[(asst_content & ~df["in_function_calls"] & df["in_think"]).to_numpy()] = "cot"
    role[(asst_content & ~df["in_function_calls"] & ~df["in_think"]).to_numpy()] = "assistant"

    df["role"] = role

    # ---- seg_ix + token_in_seg_ix (only for labeled tokens; None otherwise) ----
    is_labeled = df["role"].notna()

    prev_labeled = is_labeled.groupby(df["prompt_ix"], sort=False).shift(1, fill_value=False)
    prev_role = df.groupby("prompt_ix", sort=False)["role"].shift(1)

    is_new_seg = is_labeled & (~prev_labeled | (df["role"] != prev_role))
    seg_counter = is_new_seg.groupby(df["prompt_ix"], sort=False).cumsum()  # 1,2,3,...

    df["seg_ix"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[is_labeled, "seg_ix"] = (seg_counter[is_labeled] - 1).astype("Int64")

    df["token_in_seg_ix"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[is_labeled, "token_in_seg_ix"] = (
        df.loc[is_labeled].groupby(["prompt_ix", "seg_ix"], sort=False).cumcount().astype("Int64")
    )

    # ---- Cleanup helper columns ----
    drop_cols = [
        "seg_id",
        "is_start", "is_end", "is_eos",
        "has_nl", "is_pure_nl",
        "nl_cum", "is_first_nl", "before_header", "is_header_token",
        "is_close", "before_end", "in_body",
        "header_line", "seg_kind",
        "is_funcs_open", "is_funcs_close",
        "is_fcalls_open", "is_fcalls_close",
        "fcalls_open_cum", "fcalls_close_cum",
        "in_function_calls",
        "next_token", "ws_before_functions",
        "is_tag0",
        "in_think", "is_think_tag", "ws_after_think_local",
        "is_tag",
    ]
    out = df.drop(columns=drop_cols, errors="ignore")

    out = out.sort_values(["prompt_ix", "token_ix"]).reset_index(drop=True)
    return out

def label_glm4_content_roles(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Labels GLM4 token streams with content-only roles.

    Description:
        - Role containers start at <|system|>, <|user|>, <|assistant|>, <|observation|> (no explicit end tag).
        - Assistant-only wrappers: <think>...</think> => cot; <tool_call>...</tool_call> => tool_call.
        - Observation tool output is wrapped in <tool_response>...</tool_response> (wrappers are structural).
        - Tool-call example text inside the system tools preamble is treated as system content
          (tool wrappers are only active in assistant segments).
        - Pure newline tokens that are template-paired with tags are structural (non-content).
    
    Returns:
        - role: one of {system, user, assistant, cot, tool_call, tool} or None
        - is_content: bool
        - seg_ix: int or None
        - token_in_seg_ix: int or None
    """
    required = {"prompt_ix", "token_ix", "token"}
    missing = required - set(sample_df.columns)
    if missing:
        raise ValueError(f"sample_df missing required columns: {sorted(missing)}")

    # ---- Sentinels / wrappers (assumed single-token) ----
    SYSTEM    = "<|system|>"
    USER = "<|user|>"
    ASSISTANT = "<|assistant|>"
    OBS       = "<|observation|>"

    PREFIX_TOKENS = {"[gMASK]", "<sop>", "<eop>"}

    THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

    TCALL_OPEN, TCALL_CLOSE = "<tool_call>", "</tool_call>"
    TRESP_OPEN, TRESP_CLOSE = "<tool_response>", "</tool_response>"

    ARGK_OPEN, ARGK_CLOSE = "<arg_key>", "</arg_key>"
    ARGV_OPEN, ARGV_CLOSE = "<arg_value>", "</arg_value>"

    NOTHINK = "/nothink"

    ROLE_SENTINELS = {SYSTEM, USER, ASSISTANT, OBS}

    # Newline-only token detector (covers common newline glyphs)
    NL_ONLY_RE = re.compile(r"^[\nĊĉĈ]+$")

    df = (
        sample_df
        .sort_values(["prompt_ix", "token_ix"])
        .reset_index(drop=True)
        .copy()
    )

    # ---- Segment by role sentinel ----
    df["is_seg_start"] = df["token"].isin(ROLE_SENTINELS)
    df["seg_id"] = df.groupby("prompt_ix", sort=False)["is_seg_start"].cumsum()

    df["seg_role_token"] = (
        df["token"].where(df["is_seg_start"])
          .groupby([df["prompt_ix"], df["seg_id"]], sort=False)
          .transform("first")
          .fillna("")
    )

    df["seg_kind"] = np.select(
        [df["seg_role_token"].eq(SYSTEM), df["seg_role_token"].eq(USER), df["seg_role_token"].eq(ASSISTANT), df["seg_role_token"].eq(OBS)],
        ["system", "user", "assistant", "observation"],
        default = None,
    )

    in_segment = df["seg_id"] > 0
    is_assistant_seg = df["seg_kind"].eq("assistant")
    is_observation_seg = df["seg_kind"].eq("observation")

    # ---- Pure newline tokens ----
    df["is_pure_nl"] = df["token"].astype(str).str.fullmatch(NL_ONLY_RE, na=False)

    # prev/next token (within prompt)
    df["prev_token"] = df.groupby("prompt_ix", sort=False)["token"].shift(1)
    df["next_token"] = df.groupby("prompt_ix", sort=False)["token"].shift(-1)

    # ---- Wrapper tags (scoped so system examples remain content) ----
    # Assistant-only wrappers
    df["is_think_open"]  = is_assistant_seg & df["token"].eq(THINK_OPEN)
    df["is_think_close"] = is_assistant_seg & df["token"].eq(THINK_CLOSE)

    df["is_tcall_open"]  = is_assistant_seg & df["token"].eq(TCALL_OPEN)
    df["is_tcall_close"] = is_assistant_seg & df["token"].eq(TCALL_CLOSE)

    # Observation-only wrappers
    df["is_tresp_open"]  = is_observation_seg & df["token"].eq(TRESP_OPEN)
    df["is_tresp_close"] = is_observation_seg & df["token"].eq(TRESP_CLOSE)

    # We'll treat arg tags as structural only *inside* tool_call blocks (assistant segments)
    # (this avoids mislabeling if the assistant outputs these strings outside a tool call)
    by_seg = df.groupby(["prompt_ix", "seg_id"], sort=False)

    df["tcall_open_cum"]  = by_seg["is_tcall_open"].cumsum()
    df["tcall_close_cum"] = by_seg["is_tcall_close"].cumsum()
    df["in_tool_call"] = df["tcall_open_cum"] > df["tcall_close_cum"]

    df["think_open_cum"]  = by_seg["is_think_open"].cumsum()
    df["think_close_cum"] = by_seg["is_think_close"].cumsum()
    df["in_think"] = df["think_open_cum"] > df["think_close_cum"]

    # tool_response membership (observation)
    df["tresp_open_cum"]  = by_seg["is_tresp_open"].cumsum()
    df["tresp_close_cum"] = by_seg["is_tresp_close"].cumsum()
    df["in_tool_response"] = df["tresp_open_cum"] > df["tresp_close_cum"]

    df["is_argk_open"]  = is_assistant_seg & df["in_tool_call"] & df["token"].eq(ARGK_OPEN)
    df["is_argk_close"] = is_assistant_seg & df["in_tool_call"] & df["token"].eq(ARGK_CLOSE)
    df["is_argv_open"]  = is_assistant_seg & df["in_tool_call"] & df["token"].eq(ARGV_OPEN)
    df["is_argv_close"] = is_assistant_seg & df["in_tool_call"] & df["token"].eq(ARGV_CLOSE)

    # ---- Base tag mask (wrappers/sentinels/control tokens) ----
    df["is_prefix"] = (df["seg_id"].eq(0)) & df["token"].isin(PREFIX_TOKENS)
    df["is_role_sentinel"] = df["token"].isin(ROLE_SENTINELS)
    df["is_nothink"] = df["token"].eq(NOTHINK)

    df["is_tag0"] = (
        df["is_prefix"]
        | df["is_role_sentinel"]
        | df["is_nothink"]
        | df["is_think_open"] | df["is_think_close"]
        | df["is_tcall_open"] | df["is_tcall_close"]
        | df["is_argk_open"]  | df["is_argk_close"]
        | df["is_argv_open"]  | df["is_argv_close"]
        | df["is_tresp_open"] | df["is_tresp_close"]
    )

    # ---- Structural newline rules (pure newline tokens only; paired-with-tag cases) ----
    # We intentionally do NOT treat newline before /nothink as structural.
    struct_nl = df["is_pure_nl"] & (
        # Newline immediately after a role sentinel is template-inserted
        df["prev_token"].isin([SYSTEM, USER, ASSISTANT, OBS])

        # Assistant: newline immediately before <think> or <tool_call> is template-inserted
        | (is_assistant_seg & df["next_token"].isin([THINK_OPEN, TCALL_OPEN]))

        # Assistant: newline immediately after </think> (before visible answer) is template-inserted
        | (is_assistant_seg & df["prev_token"].eq(THINK_CLOSE))

        # Tool-call formatting: newline tokens around arg tags are template-inserted
        | (is_assistant_seg & df["in_tool_call"] & (
            df["next_token"].isin([ARGK_OPEN, TCALL_CLOSE])          # e.g., after tool name or before </tool_call>
            | df["prev_token"].isin([ARGK_CLOSE, ARGV_CLOSE])        # between </arg_key> and <arg_value>, between args
        ))

        # Observation: tool_response wrapper newlines are template-inserted
        | (is_observation_seg & (
            df["prev_token"].isin([OBS, TRESP_OPEN, TRESP_CLOSE])    # after observation sentinel/open/close
            | df["next_token"].isin([TRESP_OPEN, TRESP_CLOSE])       # before open/close
        ))
    )

    df["is_tag"] = df["is_tag0"] | struct_nl

    # ---- Content tokens ----
    df["is_content"] = in_segment & ~df["is_tag"]

    # ---- Role assignment (content-only) ----
    role = np.array([None] * len(df), dtype=object)

    # system / user
    role[(df["seg_kind"].eq("system") & df["is_content"]).to_numpy()] = "system"
    role[(df["seg_kind"].eq("user") & df["is_content"]).to_numpy()] = "user"

    # assistant: tool_call > cot > assistant
    asst_content = is_assistant_seg & df["is_content"]
    role[(asst_content & df["in_tool_call"]).to_numpy()] = "tool_call"
    role[(asst_content & ~df["in_tool_call"] & df["in_think"]).to_numpy()] = "cot"
    role[(asst_content & ~df["in_tool_call"] & ~df["in_think"]).to_numpy()] = "assistant"

    # observation: tool (we assume tool output lives in <tool_response>, but after stripping structural whitespace,
    # this typically equals “all remaining observation content”)
    obs_content = is_observation_seg & df["is_content"]
    role[obs_content.to_numpy()] = "tool"

    df["role"] = role

    # ---- seg_ix + token_in_seg_ix (only for labeled tokens) ----
    is_labeled = df["role"].notna()

    prev_labeled = is_labeled.groupby(df["prompt_ix"], sort=False).shift(1, fill_value=False)
    prev_role = df.groupby("prompt_ix", sort=False)["role"].shift(1)

    is_new_seg = is_labeled & (~prev_labeled | (df["role"] != prev_role))
    seg_counter = is_new_seg.groupby(df["prompt_ix"], sort=False).cumsum()  # 1,2,3,...

    df["seg_ix"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[is_labeled, "seg_ix"] = (seg_counter[is_labeled] - 1).astype("Int64")

    df["token_in_seg_ix"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[is_labeled, "token_in_seg_ix"] = (
        df.loc[is_labeled].groupby(["prompt_ix", "seg_ix"], sort=False).cumcount().astype("Int64")
    )

    # ---- Final cleanup ----
    drop_cols = [
        "is_seg_start", "seg_id", "seg_role_token", "seg_kind",
        "is_pure_nl", "prev_token", "next_token",
        "is_think_open", "is_think_close",
        "is_tcall_open", "is_tcall_close",
        "is_tresp_open", "is_tresp_close",
        "is_argk_open", "is_argk_close",
        "is_argv_open", "is_argv_close",
        "tcall_open_cum", "tcall_close_cum", "think_open_cum", "think_close_cum",
        "tresp_open_cum", "tresp_close_cum",
        "in_tool_call", "in_think", "in_tool_response",
        "is_prefix", "is_role_sentinel", "is_nothink",
        "is_tag0", "is_tag",
    ]
    out = df.drop(columns=drop_cols, errors="ignore")
    out = out.sort_values(["prompt_ix", "token_ix"]).reset_index(drop=True)
    return out


def label_nemotron3_content_roles(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Labels NVIDIA Nemotron 3 Nano 30B-A3B (ChatML) token streams with *content-only* roles.

    Key template structure (HF chat_template.jinja):
      - Outer messages: <|im_start|>ROLE\\n ... <|im_end|>\\n
      - Assistant messages contain <think>...</think> (possibly empty).
      - Tool calls (assistant): <tool_call> ... </tool_call>
      - Tool outputs live inside a ChatML user wrapper as <tool_response> ... </tool_response>

    Semantics:
      - role is assigned ONLY to semantic content tokens.
      - All protocol / wrapper / structural tokens are role=None and is_content=False.
      - No assumptions about turn ordering.

    Returns original columns +:
      - role: {system, developer, user, assistant, cot, tool_call, tool} or None
      - is_content: bool
      - seg_ix: contiguous run index of content tokens with same role (tags break runs)
      - token_in_seg_ix: 0-based within seg_ix (NA for non-content)
    """
    required = {"prompt_ix", "token_ix", "token"}
    missing = required - set(sample_df.columns)
    if missing:
        raise ValueError(f"sample_df missing required columns: {sorted(missing)}")

    df = (
        sample_df.sort_values(["prompt_ix", "token_ix"])
        .reset_index(drop=True)
        .copy()
    )

    # -----------------------------
    # Sentinels / wrappers
    # -----------------------------
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    BOS = "<s>"

    THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
    THINK_EMPTY = "<think></think>"

    TCALL_OPEN, TCALL_CLOSE = "<tool_call>", "</tool_call>"
    TRESP_OPEN, TRESP_CLOSE = "<tool_response>", "</tool_response>"

    # Tool-call internal closers: treat as structural only inside assistant tool_call blocks
    PARAM_CLOSE = "</parameter>"
    FUNC_CLOSE = "</function>"

    # Content-bearing open tags inside tool calls
    FUNC_OPEN_PREFIX = "<function="
    PARAM_OPEN_PREFIX = "<parameter="

    # Newline detection (handles both real '\n' and byte-BPE newline glyphs)
    NL_ONLY_RE = re.compile(r"^[\n\rĊĉĈ]+$")
    NL_ANY_RE = re.compile(r"[\n\rĊĉĈ]")

    tok = df["token"].astype(str)

    # prev/next token (within prompt)
    df["prev_token"] = df.groupby("prompt_ix", sort=False)["token"].shift(1)
    df["next_token"] = df.groupby("prompt_ix", sort=False)["token"].shift(-1)

    df["is_pure_nl"] = tok.str.fullmatch(NL_ONLY_RE, na=False)
    df["has_nl_char"] = tok.str.contains(NL_ANY_RE, na=False)

    # -----------------------------
    # Outer ChatML message parsing
    # -----------------------------
    df["is_im_start"] = tok.eq(IM_START)
    df["is_im_end"] = tok.eq(IM_END)
    df["is_bos"] = tok.eq(BOS)

    # message id increments on <|im_start|>
    df["msg_id"] = df.groupby("prompt_ix", sort=False)["is_im_start"].cumsum()
    df["pos_in_msg"] = df.groupby(["prompt_ix", "msg_id"], sort=False).cumcount()

    # Header ends at the first token that contains a newline char/glyph (e.g., after ROLE in "<|im_start|>ROLE\n")
    header_end_pos = (
        df["pos_in_msg"]
        .where((df["msg_id"] > 0) & df["has_nl_char"])
        .groupby([df["prompt_ix"], df["msg_id"]], sort=False)
        .transform("min")
    ).fillna(np.inf)
    df["header_end_pos"] = header_end_pos

    # Track whether we've seen <|im_end|> within the message
    df["im_end_cum"] = df.groupby(["prompt_ix", "msg_id"], sort=False)["is_im_end"].cumsum()

    # Header tokens are always structural
    df["is_header"] = (df["msg_id"] > 0) & (df["pos_in_msg"] <= df["header_end_pos"])

    # Body tokens are after header and before <|im_end|>
    df["in_body"] = (
        (df["msg_id"] > 0)
        & (df["pos_in_msg"] > df["header_end_pos"])
        & (df["im_end_cum"].eq(0))
    )

    # ---- FIX: reconstruct outer_role by concatenating ALL header tokens after <|im_start|> up to header_end_pos
    header_mask = (
        (df["msg_id"] > 0)
        & (df["pos_in_msg"] > 0)
        & (df["pos_in_msg"] <= df["header_end_pos"])
    )
    header_joined = (
        df.loc[header_mask]
        .groupby(["prompt_ix", "msg_id"], sort=False)["token"]
        .agg("".join)
    )

    outer_role = header_joined.astype(str)
    outer_role = outer_role.str.split("\n", n=1).str[0]
    outer_role = outer_role.str.split("\r", n=1).str[0]
    outer_role = outer_role.str.rstrip("\n\rĊĉĈ").str.strip()

    df = df.join(outer_role.rename("outer_role"), on=["prompt_ix", "msg_id"])
    df["outer_role"] = df["outer_role"].fillna("")

    is_assistant_msg = df["outer_role"].eq("assistant")
    is_user_msg = df["outer_role"].eq("user")
    is_system_msg = df["outer_role"].eq("system")
    is_developer_msg = df["outer_role"].eq("developer")

    # -----------------------------
    # Inner wrapper state tracking
    # -----------------------------
    df["is_think_open"]  = df["in_body"] & is_assistant_msg & tok.eq(THINK_OPEN)
    df["is_think_close"] = df["in_body"] & is_assistant_msg & tok.eq(THINK_CLOSE)
    df["is_think_empty"] = df["in_body"] & is_assistant_msg & tok.eq(THINK_EMPTY)

    df["is_tcall_open"]  = df["in_body"] & is_assistant_msg & tok.eq(TCALL_OPEN)
    df["is_tcall_close"] = df["in_body"] & is_assistant_msg & tok.eq(TCALL_CLOSE)

    df["is_tresp_open"]  = df["in_body"] & is_user_msg & tok.eq(TRESP_OPEN)
    df["is_tresp_close"] = df["in_body"] & is_user_msg & tok.eq(TRESP_CLOSE)

    msg_group = df.groupby(["prompt_ix", "msg_id"], sort=False)

    df["think_open_cum"]  = msg_group["is_think_open"].cumsum()
    df["think_close_cum"] = msg_group["is_think_close"].cumsum()
    df["in_think"] = df["think_open_cum"] > df["think_close_cum"]

    df["tcall_open_cum"]  = msg_group["is_tcall_open"].cumsum()
    df["tcall_close_cum"] = msg_group["is_tcall_close"].cumsum()
    df["in_tool_call"] = df["tcall_open_cum"] > df["tcall_close_cum"]

    df["tresp_open_cum"]  = msg_group["is_tresp_open"].cumsum()
    df["tresp_close_cum"] = msg_group["is_tresp_close"].cumsum()
    df["in_tool_response"] = df["tresp_open_cum"] > df["tresp_close_cum"]

    # -----------------------------
    # Tag / structural token mask
    # -----------------------------
    prev_tok = df["prev_token"].astype(str)
    next_tok = df["next_token"].astype(str)

    prev_is_func_open = prev_tok.str.startswith(FUNC_OPEN_PREFIX, na=False)
    prev_is_param_open = prev_tok.str.startswith(PARAM_OPEN_PREFIX, na=False)
    next_is_func_open = next_tok.str.startswith(FUNC_OPEN_PREFIX, na=False)
    next_is_param_open = next_tok.str.startswith(PARAM_OPEN_PREFIX, na=False)

    df["is_control"] = df["is_bos"] | df["is_im_start"] | df["is_im_end"] | (df["msg_id"].eq(0))

    df["is_wrapper_tag"] = (
        df["is_header"]
        | df["is_im_start"] | df["is_im_end"]
        | df["is_think_open"] | df["is_think_close"] | df["is_think_empty"]
        | df["is_tcall_open"] | df["is_tcall_close"]
        | df["is_tresp_open"] | df["is_tresp_close"]
    )

    df["is_toolcall_internal_close"] = (
        df["in_body"]
        & is_assistant_msg
        & df["in_tool_call"]
        & tok.isin([PARAM_CLOSE, FUNC_CLOSE])
    )

    # Template-attached structural newlines (pure newline tokens only)
    df["is_struct_nl"] = df["is_pure_nl"] & (
        prev_tok.eq(IM_END)
        | prev_tok.eq(THINK_OPEN)
        | next_tok.eq(THINK_CLOSE)
        | prev_tok.eq(THINK_CLOSE)
        | next_tok.eq(TCALL_OPEN)
        | prev_tok.eq(TCALL_OPEN)
        | next_tok.eq(TCALL_CLOSE)
        | prev_tok.eq(TCALL_CLOSE)
        | prev_tok.eq(TRESP_OPEN)
        | next_tok.eq(TRESP_CLOSE)
        | prev_tok.eq(TRESP_CLOSE)
        | next_tok.eq(TRESP_OPEN)
        | prev_is_func_open
        | prev_is_param_open
        | next_tok.eq(PARAM_CLOSE)
        | prev_tok.eq(PARAM_CLOSE)
        | next_tok.eq(FUNC_CLOSE)
        | prev_tok.eq(FUNC_CLOSE)
        | next_is_param_open
        | next_is_func_open
    )

    df["is_tag"] = (
        df["is_control"]
        | df["is_wrapper_tag"]
        | df["is_toolcall_internal_close"]
        | df["is_struct_nl"]
    )

    df["potential_content"] = df["in_body"] & ~df["is_tag"]

    # -----------------------------
    # Content-only role assignment
    # -----------------------------
    role = np.array([None] * len(df), dtype=object)

    # system / developer
    role[(df["potential_content"] & is_system_msg).to_numpy()] = "system"
    role[(df["potential_content"] & is_developer_msg).to_numpy()] = "developer"

    # user vs tool (inside tool_response)
    user_content = df["potential_content"] & is_user_msg
    role[(user_content & df["in_tool_response"]).to_numpy()] = "tool"
    role[(user_content & ~df["in_tool_response"]).to_numpy()] = "user"

    # assistant: tool_call > cot > assistant
    asst_content = df["potential_content"] & is_assistant_msg
    role[(asst_content & df["in_tool_call"]).to_numpy()] = "tool_call"
    role[(asst_content & ~df["in_tool_call"] & df["in_think"]).to_numpy()] = "cot"
    role[(asst_content & ~df["in_tool_call"] & ~df["in_think"]).to_numpy()] = "assistant"

    df["role"] = role
    df["is_content"] = df["role"].notna()

    # -----------------------------
    # seg_ix + token_in_seg_ix (content-only)
    # -----------------------------
    is_labeled = df["is_content"]

    prev_labeled = is_labeled.groupby(df["prompt_ix"], sort=False).shift(1, fill_value=False)
    prev_role = df.groupby("prompt_ix", sort=False)["role"].shift(1)

    is_new_seg = is_labeled & (~prev_labeled | (df["role"] != prev_role))
    seg_counter = is_new_seg.groupby(df["prompt_ix"], sort=False).cumsum()  # 1,2,3,...

    df["seg_ix"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[is_labeled, "seg_ix"] = (seg_counter[is_labeled] - 1).astype("Int64")

    df["token_in_seg_ix"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[is_labeled, "token_in_seg_ix"] = (
        df.loc[is_labeled]
        .groupby(["prompt_ix", "seg_ix"], sort=False)
        .cumcount()
        .astype("Int64")
    )

    # -----------------------------
    # Cleanup
    # -----------------------------
    drop_cols = [
        "prev_token", "next_token",
        "is_pure_nl", "has_nl_char",
        "is_im_start", "is_im_end", "is_bos",
        "msg_id", "pos_in_msg", "header_end_pos", "im_end_cum",
        "is_header", "in_body", "outer_role",
        "is_think_open", "is_think_close", "is_think_empty",
        "is_tcall_open", "is_tcall_close",
        "is_tresp_open", "is_tresp_close",
        "think_open_cum", "think_close_cum",
        "tcall_open_cum", "tcall_close_cum",
        "tresp_open_cum", "tresp_close_cum",
        "in_think", "in_tool_call", "in_tool_response",
        "is_control", "is_wrapper_tag", "is_toolcall_internal_close", "is_struct_nl", "is_tag",
        "potential_content",
    ]

    out = df.drop(columns=drop_cols, errors="ignore")
    out = out.sort_values(["prompt_ix", "token_ix"]).reset_index(drop=True)
    return out


def label_jamba_content_roles(sample_df: pd.DataFrame, thinking_prefix = "") -> pd.DataFrame:
    """
    Labels Jamba token streams with *content-only* roles.

    ChatML envelope:
      <|im_start|>ROLE\\n ... <|im_end|>\\n

    Key features:
      - Assistant: <think>...</think> => cot; <tool_call>...</tool_call> => tool_call
      - Tool outputs: embedded in user wrapper via <tool_response>...</tool_response> => tool
      - Jamba-specific: template may inject a natural-language thinking_prefix into USER messages;
        those tokens are treated as structural tags (role=None).

    Robustness:
      - Handles newline tokens rendered as literal newline glyphs (\\n, Ċ, etc.) OR as "<0x0A>" / "<0x0D>".

    Returns original columns +:
      - role: {system, developer, user, assistant, cot, tool_call, tool} or None
      - is_content: bool
      - seg_ix: contiguous run index of content tokens with same role within prompt (tags break runs)
      - token_in_seg_ix: 0-based within seg_ix (NA for non-content)
    """
    required = {"prompt_ix", "token_ix", "token"}
    missing = required - set(sample_df.columns)
    if missing:
        raise ValueError(f"sample_df missing required columns: {sorted(missing)}")

    df = (
        sample_df.sort_values(["prompt_ix", "token_ix"])
        .reset_index(drop=True)
        .copy()
    )

    tok = df["token"].astype(str)

    # -----------------------------
    # Sentinels / wrappers
    # -----------------------------
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"

    COMMON_BOS = {"<s>", "<|begin_of_text|>", "<|bos|>", "<|startoftext|>"}

    THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
    THINK_EMPTY = "<think></think>"

    TCALL_OPEN, TCALL_CLOSE = "<tool_call>", "</tool_call>"
    TRESP_OPEN, TRESP_CLOSE = "<tool_response>", "</tool_response>"

    # Newline markers:
    # - literal newlines / byte-BPE newline glyphs
    # - OR explicit hex byte renderings like "<0x0A>"
    NL_MARKER_ANY_RE = re.compile(r"(?i)(?:[\n\rĊĉĈ]|<0x0a>|<0x0d>)")
    NL_MARKER_ONLY_RE = re.compile(r"(?i)^(?:(?:[\n\rĊĉĈ])|(?:<0x0a>)|(?:<0x0d>))+$")
    NL_CUT_RE = r"(?is)(?:[\n\rĊĉĈ]|<0x0a>|<0x0d>).*$"
    NL_MIXED_RE = r"(?is)(?:[\n\rĊĉĈ]|<0x0a>|<0x0d>).+"

    # For wrapper matching, strip both whitespace and newline marker tokens rendered as "<0x0A>"
    STRIP_CHARS = "\n\rĊĉĈ \t"

    # Normalize "<0x0A>" / "<0x0D>" to real newlines for stripping/matching convenience
    tok_norm = tok.str.replace(r"(?i)<0x0a>", "\n", regex=True).str.replace(r"(?i)<0x0d>", "\r", regex=True)
    tok_stripped = tok_norm.str.strip(STRIP_CHARS)

    # prev/next token (within prompt)
    df["prev_token"] = df.groupby("prompt_ix", sort=False)["token"].shift(1)
    df["next_token"] = df.groupby("prompt_ix", sort=False)["token"].shift(-1)

    prev_norm = df["prev_token"].astype(str).str.replace(r"(?i)<0x0a>", "\n", regex=True).str.replace(r"(?i)<0x0d>", "\r", regex=True)
    next_norm = df["next_token"].astype(str).str.replace(r"(?i)<0x0a>", "\n", regex=True).str.replace(r"(?i)<0x0d>", "\r", regex=True)

    df["is_pure_nl"] = tok.str.fullmatch(NL_MARKER_ONLY_RE, na=False)
    df["has_nl_char"] = tok.str.contains(NL_MARKER_ANY_RE, na=False)

    # -----------------------------
    # Outer ChatML message parsing
    # -----------------------------
    df["is_im_start"] = tok.eq(IM_START)
    df["is_im_end"] = tok.eq(IM_END)
    df["is_common_bos"] = tok.isin(COMMON_BOS)

    df["msg_id"] = df.groupby("prompt_ix", sort=False)["is_im_start"].cumsum()
    df["pos_in_msg"] = df.groupby(["prompt_ix", "msg_id"], sort=False).cumcount()

    # header_end_pos = first token position containing ANY newline marker (including "<0x0A>")
    header_end_pos = (
        df["pos_in_msg"]
        .where((df["msg_id"] > 0) & df["has_nl_char"])
        .groupby([df["prompt_ix"], df["msg_id"]], sort=False)
        .transform("min")
    ).fillna(np.inf)
    df["header_end_pos"] = header_end_pos

    # Mixed header-ending token means newline marker plus extra chars after it
    df["header_end_token_mixed"] = (
        (df["msg_id"] > 0)
        & (df["pos_in_msg"] == df["header_end_pos"])
        & (~df["is_pure_nl"])
        & tok.str.contains(NL_MIXED_RE, regex=True, na=False)
    )

    # Track message end
    df["im_end_cum"] = df.groupby(["prompt_ix", "msg_id"], sort=False)["is_im_end"].cumsum()

    # Header tokens (structural): role text + header newline token (unless it's a mixed token)
    df["is_header"] = (
        (df["msg_id"] > 0)
        & (df["pos_in_msg"] > 0)
        & (
            (df["pos_in_msg"] < df["header_end_pos"])
            | ((df["pos_in_msg"] == df["header_end_pos"]) & ~df["header_end_token_mixed"])
        )
    )

    # Body tokens are after header newline (or the mixed header token itself), before <|im_end|>
    df["in_body"] = (
        (df["msg_id"] > 0)
        & (df["im_end_cum"].eq(0))
        & (
            (df["pos_in_msg"] > df["header_end_pos"])
            | ((df["pos_in_msg"] == df["header_end_pos"]) & df["header_end_token_mixed"])
        )
    )

    # Reconstruct outer_role by joining header tokens up to header_end_pos, then cutting at first newline marker.
    header_mask = (
        (df["msg_id"] > 0)
        & (df["pos_in_msg"] > 0)
        & (df["pos_in_msg"] <= df["header_end_pos"])
    )
    header_joined = (
        df.loc[header_mask]
        .groupby(["prompt_ix", "msg_id"], sort=False)["token"]
        .agg("".join)
    )
    outer_role = (
        header_joined.astype(str)
        .str.replace(NL_CUT_RE, "", regex=True)
        .str.strip()
    )

    df = df.join(outer_role.rename("outer_role"), on=["prompt_ix", "msg_id"])
    df["outer_role"] = df["outer_role"].fillna("")

    is_system_msg = df["outer_role"].eq("system")
    is_user_msg = df["outer_role"].eq("user")
    is_assistant_msg = df["outer_role"].eq("assistant")
    is_tool_msg = df["outer_role"].eq("tool")
    is_developer_msg = df["outer_role"].eq("developer")

    msg_g = df.groupby(["prompt_ix", "msg_id"], sort=False)

    # -----------------------------
    # Inner wrappers (SCOPED)
    # -----------------------------
    # Think toggles ONLY inside assistant body (prevents user injected prefix from toggling cot)
    df["is_think_open"] = df["in_body"] & is_assistant_msg & tok_stripped.eq(THINK_OPEN)
    df["is_think_close"] = df["in_body"] & is_assistant_msg & tok_stripped.eq(THINK_CLOSE)
    df["is_think_empty"] = df["in_body"] & is_assistant_msg & tok_stripped.eq(THINK_EMPTY)

    df["think_open_cum"] = msg_g["is_think_open"].cumsum()
    df["think_close_cum"] = msg_g["is_think_close"].cumsum()
    df["in_think"] = df["think_open_cum"] > df["think_close_cum"]

    # Tool call wrappers only inside assistant
    df["is_tcall_open"] = df["in_body"] & is_assistant_msg & tok_stripped.eq(TCALL_OPEN)
    df["is_tcall_close"] = df["in_body"] & is_assistant_msg & tok_stripped.eq(TCALL_CLOSE)

    df["tcall_open_cum"] = msg_g["is_tcall_open"].cumsum()
    df["tcall_close_cum"] = msg_g["is_tcall_close"].cumsum()
    df["in_tool_call"] = df["tcall_open_cum"] > df["tcall_close_cum"]

    # Tool response wrappers only inside user
    df["is_tresp_open"] = df["in_body"] & is_user_msg & tok_stripped.eq(TRESP_OPEN)
    df["is_tresp_close"] = df["in_body"] & is_user_msg & tok_stripped.eq(TRESP_CLOSE)

    df["tresp_open_cum"] = msg_g["is_tresp_open"].cumsum()
    df["tresp_close_cum"] = msg_g["is_tresp_close"].cumsum()
    df["in_tool_response"] = df["tresp_open_cum"] > df["tresp_close_cum"]

    # -----------------------------
    # Jamba-specific: detect injected thinking_prefix at start of USER body
    # -----------------------------
    def _norm_nl(s: str) -> str:
        # Normalize all newline renderings to '\n' for matching
        s = re.sub(r"(?i)<0x0d>", "\n", s)
        s = re.sub(r"(?i)<0x0a>", "\n", s)
        s = s.replace("Ċ", "\n").replace("ĉ", "\n").replace("Ĉ", "\n")
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        return s

    target_prefix = _norm_nl(thinking_prefix) if thinking_prefix else ""

    df["is_thinking_prefix"] = False
    if target_prefix:
        user_body_mask = df["in_body"] & is_user_msg

        for (p, m), g in df.loc[user_body_mask].groupby(["prompt_ix", "msg_id"], sort=False):
            g = g.sort_values("pos_in_msg", kind="stable")
            idxs = g.index.to_numpy()
            tokens = g["token"].astype(str).tolist()

            acc = ""
            matched = False
            k = 0

            for i, t in enumerate(tokens):
                acc = _norm_nl(acc + t)

                if target_prefix.startswith(acc):
                    continue

                if acc.startswith(target_prefix):
                    # Prefix boundary crossed within this token -> merged => content,
                    # so mark tokens BEFORE this one as thinking_prefix tags.
                    matched = True
                    k = i
                    break

                matched = False
                k = 0
                break

            if not matched and acc == target_prefix:
                matched = True
                k = len(tokens)

            if matched and k > 0:
                df.loc[idxs[:k], "is_thinking_prefix"] = True

    # -----------------------------
    # Tag / structural token mask
    # -----------------------------
    prev_tok = df["prev_token"].astype(str)
    next_tok = df["next_token"].astype(str)

    prev_stripped = prev_norm.str.strip(STRIP_CHARS)
    next_stripped = next_norm.str.strip(STRIP_CHARS)

    # Control tokens: envelope, BOS, and tokens before first <|im_start|>
    df["is_control"] = (
        df["is_common_bos"]
        | df["is_im_start"]
        | df["is_im_end"]
        | (df["msg_id"].eq(0))
    )

    # Wrapper tags (structural)
    df["is_wrapper_tag"] = (
        df["is_header"]
        | df["is_im_start"] | df["is_im_end"]
        | df["is_think_open"] | df["is_think_close"] | df["is_think_empty"]
        | df["is_tcall_open"] | df["is_tcall_close"]
        | df["is_tresp_open"] | df["is_tresp_close"]
    )

    # Structural pure-newline tokens: template-attached or wrapper-adjacent
    WRAPPER_STRS = {THINK_OPEN, THINK_CLOSE, THINK_EMPTY, TCALL_OPEN, TCALL_CLOSE, TRESP_OPEN, TRESP_CLOSE}
    df["is_struct_nl"] = df["is_pure_nl"] & (
        prev_tok.eq(IM_END)
        | prev_stripped.isin(WRAPPER_STRS)
        | next_stripped.isin(WRAPPER_STRS)
    )

    df["is_tag"] = df["is_control"] | df["is_wrapper_tag"] | df["is_struct_nl"] | df["is_thinking_prefix"]

    df["potential_content"] = df["in_body"] & ~df["is_tag"]

    # -----------------------------
    # Content-only role assignment
    # -----------------------------
    role = np.array([None] * len(df), dtype=object)

    role[(df["potential_content"] & is_system_msg).to_numpy()] = "system"
    role[(df["potential_content"] & is_developer_msg).to_numpy()] = "developer"
    role[(df["potential_content"] & is_tool_msg).to_numpy()] = "tool"

    user_content = df["potential_content"] & is_user_msg
    role[(user_content & df["in_tool_response"]).to_numpy()] = "tool"
    role[(user_content & ~df["in_tool_response"]).to_numpy()] = "user"

    asst_content = df["potential_content"] & is_assistant_msg
    role[(asst_content & df["in_tool_call"]).to_numpy()] = "tool_call"
    role[(asst_content & ~df["in_tool_call"] & df["in_think"]).to_numpy()] = "cot"
    role[(asst_content & ~df["in_tool_call"] & ~df["in_think"]).to_numpy()] = "assistant"

    df["role"] = role
    df["is_content"] = df["role"].notna()

    # -----------------------------
    # seg_ix + token_in_seg_ix (content-only; tags break runs)
    # -----------------------------
    is_labeled = df["is_content"]
    prev_labeled = is_labeled.groupby(df["prompt_ix"], sort=False).shift(1, fill_value=False)
    prev_role = df.groupby("prompt_ix", sort=False)["role"].shift(1)

    is_new_seg = is_labeled & (~prev_labeled | (df["role"] != prev_role))
    seg_counter = is_new_seg.groupby(df["prompt_ix"], sort=False).cumsum()

    df["seg_ix"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[is_labeled, "seg_ix"] = (seg_counter[is_labeled] - 1).astype("Int64")

    df["token_in_seg_ix"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[is_labeled, "token_in_seg_ix"] = (
        df.loc[is_labeled]
        .groupby(["prompt_ix", "seg_ix"], sort=False)
        .cumcount()
        .astype("Int64")
    )

    # -----------------------------
    # Cleanup
    # -----------------------------
    drop_cols = [
        "prev_token", "next_token",
        "is_pure_nl", "has_nl_char",
        "is_im_start", "is_im_end", "is_common_bos",
        "msg_id", "pos_in_msg", "header_end_pos", "header_end_token_mixed", "im_end_cum",
        "is_header", "in_body", "outer_role",
        "is_think_open", "is_think_close", "is_think_empty",
        "think_open_cum", "think_close_cum", "in_think",
        "is_tcall_open", "is_tcall_close", "tcall_open_cum", "tcall_close_cum", "in_tool_call",
        "is_tresp_open", "is_tresp_close", "tresp_open_cum", "tresp_close_cum", "in_tool_response",
        "is_thinking_prefix",
        "is_control", "is_wrapper_tag", "is_struct_nl", "is_tag",
        "potential_content",
    ]
    out = df.drop(columns=drop_cols, errors="ignore")
    out = out.sort_values(["prompt_ix", "token_ix"]).reset_index(drop=True)
    return out


def label_glm47flash_content_roles(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Labels GLM-4.7-Flash token streams with *content-only* roles.

    Template characteristics:
      - Prefix: "[gMASK]<sop>"
      - Role sentinels: <|system|>, <|user|>, <|assistant|>, <|observation|>
      - Assistant thinking:
          * emits "<think>... </think>" when reasoning is included
          * otherwise emits a standalone "</think>" placeholder
      - Tool calls in assistant:
          <tool_call>{name}<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>
        Adjacent XML tags may be fused into a single token (e.g. "</arg_key><arg_value>"),
        so we treat tokens that are *pure sequences of allowed tags* as structural tags.
      - Tool outputs in observation:
          <tool_response>...</tool_response>

    Semantics:
      - `role` is assigned ONLY to semantic content tokens.
      - Structural/template/sentinel tokens are role=None and is_content=False.
      - We do NOT assume any fixed turn ordering.

    Returns original columns +:
      - role: {system, user, assistant, cot, tool_call, tool} or None
      - is_content: bool
      - seg_ix: contiguous run index of content tokens with same role (tags break runs)
      - token_in_seg_ix: 0-based within seg_ix (NA for non-content)
    """
    required = {"prompt_ix", "token_ix", "token"}
    missing = required - set(sample_df.columns)
    if missing:
        raise ValueError(f"sample_df missing required columns: {sorted(missing)}")

    df = (
        sample_df.sort_values(["prompt_ix", "token_ix"])
        .reset_index(drop=True)
        .copy()
    )

    # -----------------------------
    # Constants
    # -----------------------------
    SYSTEM = "<|system|>"
    USER = "<|user|>"
    ASSISTANT = "<|assistant|>"
    OBS = "<|observation|>"

    ROLE_SENTINELS = {SYSTEM, USER, ASSISTANT, OBS}

    # Prefix is often one token in Flash, but keep older variants too.
    PREFIX_TOKENS = {"[gMASK]<sop>", "[gMASK]", "<sop>", "<eop>"}

    STRIP_CHARS = "\n\rĊĉĈ \t"

    # Wrapper tags
    THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
    TCALL_OPEN, TCALL_CLOSE = "<tool_call>", "</tool_call>"
    TRESP_OPEN, TRESP_CLOSE = "<tool_response>", "</tool_response>"

    ARGK_OPEN, ARGK_CLOSE = "<arg_key>", "</arg_key>"
    ARGV_OPEN, ARGV_CLOSE = "<arg_value>", "</arg_value>"

    # Pure-tag-sequence regexes (for handling fused adjacent tags like "</arg_key><arg_value>")
    ASST_TAG_SEQ_RE = re.compile(
        r"^(?:"
        r"<think>|</think>|"
        r"<tool_call>|</tool_call>|"
        r"<arg_key>|</arg_key>|"
        r"<arg_value>|</arg_value>"
        r")+$"
    )
    OBS_TAG_SEQ_RE = re.compile(r"^(?:<tool_response>|</tool_response>)+$")

    # -----------------------------
    # Normalize token for tag matching (strip whitespace/newline glyphs)
    # -----------------------------
    df["token_norm"] = df["token"].astype(str).str.strip(STRIP_CHARS)

    # -----------------------------
    # Segment by role sentinel
    # -----------------------------
    df["is_seg_start"] = df["token_norm"].isin(ROLE_SENTINELS)
    df["seg_id"] = df.groupby("prompt_ix", sort=False)["is_seg_start"].cumsum()

    df["seg_role_token"] = (
        df["token_norm"]
        .where(df["is_seg_start"])
        .groupby([df["prompt_ix"], df["seg_id"]], sort=False)
        .transform("first")
        .fillna("")
    )

    df["seg_kind"] = np.select(
        [
            df["seg_role_token"].eq(SYSTEM),
            df["seg_role_token"].eq(USER),
            df["seg_role_token"].eq(ASSISTANT),
            df["seg_role_token"].eq(OBS),
        ],
        ["system", "user", "assistant", "observation"],
        default=None,
    )

    in_segment = df["seg_id"] > 0
    is_assistant_seg = df["seg_kind"].eq("assistant")
    is_observation_seg = df["seg_kind"].eq("observation")

    # -----------------------------
    # Tool-call / think membership (assistant only)
    # Use substring-counts to survive mild token fusion like "<tool_call>name"
    # -----------------------------
    tok_raw = df["token"].astype(str)

    df["tcall_open_ct"] = np.where(is_assistant_seg, tok_raw.str.count(re.escape(TCALL_OPEN)), 0)
    df["tcall_close_ct"] = np.where(is_assistant_seg, tok_raw.str.count(re.escape(TCALL_CLOSE)), 0)

    df["think_open_ct"] = np.where(is_assistant_seg, tok_raw.str.count(re.escape(THINK_OPEN)), 0)
    df["think_close_ct"] = np.where(is_assistant_seg, tok_raw.str.count(re.escape(THINK_CLOSE)), 0)

    by_seg = df.groupby(["prompt_ix", "seg_id"], sort=False)

    df["tcall_open_cum"] = by_seg["tcall_open_ct"].cumsum()
    df["tcall_close_cum"] = by_seg["tcall_close_ct"].cumsum()
    df["in_tool_call"] = df["tcall_open_cum"] > df["tcall_close_cum"]

    df["think_open_cum"] = by_seg["think_open_ct"].cumsum()
    df["think_close_cum"] = by_seg["think_close_ct"].cumsum()
    df["in_think"] = df["think_open_cum"] > df["think_close_cum"]

    # -----------------------------
    # Tag mask
    # -----------------------------
    df["is_prefix"] = (df["seg_id"].eq(0)) & df["token_norm"].isin(PREFIX_TOKENS)
    df["is_role_sentinel"] = df["is_seg_start"]

    # Assistant: structural tags include pure sequences of think/tool_call/arg tags.
    # (Scoped to assistant so system tool examples remain system CONTENT.)
    df["is_asst_pure_tag_seq"] = is_assistant_seg & df["token_norm"].str.fullmatch(ASST_TAG_SEQ_RE, na=False)

    # Observation: structural tags include pure sequences of tool_response tags.
    df["is_obs_pure_tag_seq"] = is_observation_seg & df["token_norm"].str.fullmatch(OBS_TAG_SEQ_RE, na=False)

    # Final tag decision
    df["is_tag"] = df["is_prefix"] | df["is_role_sentinel"] | df["is_asst_pure_tag_seq"] | df["is_obs_pure_tag_seq"]

    # Content tokens
    df["is_content"] = in_segment & ~df["is_tag"]

    # -----------------------------
    # Role assignment (content-only)
    # -----------------------------
    role = np.array([None] * len(df), dtype=object)

    # system / user
    role[(df["seg_kind"].eq("system") & df["is_content"]).to_numpy()] = "system"
    role[(df["seg_kind"].eq("user") & df["is_content"]).to_numpy()] = "user"

    # assistant precedence: tool_call > cot > assistant
    asst_content = is_assistant_seg & df["is_content"]
    role[(asst_content & df["in_tool_call"]).to_numpy()] = "tool_call"
    role[(asst_content & ~df["in_tool_call"] & df["in_think"]).to_numpy()] = "cot"
    role[(asst_content & ~df["in_tool_call"] & ~df["in_think"]).to_numpy()] = "assistant"

    # observation: tool output
    obs_content = is_observation_seg & df["is_content"]
    role[obs_content.to_numpy()] = "tool"

    df["role"] = role

    # -----------------------------
    # seg_ix + token_in_seg_ix (content-only; tags break runs)
    # -----------------------------
    is_labeled = df["role"].notna()

    prev_labeled = is_labeled.groupby(df["prompt_ix"], sort=False).shift(1, fill_value=False)
    prev_role = df.groupby("prompt_ix", sort=False)["role"].shift(1)

    is_new_seg = is_labeled & (~prev_labeled | (df["role"] != prev_role))
    seg_counter = is_new_seg.groupby(df["prompt_ix"], sort=False).cumsum()  # 1,2,3,...

    df["seg_ix"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[is_labeled, "seg_ix"] = (seg_counter[is_labeled] - 1).astype("Int64")

    df["token_in_seg_ix"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df.loc[is_labeled, "token_in_seg_ix"] = (
        df.loc[is_labeled].groupby(["prompt_ix", "seg_ix"], sort=False).cumcount().astype("Int64")
    )

    # -----------------------------
    # Cleanup
    # -----------------------------
    drop_cols = [
        "token_norm",
        "is_seg_start", "seg_id", "seg_role_token", "seg_kind",
        "is_prefix", "is_role_sentinel",
        "tcall_open_ct", "tcall_close_ct", "tcall_open_cum", "tcall_close_cum", "in_tool_call",
        "think_open_ct", "think_close_ct", "think_open_cum", "think_close_cum", "in_think",
        "is_asst_pure_tag_seq", "is_obs_pure_tag_seq",
        "is_tag",
    ]
    out = df.drop(columns=drop_cols, errors="ignore")
    out = out.sort_values(["prompt_ix", "token_ix"]).reset_index(drop=True)
    return out


def label_content_roles(model_prefix, sample_df):
    """
    Takes a token-level df, labels each token with its role only within the content span. Makes no assumption about the number of messages in the sequence.
    The input is a token-level dataframe spanning multiple prompts (conversations). Each prompt is identified by prompt_ix and contains a flat serialized token stream (including template tags, role sentinels, wrappers, and semantic content).

    Params: 
        @sample_df: A df with columns:
        - prompt_ix: unique id for each serialized prompt/sequence, equivalent to an index on (batch_ix, sequence_ix)
        - token_ix: position within the prompt
        - token: decoded token string

    Description:
        - We assign `role` ONLY to semantic content tokens.
        - Template/sentinel/control tokens (role tags, begin/end markers, wrappers like <think>, tool tags, BOS/EOS, and template-attached 
          structural whitespace) are labeled with role=None.
        - We never default to a role. Tokens are labeled only when they fall inside a recognized content span for the given template.

    Returns:
        The original df with new columns:
        - role: str in {system, user, developer, cot, assistant, commentary, tool}, or None
        - is_content: bool
        - seg_ix: int or None
        - token_in_seg_ix: int or None
    """
    required_cols = {"prompt_ix", "token_ix", "token"}
    missing = required_cols - set(sample_df.columns)
    if missing:
        raise ValueError(f"sample_df missing required columns: {sorted(missing)}")

    dispatch = {
        'gptoss-20b': label_gptoss_content_roles,
        'gptoss-120b': label_gptoss_content_roles,
        'nemotron-3-nano': label_nemotron3_content_roles,
        'qwen3-30b-a3b': label_qwen3_content_roles,
        'glm-4.6v-flash': label_glm4_content_roles,
        'apriel-1.6-15b-thinker': label_apriel_content_roles,
        'olmo3-7b-think': label_olmo3_content_roles,
        'jamba-reasoning': label_jamba_content_roles,
        'glm-4.7-flash': label_glm47flash_content_roles
    }

    try:
        fn = dispatch[model_prefix]
    except KeyError:
        raise ValueError(f"Model architecture {model_prefix} not supported")

    return fn(sample_df)