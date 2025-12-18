"""
Helper functions to assign tokens to their roles
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

    def _has_channel(toks_lc, channel: str) -> bool:
        # Support either "<|channel|>analysis" as one token OR "<|channel|>", "analysis" as two tokens.
        combined = f"<|channel|>{channel}"
        if combined in toks_lc:
            return True
        for i in range(len(toks_lc) - 1):
            if toks_lc[i] == "<|channel|>" and toks_lc[i + 1] == channel:
                return True
        return False

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
    df["fcalls_open_cum"] = by_seg["is_fcalls_open"].cumsum()
    df["fcalls_close_cum"] = by_seg["is_fcalls_close"].cumsum()
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


def label_devstral2_content_roles(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Label Devstral-Small-2-24B-Instruct token streams with content-only roles.

    Returns:
        Original df + {role, is_content, seg_ix, token_in_seg_ix}.

    Description:
        - `[AVAILABLE_TOOLS] ... [/AVAILABLE_TOOLS]` content is labeled as `system` (wrappers are non-content).
        - Assistant has no explicit opener: any tokens outside bracketed blocks after a `[/INST]` (or `[/TOOL_RESULTS]`)
          are treated as assistant content until `</s>`; if `</s>` is missing (truncated), a new `[INST]` or
          `[TOOL_RESULTS]` implicitly ends the assistant span.
        - Tool calls have no explicit closer: tool_call content starts after `[TOOL_CALLS]` and runs until `</s>`
          or a new top-level block; `[ARGS]` is structural.
        - All wrappers (`[INST]`, `[/INST]`, `[SYSTEM_PROMPT]`, `[/SYSTEM_PROMPT]`, `[AVAILABLE_TOOLS]`, `[/AVAILABLE_TOOLS]`,
          `[TOOL_RESULTS]`, `[/TOOL_RESULTS]`, `[TOOL_CALLS]`, `[ARGS]`, `[IMG]`, `<s>`, `</s>`) are non-content (role=None).
    """
    required = {"prompt_ix", "token_ix", "token"}
    missing = required - set(sample_df.columns)
    if missing:
        raise ValueError(f"sample_df missing required columns: {sorted(missing)}")

    TAG_SPECS = [
        ("bos", "<s>"),
        ("eos", "</s>"),
        ("sys_open", "[SYSTEM_PROMPT]"),
        ("sys_close", "[/SYSTEM_PROMPT]"),
        ("tools_open", "[AVAILABLE_TOOLS]"),
        ("tools_close", "[/AVAILABLE_TOOLS]"),
        ("inst_open", "[INST]"),
        ("inst_close", "[/INST]"),
        ("toolres_open", "[TOOL_RESULTS]"),
        ("toolres_close", "[/TOOL_RESULTS]"),
        ("toolcalls", "[TOOL_CALLS]"),
        ("args", "[ARGS]"),
        ("img", "[IMG]"),
    ]

    def _find_token_spans(text: str, starts: np.ndarray, ends: np.ndarray, literal: str):
        """Find occurrences of `literal` in `text`, return spans as (tok_start, tok_end)."""
        spans = []
        for m in re.finditer(re.escape(literal), text):
            cs, ce = m.start(), m.end()
            tok_start = int(np.searchsorted(ends, cs, side="right"))
            tok_end = int(np.searchsorted(starts, ce, side="left") - 1)
            if 0 <= tok_start <= tok_end < len(starts):
                spans.append((tok_start, tok_end))
        return spans

    def _process_prompt(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("token_ix").reset_index(drop=True).copy()
        toks = g["token"].astype(str).tolist()
        n = len(toks)

        # Build char offsets to map substring matches back to token indices
        lens = np.fromiter((len(t) for t in toks), dtype=int, count=n)
        starts = np.zeros(n, dtype=int)
        if n > 1:
            starts[1:] = np.cumsum(lens)[:-1]
        ends = starts + lens
        text = "".join(toks)

        # Precompute possible tag spans; we still context-gate acceptance during scanning.
        events_start: dict[int, list[tuple[str, int, int]]] = {}
        for tag_name, literal in TAG_SPECS:
            for s, e in _find_token_spans(text, starts, ends, literal):
                events_start.setdefault(s, []).append((tag_name, e, len(literal)))

        # Prefer longer literals if multiple tags start at the same token index
        for s in events_start:
            events_start[s].sort(key=lambda x: (-x[2], -(x[1] - s)))

        # FSM state
        mode = "outside"  # outside | system | available_tools | user | tool_results | assistant
        in_tool_call = False
        seen_user = False  # used only to gate system_open to the very beginning

        role = np.array([None] * n, dtype=object)

        def _accept(tag: str) -> bool:
            nonlocal mode, in_tool_call, seen_user

            if tag in ("bos", "eos"):
                return True

            if tag == "sys_open":
                return mode == "outside" and not seen_user
            if tag == "sys_close":
                return mode == "system"

            if tag == "tools_open":
                return mode == "outside"
            if tag == "tools_close":
                return mode == "available_tools"

            if tag == "inst_open":
                # allow this even if EOS is missing (truncated assistant)
                return mode in ("outside", "assistant")
            if tag == "inst_close":
                return mode == "user"

            if tag == "toolres_open":
                # allow this even if EOS is missing (truncated assistant)
                return mode in ("outside", "assistant")
            if tag == "toolres_close":
                return mode == "tool_results"

            if tag == "toolcalls":
                return mode == "assistant"
            if tag == "args":
                return mode == "assistant" and in_tool_call
            if tag == "img":
                return mode == "user"

            return False

        def _transition(tag: str):
            nonlocal mode, in_tool_call, seen_user

            if tag == "bos":
                return
            if tag == "eos":
                mode = "outside"
                in_tool_call = False
                return

            if tag == "sys_open":
                mode = "system"
                return
            if tag == "sys_close":
                mode = "outside"
                return

            if tag == "tools_open":
                mode = "available_tools"
                return
            if tag == "tools_close":
                mode = "outside"
                return

            if tag == "inst_open":
                mode = "user"
                in_tool_call = False
                seen_user = True
                return
            if tag == "inst_close":
                mode = "assistant"
                in_tool_call = False
                return

            if tag == "toolres_open":
                mode = "tool_results"
                in_tool_call = False
                return
            if tag == "toolres_close":
                mode = "assistant"
                in_tool_call = False
                return

            if tag == "toolcalls":
                in_tool_call = True
                return
            if tag == "args":
                return
            if tag == "img":
                return

        i = 0
        while i < n:
            accepted = False
            if i in events_start:
                for tag_name, tag_end, _ in events_start[i]:
                    if _accept(tag_name):
                        # Tag span => structural (role None)
                        role[i : tag_end + 1] = None
                        _transition(tag_name)
                        i = tag_end + 1
                        accepted = True
                        break
            if accepted:
                continue

            # Content token (role depends on current mode)
            if mode == "system":
                role[i] = "system"
            elif mode == "available_tools":
                role[i] = "system"  # explicit design choice: AVAILABLE_TOOLS content is system
            elif mode == "user":
                role[i] = "user"
            elif mode == "tool_results":
                role[i] = "tool"
            elif mode == "assistant":
                role[i] = "tool_call" if in_tool_call else "assistant"
            else:
                role[i] = None

            i += 1

        # Content mask is exactly "role is not None"
        role_s = pd.Series(role, index=g.index, dtype="object")
        is_content = role_s.notna()

        # seg_ix / token_in_seg_ix only for labeled tokens
        prev_labeled = is_content.shift(1, fill_value=False)
        prev_role = role_s.shift(1)

        is_new_seg = is_content & (~prev_labeled | (role_s != prev_role))
        seg_counter = is_new_seg.cumsum()

        seg_ix = pd.Series(pd.NA, index=g.index, dtype="Int64")
        seg_ix.loc[is_content] = (seg_counter.loc[is_content] - 1).astype("Int64")

        token_in_seg_ix = pd.Series(pd.NA, index=g.index, dtype="Int64")
        token_in_seg_ix.loc[is_content] = (
            g.loc[is_content].groupby(seg_ix.loc[is_content], sort=False).cumcount().astype("Int64")
        )

        g["role"] = role_s.where(is_content, None)
        g["is_content"] = is_content.astype(bool)
        g["seg_ix"] = seg_ix
        g["token_in_seg_ix"] = token_in_seg_ix
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


def label_qwen3coder_content_roles(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Label Qwen3-Coder (ChatML) token streams with content-only roles.

    Description:
        - Messages are ChatML-framed: <|im_start|>{role}\\n ... <|im_end|>\\n.
        - Tool calls are delimited by <tool_call>...</tool_call> inside assistant messages
          (wrappers are structural/non-content).
        - Tool results are encoded inside user messages as <tool_response>...</tool_response>
          blocks (wrappers are structural/non-content; block contents are role=tool).
        - The injected tools schema block is treated as system content (i.e., <tool_call> text in
          the system instructions is NOT treated as a tool-call wrapper).
        - Newline-only tokens that are ALWAYS template-inserted adjacent to tags (header newline,
          post-<|im_end|> newline, newline immediately after open wrappers, newline immediately
          before close wrappers, and newline immediately before <tool_call>) are non-content.

    Returns:
        - role: one of {system, user, assistant, tool_call, tool} or None
        - is_content: bool
        - seg_ix: int or None
        - token_in_seg_ix: int or None
    """
    required = {"prompt_ix", "token_ix", "token"}
    missing = required - set(sample_df.columns)
    if missing:
        raise ValueError(f"sample_df missing required columns: {sorted(missing)}")

    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"

    TOOL_CALL_OPEN = "<tool_call>"
    TOOL_CALL_CLOSE = "</tool_call>"
    TOOL_RESP_OPEN = "<tool_response>"
    TOOL_RESP_CLOSE = "</tool_response>"

    # Newline-only token detector (covers common newline glyphs too)
    NL_ONLY_RE = re.compile(r"^[\nĊĉĈ]+$")

    def _norm_nl(s: str) -> str:
        # Normalize common "newline glyph" tokens to '\n' so header detection works reliably.
        return s.replace("Ċ", "\n").replace("ĉ", "\n").replace("Ĉ", "\n")

    def _process_prompt(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("token_ix").reset_index(drop=True)
        toks = g["token"].astype(str).tolist()
        n = len(toks)

        # Fast primitive masks
        is_im_start = np.fromiter((t == IM_START for t in toks), dtype=bool, count=n)
        is_im_end = np.fromiter((t == IM_END for t in toks), dtype=bool, count=n)
        is_nl_only = np.fromiter((NL_ONLY_RE.fullmatch(t) is not None for t in toks), dtype=bool, count=n)

        # ---- Pass 1: assign ChatML container role per token (system/user/assistant) and header tokens ----
        container = np.array([None] * n, dtype=object)      # role for BODY tokens only
        is_header_tok = np.zeros(n, dtype=bool)

        in_msg = False
        in_header = False
        header_buf = ""
        current_role = None

        for i, tok in enumerate(toks):
            if tok == IM_START:
                in_msg = True
                in_header = True
                header_buf = ""
                current_role = None
                continue

            if tok == IM_END:
                in_msg = False
                in_header = False
                current_role = None
                continue

            if in_header:
                # Everything between <|im_start|> and the first newline char is structural header.
                is_header_tok[i] = True
                header_buf += tok

                if "\n" in _norm_nl(tok):
                    # Role is whatever precedes the first newline in the header buffer.
                    role_part = _norm_nl(header_buf).split("\n", 1)[0].strip()
                    current_role = role_part
                    in_header = False
                continue

            if in_msg:
                container[i] = current_role

        # ---- Wrapper tags (structural only in the intended container role) ----
        tok_eq = np.fromiter  # alias

        is_tool_call_open = tok_eq((t == TOOL_CALL_OPEN for t in toks), dtype=bool, count=n) & (container == "assistant")
        is_tool_call_close = tok_eq((t == TOOL_CALL_CLOSE for t in toks), dtype=bool, count=n) & (container == "assistant")

        is_tool_resp_open = tok_eq((t == TOOL_RESP_OPEN for t in toks), dtype=bool, count=n) & (container == "user")
        is_tool_resp_close = tok_eq((t == TOOL_RESP_CLOSE for t in toks), dtype=bool, count=n) & (container == "user")

        # Depths (open included; close excluded). Tag mask will remove the open token anyway.
        tool_call_depth = np.cumsum(is_tool_call_open.astype(int) - is_tool_call_close.astype(int))
        in_tool_call = tool_call_depth > 0

        tool_resp_depth = np.cumsum(is_tool_resp_open.astype(int) - is_tool_resp_close.astype(int))
        in_tool_resp = tool_resp_depth > 0

        # ---- Structural newline-only tokens always paired with tags ----
        # - newline immediately after <|im_end|> is always template-inserted
        prev_is_im_end = np.concatenate([[False], is_im_end[:-1]])
        is_nl_after_im_end = is_nl_only & prev_is_im_end

        # - newline immediately after open wrappers is always template-inserted
        prev_is_open_wrapper = np.concatenate([[False], (is_tool_call_open | is_tool_resp_open)[:-1]])
        is_nl_after_open_wrapper = is_nl_only & prev_is_open_wrapper

        # - newline immediately before close wrappers is always template-inserted
        next_is_close_wrapper = np.concatenate([(is_tool_call_close | is_tool_resp_close)[1:], [False]])
        is_nl_before_close_wrapper = is_nl_only & next_is_close_wrapper

        # - newline immediately after </tool_response> is always template-inserted
        prev_is_tool_resp_close = np.concatenate([[False], is_tool_resp_close[:-1]])
        is_nl_after_tool_resp_close = is_nl_only & prev_is_tool_resp_close

        # - newline immediately before <tool_call> is always template-inserted by the template branch that emits tool calls
        next_is_tool_call_open = np.concatenate([is_tool_call_open[1:], [False]])
        is_nl_before_tool_call_open = is_nl_only & next_is_tool_call_open

        is_struct_nl = (
            is_nl_after_im_end
            | is_nl_after_open_wrapper
            | is_nl_before_close_wrapper
            | is_nl_after_tool_resp_close
            | is_nl_before_tool_call_open
        )

        # ---- Tag mask ----
        # Note: We treat <tool_call> / <tool_response> as tags ONLY when they are wrappers
        # (i.e., in assistant/user containers respectively). This prevents mislabeling the
        # system tool-instruction example text as structural.
        is_tag = (
            is_im_start
            | is_im_end
            | is_header_tok
            | is_tool_call_open | is_tool_call_close
            | is_tool_resp_open | is_tool_resp_close
            | is_struct_nl
        )

        # ---- Content + role assignment ----
        is_content = (container != None) & ~is_tag

        role = np.array([None] * n, dtype=object)

        # Base container roles
        role[is_content & (container == "system")] = "system"

        user_content = is_content & (container == "user")
        role[user_content & in_tool_resp] = "tool"
        role[user_content & ~in_tool_resp] = "user"

        asst_content = is_content & (container == "assistant")
        role[asst_content & in_tool_call] = "tool_call"
        role[asst_content & ~in_tool_call] = "assistant"

        g["is_content"] = is_content.astype(bool)
        g["role"] = role

        # ---- seg_ix + token_in_seg_ix (per prompt; only for role!=None) ----
        is_labeled = role != None

        prev_labeled = np.concatenate([[False], is_labeled[:-1]])
        prev_role = np.concatenate([[None], role[:-1]])

        is_new_seg = is_labeled & (~prev_labeled | (role != prev_role))
        seg_counter = np.cumsum(is_new_seg.astype(int))  # 1,2,3,... at segment starts

        seg_ix_int = np.full(n, -1, dtype=int)
        seg_ix_int[is_labeled] = (seg_counter[is_labeled] - 1)

        token_in_seg_int = np.full(n, -1, dtype=int)
        prev_seg = -1
        pos = 0
        for i in range(n):
            s = seg_ix_int[i]
            if s < 0:
                prev_seg = -1
                continue
            if s != prev_seg:
                prev_seg = s
                pos = 0
            else:
                pos += 1
            token_in_seg_int[i] = pos

        seg_ix_series = pd.Series(seg_ix_int, dtype="Int64").mask(pd.Series(seg_ix_int) < 0)
        tok_in_seg_series = pd.Series(token_in_seg_int, dtype="Int64").mask(pd.Series(token_in_seg_int) < 0)

        g["seg_ix"] = seg_ix_series
        g["token_in_seg_ix"] = tok_in_seg_series
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
        'gptoss20': label_gptoss_content_roles,
        'gptoss120': label_gptoss_content_roles,
        'olmo3-7i': label_olmo3_content_roles,
        'glm-46v-flash': label_glm4_content_roles,
        'apriel': label_apriel_content_roles,
        'qwen3coder': label_qwen3coder_content_roles
    }

    try:
        fn = dispatch[model_prefix]
    except KeyError:
        raise ValueError(f"Model architecture {model_prefix} not supported")

    return fn(sample_df)