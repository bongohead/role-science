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
        - is_content: bool (True iff token is semantic content inside a message span)
        - seg_ix: int or None (contiguous runs of labeled content tokens with same role, per prompt)
        - token_in_seg_ix: int or None (0-based index within seg_ix)
    """
    required = {"prompt_ix", "token_ix", "token"}
    missing = required - set(sample_df.columns)
    if missing:
        raise ValueError(f"sample_df missing required columns: {sorted(missing)}")

    d = (
        sample_df
        .sort_values(["prompt_ix", "token_ix"])
        .reset_index(drop=True)
        .copy()
    )

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
        columns=[
            "msg_seg_id", "is_message", "is_close", "after_msg", "before_end",
            "is_header_tok", "segment_role",
        ],
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
        # NEW: assistant-only structural whitespace adjacency
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

def label_content_roles(model_architecture, sample_df):
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
        "gptoss": label_gptoss_content_roles,
        # "qwen3moe": label_qwen3_content_roles,
        # "glm46v": label_glm4_content_roles,
        # "olmo3": label_olmo3_content_roles,
        "apriel": label_apriel_content_roles,
        # "devstral2": label_devstral2_content_roles,
    }

    if model_architecture == "devstral2":
        raise NotImplementedError("devstral2 not yet supported")
    try:
        fn = dispatch[model_architecture]
    except KeyError:
        raise ValueError(f"Model architecture {model_architecture} not supported")

    return fn(sample_df)