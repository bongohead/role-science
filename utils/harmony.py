"""
Custom chat templater without prebuilt junk from GPT-OSS (dates etc) from tokenizer.apply_chat_template
"""

def harmony_message(role: str, content: str, *, channel: str | None = None) -> str:
    """
    Wrap arbitrary text as a Harmony message for GPT-OSS.
    Returns a fully formed message ending with <|end|>.
    """
    if role not in {'system', 'developer', 'user', 'assistant'}:
        raise ValueError(f"role must be one of {'system', 'developer', 'user', 'assistant'}")
    if role == "assistant":
        if channel not in {'analysis', 'commentary', 'final'}:
            raise ValueError("assistant messages require channel in {'analysis', 'commentary', 'final'}")
        header = f"{role}<|channel|>{channel}<|message|>"
    else:
        if channel is not None:
            raise ValueError("only assistant messages may specify a channel")
        header = f"{role}<|message|>"
    return f"<|start|>{header}{content}<|end|>"

def render_prompt(messages: list[tuple[str, str, str | None]], *, open_for_completion: bool = True) -> str:
    """
    Messages: list of (role, content, channel_or_None)
      - non-assistant: (role, content, None)
      - assistant: (assistant, content, channel)
    If open_for_completion = True, appends '<|start|>assistant' for the next turn.

    Example:
        messages = [
            ('user', 'Hello, how are you?', None),
            ('assistant', 'I am good, thank you!', 'analysis'),
            ('assistant', 'My favorite color is blue.', 'final')
            ('user', 'What is your favorite color?', None)
        ]
        render_prompt(messages)
    """
    parts = [harmony_message(r, c, channel = ch) for (r, c, ch) in messages]
    return "".join(parts) + ("<|start|>assistant" if open_for_completion else "")

def normalize_harmony(reply: str) -> str:
    """
    After generation, replace trailing <|return|> with <|end|> before storing in history (as per Harmony guidance) for multiturn.
    """
    reply = reply.rstrip()
    return reply[:-10] + "<|end|>" if reply.endswith("<|return|>") else reply


