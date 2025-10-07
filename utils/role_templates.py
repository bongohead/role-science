"""
Render single messages
""" 

def render_single_gptoss(role: str, content: str, *, tool_name = None) -> str:
    """
    Wrap arbitrary text as a Harmony message for GPT-OSS.
    Notes:
        - Allows for an empty tool name.
    """
    if role in ['system', 'developer', 'user']:
        header = f"{role}<|message|>"
    elif role == 'assistant-cot':
        header = f"assistant<|channel|>analysis<|message|>"
    elif role == 'assistant-final':
        header = f"assistant<|channel|>final<|message|>"
    elif role == 'tool':
        tool_name = tool_name or ''
        header = f"functions.{tool_name} to=assistant<|channel|>commentary<|message|>"
    else:
        raise ValueError("Invalid role!")
    return f"<|start|>{header}{content}<|end|>"

def render_single_qwen3moe(role: str, content: str) -> str:
    """
    Wrap arbitrary text as a single Qwen3Moe message.
    Notes:
      - 'assistant-cot' is rendered as an assistant message with a <think>...</think> block only.
      - 'assistant-final' is rendered as an assistant message with visible content only.
      - 'tool' represents TOOL OUTPUT, which Qwen3 wraps as a user turn containing a <tool_response>...</tool_response> block.
        (The tool name is not used here; function *calls* are emitted from assistant messages via <tool_call>...</tool_call>.)
    """
    if role == 'system':
        return f"<|im_start|>system\n{content}<|im_end|>\n"
    elif role == 'user':
        return f"<|im_start|>user\n{content}<|im_end|>\n"
    elif role == 'assistant-cot':
        # Reasoning-only (no visible content)
        return f"<|im_start|>assistant\n<think>\n{content}\n</think>\n<|im_end|>\n"
    elif role == 'assistant-final':
        # return f"<|im_start|>assistant\n{content}<|im_end|>\n"
        return f"<|im_start|>assistant\n<think>\n</think>\n\n{content}<|im_end|>\n"
    elif role == 'tool':
        # Tool output fed back as a user message block
        return f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
    else:
        raise ValueError("Invalid role!")

def render_single_glm45(role: str, content: str, *, tool_name = None) -> str:
    """
    Render a single GLM-4.5 message segment.
    Notes:
      - Does NOT include the global '[gMASK]<sop>' prefix (add once per full prompt).
      - 'assistant-cot' emits only a <think> block; 'assistant-final' emits an empty think + visible content.
      - Tool OUTPUT is wrapped as an observation + <tool_response> block.
    """
    if role == 'system':
        return f"<|system|>\n{content}\n"
    elif role == 'user':
        return f"<|user|>\n{content}\n"
    elif role == 'assistant-cot':
        return f"<|assistant|>\n<think>\n{content}\n</think>\n"
    elif role == 'assistant-final':
        return f"<|assistant|>\n<think></think>\n{content}\n"
    elif role == 'tool':
        # Tool OUTPUT (results). GLM-4.6 groups consecutive tool responses under <|observation|>.
        # For a single message helper we always include the prefix.
        return f"<|observation|>\n<tool_response>\n{content}\n</tool_response>\n"
    else:
        raise ValueError("Invalid role!")

def render_single_message(model_architecture, role, content, tool_name = None) -> str:
    """
    Params:
        @model_architecture: One of several suppored model types, includes:
            - gptoss
        @role: One of several suppored roles, includes:
            - system
            - developer
            - user
            - assistant-cot
            - assistant-final
            - tool
        @content: The content of the message.
        @tool_name: The name of the tool to use.

    Example:
        messages = [
            ('user', 'Hello, how are you?', None),
            ('assistant-cot', 'I am good, thank you!', None),
            ('assistant-final', 'My favorite color is blue.', None)
            ('user', 'What is your favorite color?', None)
        ]
        render_prompt(messages)
    """
    if model_architecture == 'gptoss':
        res = render_single_gptoss(role, content, tool_name = tool_name)
    elif model_architecture == 'qwen3moe':
        res = render_single_qwen3moe(role, content)
    else:
        raise ValueError("Invalid model!")

    return res
