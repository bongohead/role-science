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

def render_single_olmo3(role: str, content: str) -> str:
    """
    Wrap arbitrary text as a single Olmo-3 Think-style message.

    Notes:
      - Uses the Olmo-3 ChatML-style envelope:
            <|im_start|>role\\n ... <|im_end|>
      - 'assistant-cot' is rendered as an assistant message whose content
        lives inside a <think>...</think> block only (no visible answer).
      - 'assistant-final' is rendered as an assistant message with an empty
        <think>...</think> stub followed by visible content, paralleling the
        Qwen3 helper.
      - 'tool' represents TOOL OUTPUT, which Olmo-3 encodes as an
        `environment` role; we map your 'tool' role to that.
    """
    if role == 'system':
        return f"<|im_start|>system\n{content}<|im_end|>\n"
    elif role == 'user':
        return f"<|im_start|>user\n{content}<|im_end|>\n"
    elif role == 'assistant-cot':
        return f"<|im_start|>assistant\n<think>{content}</think><|im_end|>\n"
    elif role == 'assistant-final':
        # return f"<|im_start|>assistant\n{content}<|im_end|>\n"
        return f"<|im_start|>assistant\n<think></think>{content}<|im_end|>\n"
        # return f"<|im_start|>assistant\n{content}<|im_end|>\n" ## OR try this - no think at all (instruct variant)
    elif role == 'tool': # Tool output / environment feedback Olmo-3 uses the `environment` role for this.
        return f"<|im_start|>environment\n{content}<|im_end|>\n"
    else:
        raise ValueError("Invalid role!")


def render_single_glm4moe(role: str, content: str) -> str:
    """
    Render a single GLM-4 message segment.

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
            - qwen3moe
            - glm4moe
        @role: One of several suppored roles, includes:
            - system
            - developer (only gpt-oss)
            - user
            - assistant-cot
            - assistant-final
            - tool
        @content: The content of the message.
        @tool_name: The name of the tool to use (only for gpt-oss)

    Example:
        render_single_message('gptoss', 'user', 'XXX', None)
    """
    if model_architecture == 'gptoss':
        res = render_single_gptoss(role, content, tool_name = tool_name)
    elif model_architecture in ['qwen3moe', 'qwen3vlmoe']:
        res = render_single_qwen3moe(role, content)
    elif model_architecture == 'glm4moe':
        res = render_single_glm4moe(role, content)
    elif model_architecture == 'olmo3':
        res = render_single_olmo3(role, content)

    else:
        raise ValueError("Invalid model!")

    return res

def load_chat_template(parent_dir, model_architecture) -> str:
    """
    Load a J2 chat template file for use in apply_chat_template; these are slightly modified versions of the real chat template. 
     No other changes are made other than those listed below.

    Description:
        These are created by taking the default tokenizer.chat_template and applying minimal model-specific changes listed below.
    
    Notes:
        - For Qwen3MoE, this:
            (1) prevents old <think></think> tags from being stripped.
        - For GLM4MoE, this does not; it's just the standard chat template.
        - For GPT-OSS, this:
            (1) removes the default system prompt;
            (2) has {"role": "system", "content", "..."} propagate to the system prompt instead of the developer prompt;
            (3) supports passing <think> directly (instead of a separate thinking key as in https://huggingface.co/spaces/huggingfacejs/chat-template-playground?modelId=openai%2Fgpt-oss-20b&example=hello-world).
    """
    with open(f'{parent_dir}/{model_architecture}.j2', 'r') as f:
        chat_template_str = f.read()

    return chat_template_str