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
    elif role == 'cot':
        header = f"assistant<|channel|>analysis<|message|>"
    elif role == 'assistant':
        header = f"assistant<|channel|>final<|message|>"
    elif role == 'tool':
        tool_name = tool_name or ''
        header = f"functions.{tool_name} to=assistant<|channel|>commentary<|message|>"
    else:
        raise ValueError("Invalid role!")
    return f"<|start|>{header}{content}<|end|>"

def render_single_nemotron3(role: str, content: str) -> str:
    """
    Wrap arbitrary text as a single Nemotron3 message
    """
    if role == 'system':
        return f"<|im_start|>system\n{content}<|im_end|>\n"
    elif role == 'user':
        return f"<|im_start|>user\n{content}<|im_end|>\n"
    elif role == 'cot':
        return f"<|im_start|>assistant\n<think>\n{content}\n</think>\n<|im_end|>\n"
    elif role == 'assistant':
        return f"<|im_start|>assistant\n<think></think>{content}<|im_end|>\n"
    elif role == 'tool':
        return f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response>\n<|im_end|>\n"
    else:
        raise ValueError("Invalid role!")
    
def render_single_glm4(role: str, content: str) -> str:
    """
    Render a single GLM-4 message segment.

    Notes:
      - Does NOT include the global '[gMASK]<sop>' prefix (add once per full prompt).
      - 'cot' emits only a <think> block; 'assistant' emits an empty think + visible content.
      - Tool OUTPUT is wrapped as an observation + <tool_response> block.
    """
    if role == 'system':
        return f"<|system|>\n{content}"
    elif role == 'user':
        return f"<|user|>\n{content}"
    elif role == 'cot':
        return f"<|assistant|>\n<think>{content}</think>\n"
    elif role == 'assistant':
        return f"<|assistant|>\n<think></think>\n{content}"
    elif role == 'tool':
        # Tool OUTPUT (results). GLM-4.6 groups consecutive tool responses under <|observation|>.
        # For a single message helper we always include the prefix.
        return f"<|observation|>\n<tool_response>\n{content}\n</tool_response>"
    else:
        raise ValueError("Invalid role!")
    
def render_single_apriel(role, content):
    """
    Render for Apriel format
    """
    if role == 'system':
        return f"<|begin_system|>\n{content}\n"
    elif role == 'user':
        return f"<|begin_user|>\n{content}\n"
    elif role == 'cot':
        return f"<|begin_assistant|>\nHere are my reasoning steps:\n{content}\n<|end|>"
    elif role == 'assistant':
        return f"<|begin_assistant|>\n[BEGIN FINAL RESPONSE]\n{content}\n<|end|>"
    elif role == 'tool':
        return f"<|begin_tool_result|>\n{content}\n\n\n"
    
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
    elif role == 'cot':
        return f"<|im_start|>assistant\n<think>{content}</think><|im_end|>\n"
    elif role == 'assistant':
        return f"<|im_start|>assistant\n{content}<|im_end|>\n"
    elif role == 'tool':
        return f"<|im_start|>environment\n{content}<|im_end|>\n"
    else:
        raise ValueError("Invalid role!")

def render_single_qwen3(role: str, content: str) -> str:
    """
    Wrap arbitrary text as a single Qwen3Moe message.
    Notes:
      - 'cot' is rendered as an assistant message with a <think>...</think> block only.
      - 'assistant' is rendered as an assistant message with visible content only.
      - 'tool' represents TOOL OUTPUT, which Qwen3 wraps as a user turn containing a <tool_response>...</tool_response> block.
        (The tool name is not used here; function *calls* are emitted from assistant messages via <tool_call>...</tool_call>.)
    """
    if role == 'system':
        return f"<|im_start|>system\n{content}<|im_end|>\n"
    elif role == 'user':
        return f"<|im_start|>user\n{content}<|im_end|>\n"
    elif role == 'cot':
        return f"<|im_start|>assistant\n<think>\n{content}\n</think>\n\n<|im_end|>\n"
    elif role == 'assistant':
        return f"<|im_start|>assistant\n{content}<|im_end|>\n"
        # return f"<|im_start|>assistant\n<think>\n\n</think>\n\n{content}<|im_end|>\n"
    elif role == 'tool':
        # Tool output fed back as a user message block
        return f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
    else:
        raise ValueError("Invalid role!")
    
def render_single_rnj1(role, content):
    """
    Render for RNJ-1 format
    """
    if role == 'system':
        return f"<|start_header_id|>system<|end_header_id|>\n{content}<|eot_id|>"
    elif role == 'user':
        return f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>"
    elif role == 'assistant':
        return f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>"
    elif role == 'tool':
        return f"<|start_header_id|>user<|end_header_id|>\n<tool_response>\n{content}\n</tool_response><|eot_id|>"

def render_single_lfm25(role: str, content: str) -> str:
    """
    Render for LiquidAI LFM2.5 models
    """
    if role == "system":
        return f"<|im_start|>system\n{content}<|im_end|>\n"
    elif role == "user":
        return f"<|im_start|>user\n{content}<|im_end|>\n"
    elif role == "assistant":
        return f"<|im_start|>assistant\n{content}<|im_end|>\n"
    # elif role == "cot":
    #     return f"<|im_start|>assistant\n<think>\n{content}\n</think><|im_end|>\n"
    elif role == "tool":
        return f"<|im_start|>tool\n{content}<|im_end|>\n"
    else:
        raise ValueError("Invalid role!")

def render_single_message(model_prefix, role, content, tool_name = None) -> str:
    """
    Params:
        @model_architecture: The model prefix; see code for supported models
        @role: One of several suppored roles, includes:
            - system
            - developer (only gpt-oss)
            - user
            - cot
            - final
            - tool
        @content: The content of the message.
        @tool_name: The name of the tool to use (only for gpt-oss)

    Example:
        render_single_message('gptoss', 'user', None)
    """
    if model_prefix in ['gptoss-20b', 'gptoss-120b']:
        res = render_single_gptoss(role, content, tool_name = tool_name)
    elif model_prefix in ['nemotron-3-nano']:
        res = render_single_nemotron3(role, content)
    elif model_prefix in ['glm-4.6v-flash']:
        res = render_single_glm4(role, content)
    elif model_prefix in ['apriel-1.6b-thinker']:
        res = render_single_apriel(role, content)
    elif model_prefix in ['olmo3-7b-think']:
        res = render_single_olmo3(role, content)
    elif model_prefix in ['lfm2.5-1.2b']:
        res = render_single_lfm25(role, content)
    else:
        raise ValueError("Invalid model!")

    return res

def render_mixed_cot(model_prefix, cot, assistant) -> str:
    """
    Renders a mixed cot + assistant message. Only valid for reasoning models.

    Params:
        @model_architecture: One of several suppored model types, includes.
        @cot: The assistant-cot text
        @assistant: The assistant-final text

    Example:
        render_mixed_cot('gptoss', 'The user says..', 'The user says')
    """
    if model_prefix in ['gptoss-20b', 'gptoss-120b']:
        return f"<|start|>assistant<|channel|>analysis<|message|>{cot}<|end|><|start|>assistant<|channel|>final<|message|>{assistant}<|end|>"
    elif model_prefix in ['nemotron-3-nano']:
        return f"<|im_start|>assistant\n<think>\n{cot}\n</think>\n{assistant}<|im_end|>\n"
    elif model_prefix in ['glm-4.6v-flash']:
        return f"<|assistant|>\n<think>{cot}</think>\n{assistant}"
    elif model_prefix in ['apriel-1.6b-thinker']:
        return f"<|begin_assistant|>\nHere are my reasoning steps:\n{cot}\n[BEGIN FINAL RESPONSE]\n{assistant}\n<|end|>\n"
    elif model_prefix in ['olmo3-7b-think']:
        return f"<|im_start|>assistant\n<think>{cot}</think>{assistant}<|im_end|>\n"
    else:
        raise ValueError("Invalid model!")

def load_chat_template(parent_dir, model_prefix) -> str:
    """
    Load a J2 chat template file for use in apply_chat_template; these are slightly modified versions of the real chat template. 
     No other changes are made other than those listed below.

    Description:
        These are created by taking the default tokenizer.chat_template and applying minimal model-specific changes listed below. 
        These are modified for consistency such that for all models:
        - There's no default system prompt
        - No BOS token by default
        - Previous thoughts aren't stripped (for reasoning models), and follow the exact same format as "current thought"
        - Previous thoughts can be passed into assistant roles via `{"role": "assistant", "content": "<think>xx</think>yyy"}` -> sometimes this requires 
          the CoT to be reformatted (e.g., linebreaks) to match the proper "current thought" format
          
    
    Notes:
        - For GPT-OSS, this:
            1. Removes the default system prompt
            2. Has {"role": "system", "content", "..."} propagate to the system prompt instead of the developer prompt;
            3. Supports passing <think> directly (instead of a separate thinking key as in https://huggingface.co/spaces/huggingfacejs/chat-template-playground?modelId=openai%2Fgpt-oss-20b&example=hello-world).
        - For Nemotron-3-Nano:
            1. Preserves old thinks
            2. Prevents reformatting for old thinks
            3. Allows passing in <think> into the assistant cot -> reformats to <think>\n...\n</think>

        - For GLM4, this does not; it's just the standard chat template.
            (1) removes the [gMASK]<sop> prefix
        - For LFM.5-1.2b:
            (1) Retains old thinks (not a reasoning model, so irrelevant)
            (2) Removes BOS token
        - For Olmo3, this:
            (1) does nothing, it's just the base template
        - For Apriel, this:
            (1) removes the default system prompt
        - For Qwen, this:
            (1) prevents old <think></think> tags from being stripped.

    """
    if model_prefix in ['gptoss-20b', 'gptoss-120b']:
        instruct_format = 'gptoss'
    elif model_prefix in ['nemotron-3-nano']:
        instruct_format = 'nemotron'
    elif model_prefix in ['glm-46v-flash']:
        instruct_format = 'glm4'
    elif model_prefix in ['apriel-1.6b-thinker']:
        instruct_format = 'apriel'
    elif model_prefix in ['olmo3-7b-think']:
        instruct_format = 'olmo3'
    elif model_prefix in ['lfm2.5-1.2b']:
        instruct_format = 'lfm2'

    else:
        raise ValueError(f"Model prefix {model_prefix} not supported")

    with open(f'{parent_dir}/{instruct_format}.j2', 'r') as f:
        chat_template_str = f.read()

    return chat_template_str
    