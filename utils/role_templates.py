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
    Wrap arbitrary text as a single GLM4 message
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
        return f"<|observation|>\n<tool_response>\n{content}\n</tool_response>\n"
    else:
        raise ValueError("Invalid role!")
    
def render_single_apriel(role, content):
    """
    Wrap arbitrary text as a single Apriel-1.6 message
    """
    if role == 'system':
        return f"<|begin_system|>\n{content}\n"
    elif role == 'user':
        return f"<|begin_user|>\n{content}"
    elif role == 'cot':
        return f"\n<|begin_assistant|>\nHere are my reasoning steps:\n{content}\n<|end|>\n"
    elif role == 'assistant':
        return f"\n<|begin_assistant|>\n[BEGIN FINAL RESPONSE]\n{content}\n<|end|>"
    elif role == 'tool':
        return f"<|begin_tool_result|>\n{content}\n\n"
    else:
        raise ValueError("Invalid role!")
    
def render_single_olmo3(role: str, content: str) -> str:
    """
    Wrap arbitrary text as a single Olmo3 message
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
    Wrap arbitrary text as a single Qwen3 message
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
        return f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
    else:
        raise ValueError("Invalid role!")
        
def render_single_jamba_reasoning(role: str, content: str) -> str:
    """
    Wrap arbitrary text for the Jamba reasoning chat template.
    """
    if role == "system":
        return f"<|im_start|>system\n{content}<|im_end|>\n"
    elif role == "user":
        return f"<|im_start|>user\n{content}<|im_end|>\n"
    elif role == "assistant":
        return f"<|im_start|>assistant\n{content}<|im_end|>\n"
    elif role == "cot":
        return f"<|im_start|>assistant\n<think>\n{content}\n</think>\n\n<|im_end|>\n"
    elif role == "tool":
        return f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
    else:
        raise ValueError("Invalid role!")

def render_single_glm47_flash(role: str, content: str) -> str:
    """
    Wrap arbitrary text for GLM-4.7-Flash.
    """
    if role == "system":
        return f"<|system|>{content}"
    elif role == "user":
        return f"<|user|>{content}"
    elif role == "assistant":
        return f"<|assistant|><think></think>{content}"
    elif role == "cot":
        return f"<|assistant|><think>{content}</think>"
    elif role == "tool":
        return f"<|observation|><tool_response>{content}</tool_response>"
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
    elif model_prefix in ['apriel-1.6-15b-thinker']:
        res = render_single_apriel(role, content)
    elif model_prefix in ['olmo3-7b-think']:
        res = render_single_olmo3(role, content)
    elif model_prefix in ['qwen3-30b-a3b']:
        res = render_single_qwen3(role, content)
    elif model_prefix in ['jamba-reasoning']:
        res = render_single_jamba_reasoning(role, content)
    elif model_prefix in ['glm-4.7-flash']:
        res = render_single_glm47_flash(role, content)
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
    elif model_prefix in ['qwen3-30b-a3b']:
        return f"<|im_start|>assistant\n<think>\n{cot}\n</think>\n\n{assistant}<|im_end|>\n"
    elif model_prefix in ['glm-4.6v-flash']:
        return f"<|assistant|>\n<think>{cot}</think>\n{assistant}"
    elif model_prefix in ['apriel-1.6-15b-thinker']:
        return f"\n<|begin_assistant|>\nHere are my reasoning steps:\n{cot}\n[BEGIN FINAL RESPONSE]\n{assistant}\n<|end|>\n"
    elif model_prefix in ['olmo3-7b-think']:
        return f"<|im_start|>assistant\n<think>{cot}</think>{assistant}<|im_end|>\n"
    elif model_prefix in ['jamba-reasoning']:
        return f"<|im_start|>assistant\n<think>\n{cot}\n</think>\n\n{assistant}<|im_end|>\n"
    elif model_prefix in ['glm-4.7-flash']:
        return f"<|assistant|><think>{cot}</think>{assistant}"        
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
    """
    if model_prefix in ['gptoss-20b', 'gptoss-120b']:
        instruct_format = 'gptoss'
    elif model_prefix in ['nemotron-3-nano']:
        instruct_format = 'nemotron'
    elif model_prefix in ['glm-4.6v-flash']:
        instruct_format = 'glm4'
    elif model_prefix in ['apriel-1.6-15b-thinker']:
        instruct_format = 'apriel'
    elif model_prefix in ['olmo3-7b-think']:
        instruct_format = 'olmo3'
    elif model_prefix in ['qwen3-30b-a3b']: 
        instruct_format = 'qwen3'
    elif model_prefix in ['jamba-reasoning']: 
        instruct_format = 'jamba'
    elif model_prefix in ['glm-4.7-flash']:
        instruct_format = 'glm47'
    else:
        raise ValueError(f"Model prefix {model_prefix} not supported")

    with open(f'{parent_dir}/{instruct_format}.j2', 'r') as f:
        chat_template_str = f.read()

    return chat_template_str
    
def fold_cot_into_final(convs):
    """
    Fold CoT into the following assistant message as a <think></think> tag.
    """
    result = []
    for conv in convs:
        new_conv = []
        it = iter(enumerate(conv))
        for i, msg in it:
            if msg['role'] == 'cot':
                if i + 1 >= len(conv) or conv[i + 1]['role'] != 'assistant':
                    raise ValueError("cot must be followed by assistant")
                _, next_msg = next(it)
                new_conv.append({'role': 'assistant', 'content': f"<think>{msg['content']}</think>{next_msg['content']}"})
            else:
                new_conv.append({'role': msg['role'], 'content': msg['content']})
        result.append(new_conv)
    return result
