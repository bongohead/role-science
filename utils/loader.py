"""
Model loaders
"""
import torch
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers import AutoTokenizer, AutoModelForCausalLM, Glm4vForConditionalGeneration, AutoModelForImageTextToText
from packaging import version
import transformers
import importlib
TRANSFORMERS_VERSION = version.parse(transformers.__version__).major

def get_supported_model_metadata(model_prefix):
    """
    Get a list of supported models
    
    Params:
        @model_prefix: The model prefix string - must be one of below supported models.

    Returns:
        A tuple with:
        - HF model id
        - model arch (used for loading custom forward pass)
        - attn implementation
        - whether to use the HF default implementation
        - # hidden layers
    """
    models = {
        'gptoss-20b': ('openai/gpt-oss-20b', 'gptoss', 'kernels-community/vllm-flash-attn3', True, 24),
        'gptoss-120b': ('openai/gpt-oss-120b', 'gptoss', 'kernels-community/vllm-flash-attn3', True, 36),
        'nemotron-3-nano': ('nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16', 'nemotron3', None, False, 52),
        'qwen3-30b-a3b': ('Qwen/Qwen3-30B-A3B-Thinking-2507', 'qwen3moe', None, True, 48),
        'jamba-reasoning': ('ai21labs/AI21-Jamba-Reasoning- 3B',  'jamba', None, True, 28),
        'apriel-1.6-15b-thinker': ('ServiceNow-AI/Apriel-1.6-15b-Thinker', 'apriel', None, True, 48),
        'olmo3-7b-think': ('allenai/Olmo-3-7B-Think', 'olmo3', None, True, 32),
        'glm-4.6v-flash': ('zai-org/GLM-4.6V-Flash', 'glm46v', None, True, 40),
        'glm-4.7-flash': ('zai-org/GLM-4.7-Flash', 'glm4moelite', None, True, 46)
    }
    if model_prefix not in models:
        raise ValueError(f"Model index {model_prefix} not recognized. Available models: {list(models.keys())}")
    
    if TRANSFORMERS_VERSION != 5 and model_prefix in ['glm-4.6v-flash', 'glm-4.7-flash']:
        raise ValueError(f"GLM-4.6V and GLM-4.7 require transformers v5+. Current version: {transformers.__version__}")

    if TRANSFORMERS_VERSION != 4 and model_prefix in ['nemotron-3-nano', 'qwen3-30b-a3b', 'apriel-1.6-15b-thinker']:
        raise ValueError(f"Model {model_prefix} requires transformers v4.x. Current version: {transformers.__version__}")

    return models[model_prefix]

def load_model_and_tokenizer(model_prefix, device):
    """
    Load the model and tokenizer from HF, or from file if already downloaded.

    Params:
        @model_prefix: The model prefix string - must be one of supported models.
        @device: The device to load the model onto.
    
    Returns:
        A tuple with:
        - The tokenizer object
        - The model object
        - model architecture
        - # hidden layers
    """
    model_id, model_architecture, model_attn, model_use_hf, model_n_layers = get_supported_model_metadata(model_prefix)

    cache_dir = '/workspace/hf'
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = cache_dir, add_eos_token = False, add_bos_token = False, padding_side = 'left', trust_remote_code = True)
    load_params = {'cache_dir': cache_dir, 'dtype': 'auto', 'trust_remote_code': not model_use_hf, 'device_map': None, 'attn_implementation': model_attn}    
    if model_architecture == 'glm46v':
        model = Glm4vForConditionalGeneration.from_pretrained(model_id, **load_params).to(device).eval()
    elif model_architecture == 'apriel':
        model = AutoModelForImageTextToText.from_pretrained(model_id, **load_params).to(device).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_params).to(device).eval()

    # Check loaded in correct format (MXFP4 / FA3)
    if model_architecture == 'gptoss':
        print(f"Expert precision: {model.model.layers[0].mlp.experts.down_proj.dtype}")
        print(f"Attention implementation: {model.model.config._attn_implementation}")

    # Set pad tokens (Nemotron-3-Nano missing by default)
    if tokenizer.pad_token is None:
        print('Setting pad token automatically')
        tokenizer.pad_token = tokenizer.eos_token

    # In transformers v5, avoid non-deterministic MoE implementations
    if TRANSFORMERS_VERSION == 5 and hasattr(model, 'set_experts_implementation'):
        model.set_experts_implementation('eager')

    return tokenizer, model, model_architecture, model_n_layers

def load_custom_forward_pass(model_architecture, model = None, tokenizer = None):
    """
    Load the custom forward pass function for a given model architecture 
    
    Description:
        This loads a custom forward pass from utils.pretrained_models and verifies that it replicates the real model's forward pass exactly.
        The custom forward pass is used to extract hidden states per layer both post-attention and post-MLP.
        Also extracts a variety of MoE-related routing metadata.
        (For more basic purposes, this can be replaced by simpler hooks if desired).

    Params:
        @model_architecture: One of supported architectures returned by get_supported_model_metadata.
        @model: (optional) The standard HF model object; used for validation against the custom forward pass if passed with `model`.
        @tokenizer: (optional) The tokenizer object; used for validation against the custom forward pass if passed with `tokenizer`.
    """
    model_module = importlib.import_module(f"utils.pretrained_models.{model_architecture}")
    run_forward_with_hs = getattr(model_module, f"run_{model_architecture}_return_topk")

    @torch.no_grad()
    def _verify_custom_forward_pass(model, pad_token_id = tokenizer.pad_token_id):
        inputs = tokenizer(
            ['Hi! I am a dog and I like to bark', 'Vegetables are good for'],
            return_tensors = 'pt', padding = 'max_length', truncation = True, max_length = 640
        ).to(model.device)

        original_results = model(**inputs, use_cache = False)
        custom_results = run_forward_with_hs(model, inputs['input_ids'], inputs['attention_mask'], return_hidden_states = True)
        
        assert torch.equal(original_results.logits, custom_results['logits']), 'Error in custom forward'
        
        loss = ForCausalLMLoss(
            custom_results['logits'], torch.where(inputs['input_ids'] == pad_token_id, torch.tensor(-100), inputs['input_ids']), custom_results['logits'].size(-1)
        ).detach().cpu().item()

        print(f"LM loss: {loss}")
        print(f"Hidden states layers (pre-mlp | post-layer): {len(custom_results['all_pre_mlp_hidden_states'])} | {len(custom_results['all_hidden_states'])}")
        print(f"Hidden state size (pre-mlp | post-layer): {(custom_results['all_pre_mlp_hidden_states'][0].shape)} | {(custom_results['all_hidden_states'][0].shape)}")
        print('Verified custom forward pass successfully matches original model output!')

    if model is not None and tokenizer is not None:
        _verify_custom_forward_pass(model, tokenizer.pad_token_id)

    return run_forward_with_hs

