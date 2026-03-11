#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Initialization Utilities.

Provides factory functions for initializing language models including
Azure OpenAI, local Hugging Face models, and PEFT/LoRA fine-tuned models.

@author: Khushboo Thaker, Yony Bresler, Khalid Eidoo
@license: Crater Labs (C)
"""

import logging
from typing import Optional, Any, Tuple
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

logger = logging.getLogger(__name__)


class LLMConnection:
    """Lightweight container for LLM API connection settings."""

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0, max_tokens: int = 5000):
        """Store LLM API connection settings.

        Args:
            base_url: API endpoint URL (Azure or OpenAI-compatible).
            api_key: Authentication key for the API.
            model: Deployment or model name to use in requests.
            temperature: Sampling temperature (0 = deterministic).
            max_tokens: Maximum tokens to generate per request.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens


def initialize_azure_openai_llm(
    connection_settings: dict,
    deployment_names: dict,
    llm_parameters: dict,
    llm_params_override: dict = None
):
    """Initialize and return an LLM connection instance.

    Args:
        connection_settings: Dictionary with 'endpoint', 'api_key', 'api_version'.
        deployment_names: Dictionary with 'chat_deployment_name'.
        llm_parameters: Dictionary with default LLM parameters like 'temperature', 'max_tokens', 'top_p'.
        llm_params_override: A dictionary of parameters to override the defaults.

    Returns:
        LLMConnection: An initialized LLM connection instance.
    """
    logger.info("Initializing OpenAI-compatible LLM connection...")

    final_llm_params = llm_parameters.copy()
    if llm_params_override:
        final_llm_params.update(llm_params_override)

    logger.debug(f"Using LLM parameters: {final_llm_params}")
    llm = LLMConnection(
        base_url=connection_settings['endpoint'],
        api_key=connection_settings['api_key'],
        model=deployment_names['chat_deployment_name'],
        temperature=final_llm_params.get('temperature', 0),
        max_tokens=final_llm_params.get('max_tokens', 5000),
    )

    logger.info("LLM connection initialized.")
    return llm




def _detect_attn_implementation() -> str:
    """Return the best available attention backend.

    Returns:
        "flash_attention_2" if flash-attn is installed, otherwise "sdpa"
        (PyTorch native scaled-dot-product attention, always available).
    """
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def create_bnb_config(quantization: Optional[int]) -> Optional[BitsAndBytesConfig]:
    """Create a BitsAndBytesConfig for the specified quantization level.

    Args:
        quantization: Bit width for quantization. Use 4 for 4-bit NF4
            (recommended), 8 for 8-bit, or None to skip quantization.

    Returns:
        BitsAndBytesConfig for the requested precision, or None if
        quantization is None.
    """
    if quantization == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if quantization == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    return None


def initialize_local_llm(model_id: str, quantization: Optional[int], cache_dir: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load a Hugging Face model for local inference.

    Args:
        model_id: HuggingFace model identifier.
        quantization: Bit quantization (4 or 8), or None for full precision.
        cache_dir: Directory for model cache.

    Returns:
        Tuple of (tokenizer, model). Note: order is (tokenizer, model) — the
        reverse of ``initialize_local_peft``, which returns (model, tokenizer).
    """
    logger.info(f"Loading local model: {model_id}")

    bnb_config = create_bnb_config(quantization)

    # If no quantization, log info for modern GPUs
    if bnb_config is None and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        logger.info("No quantization specified. Loading model in bfloat16 for speed.")

    attn_impl = _detect_attn_implementation()
    logger.info("Using attention implementation: %s", attn_impl)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        cache_dir=cache_dir,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

    # Set model to evaluation mode
    model.eval()
    logger.info("Model loaded and set to evaluation mode.")

    return tokenizer, model

def initialize_local_peft(
    model_id: str,
    adapter_path: str,
    cache_dir: str = None,
    quantization: Optional[int] = None,
) -> Tuple[Any, Any]:
    """Load a base model with PEFT adapter merged for fine-tuned inference.

    Args:
        model_id: Base model HuggingFace identifier.
        adapter_path: Path to the PEFT/LoRA adapter weights.
        cache_dir: Directory for model cache.
        quantization: Bit quantization (4 or 8), or None.

    Returns:
        Tuple of (model, tokenizer) with adapter merged. Note: order is
        (model, tokenizer) — the reverse of ``initialize_local_llm``, which
        returns (tokenizer, model).
    """
    logger.info(f"Loading base model: {model_id} for PEFT inference...")
    
    bnb_config = create_bnb_config(quantization)

    attn_impl = _detect_attn_implementation()
    logger.info("Using attention implementation: %s", attn_impl)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        cache_dir=cache_dir,
        # trust_remote_code=True allows the model repo to execute custom Python code during
        # loading (e.g. custom modeling files). Only load models from sources you trust.
        trust_remote_code=True, # Required for Qwen models [1]
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
    )
    base_model.config.use_cache = True

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'
        if tokenizer.pad_token == '[PAD]' and '[PAD]' not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.warning("Added [PAD] token to tokenizer for PEFT inference.")

    logger.info(f"Loading PEFT adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    logger.info("Merging PEFT adapter weights into base model...")
    model = model.merge_and_unload()


    model.eval() # Set model to evaluation mode

    logger.info("Fine-tuned PEFT model loaded and merged.")
    return model, tokenizer