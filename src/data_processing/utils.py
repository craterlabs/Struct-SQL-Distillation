#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for data curation.

Small, stateless helpers for prompt loading, target count calculation,
and inference result validation.

@author: Khushboo Thaker, Yony Bresler, Khalid Eidoo
@license: Crater Labs (C)
"""

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts/ directory.

    Args:
        template_name: Filename stem (without .txt) inside the prompts/ directory.

    Returns:
        Full text content of the template file.

    Raises:
        FileNotFoundError: If no file exists at prompts/<template_name>.txt.
    """
    prompt_path = Path("prompts") / f"{template_name}.txt"
    try:
        return prompt_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        logger.error("Prompt template file not found: %s", prompt_path)
        raise FileNotFoundError(f"Could not find {prompt_path}")


def is_successful_inference(result: dict) -> bool:
    """Return True only if an inference result has both successful generation and execution.

    Args:
        result: A per-sample result dict, typically loaded from the individual results cache.

    Returns:
        True when generation_status == "Success" AND execution_match is exactly True.
    """
    return result.get("generation_status") == "Success" and result.get("execution_match") is True


def get_target_counts(total_size: int, distribution: Dict[str, float]) -> Dict[str, int]:
    """Compute integer target counts for each category, summing exactly to total_size.

    Rounding differences are distributed one-at-a-time across categories to ensure
    the returned counts always add up to total_size.

    Args:
        total_size: Total number of samples to allocate.
        distribution: Mapping of category name to fractional proportion (should sum to 1.0).

    Returns:
        Mapping of category name to integer target count.
    """
    targets = {cat: round(prop * total_size) for cat, prop in distribution.items()}
    diff = total_size - sum(targets.values())
    # Distribute the rounding difference one-at-a-time across categories
    if diff > 0:
        sorted_cats = sorted(targets, key=lambda c: targets[c])
        for i in range(diff):
            targets[sorted_cats[i % len(sorted_cats)]] += 1
    elif diff < 0:
        sorted_cats = sorted(targets, key=lambda c: targets[c], reverse=True)
        for i in range(-diff):
            targets[sorted_cats[i % len(sorted_cats)]] -= 1
    return targets
