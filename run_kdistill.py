#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Distillation Orchestrator.

Main script for running LoRA-based knowledge distillation experiments.
Manages experiment configurations, model training, and checkpoint resumption.

@author: Khushboo Thaker, Yony Bresler, Khalid Eidoo
@license: Crater Labs (C)
"""

import argparse
import logging
import os
from typing import Dict, Any, List

import torch
from accelerate import Accelerator

from src.distillation_utils import run_knowledge_distillation
from src.config_loader import ConfigLoader, ConfigurationError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _make_run_id(model_name: str, config: Dict[str, Any], max_len: int = 128) -> str:
    """Build a filesystem-safe run identifier from experiment config.

    Args:
        model_name: HuggingFace model identifier (e.g. 'Qwen/Qwen3-4B').
        config: Experiment config dict with keys 'lora_r', 'lora_alpha',
            'learning_rate', 'epochs', and optionally 'quantization'.
        max_len: Maximum character length of the returned string.

    Returns:
        Filesystem-safe run identifier string (alphanumeric, underscores,
        hyphens).
    """
    short_name = model_name.split('/')[-1].replace('-', '')
    lr = config.get('learning_rate', '')
    lr_str = f"LR{str(lr).replace('.', 'p').replace('e-0', 'e')}"
    quant = config.get('quantization')
    parts = [
        f"KDS_QPPro_{short_name}",
        f"R{config.get('lora_r', 'N')}A{config.get('lora_alpha', 'N')}",
        lr_str,
        f"E{config.get('epochs', 'N')}",
        f"Q{quant}" if quant else "",
    ]
    run_id = "_".join(p for p in parts if p)
    run_id = "".join(c for c in run_id if c.isalnum() or c in "_-").replace("__", "_")
    return run_id[:max_len]


if __name__ == "__main__":
    # PyTorch memory optimization for large GPUs
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="Orchestrate multiple LoRA Distillation experiments for text-to-SQL task.")
    parser.add_argument("--config-file", type=str, default="config.ini",
                        help="Path to the main configuration file (e.g., config.ini).")
    parser.add_argument("--mode", type=str, choices=["auto", "ddp"], default="ddp",
                    help="Select the Distillation mode: 'auto' for device_map or 'ddp' for distributed data parallel.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="HuggingFace dataset to use (e.g., 'craterlabs/struct-sql-data'). "
                             "When provided, loads data directly from HuggingFace and bakes schema context. "
                             "When omitted, falls back to local JSON files from kd_data_dir in config.")
    args = parser.parse_args()

    # Each dictionary represents a single Distillation experiment run.
    # The output directory name is generated automatically from the config keys.
    # Supported keys:
    #   model_name   (str)  – HuggingFace model identifier for the base model.
    #   resume       (bool) – Resume from last checkpoint if True; start fresh if False.
    #   lora_r       (int)  – LoRA rank.
    #   lora_alpha   (int)  – LoRA alpha scaling factor.
    #   learning_rate(float)– Optimizer learning rate.
    #   batch_size   (int)  – Per-device training batch size.
    #   epochs       (int)  – Max training epochs.
    #   quantization (int)  – Bit width (4 or 8), or omit for full precision.
    #   max_steps    (int)  – Cap training steps (positive int), or -1 to use epochs.
    #   description  (str)  – Optional human-readable label shown in log output.
    finetuning_experiment_configs: List[Dict[str, Any]] = [

        {
            "model_name": "Qwen/Qwen3-4B-Instruct-2507", # Base model to fine-tune
            "resume": True, # Set to True if you want to resume a previous finetuning run
            "lora_r": 64,
            "lora_alpha": 128,
            "learning_rate": 1e-4,
            "batch_size": 6,
            "epochs": 10,
            "quantization": 4,
            "max_steps": -1,  # Set to a positive int (e.g. 5) to limit training steps for debugging
        },
    ]

    # Load general paths from config once
    try:
        config_loader = ConfigLoader(args.config_file)
        general_paths = config_loader.get_general_paths()
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}. Please check your '{args.config_file}' file.")
        exit(1)
    accelerator = Accelerator()


    for i, exp_config in enumerate(finetuning_experiment_configs):
        


        if accelerator.is_main_process:

            logger.info(f"\n--- Starting Distillation Experiment {i+1}/{len(finetuning_experiment_configs)}: {exp_config.get('description', 'Unnamed Experiment')} ---")

        # Extract parameters for the current experiment
        model_name = exp_config["model_name"]
        resume_run = exp_config.get("resume", False)
        description = exp_config.get("description", "Distillation Experiment")

        output_dir_prefix = _make_run_id(model_name, exp_config)
        # Determine the output directory for this specific run
        run_output_dir = os.path.join(general_paths['base_output_dir'], output_dir_prefix)
        if accelerator.is_main_process:
            os.makedirs(run_output_dir, exist_ok=True)
            logger.info(f"Output directory for this run: {run_output_dir}")


        try:
            logger.info("Running LoRA Distillation step...")
            finetuning_results = run_knowledge_distillation(
                accelerator=accelerator,
                model_name=model_name,
                output_dir=run_output_dir,
                config_file=args.config_file,
                lora_r=exp_config.get('lora_r'),
                lora_alpha=exp_config.get('lora_alpha'),
                learning_rate=exp_config.get('learning_rate'),
                batch_size=exp_config.get('batch_size'),
                epochs=exp_config.get('epochs'),
                quantization=exp_config.get('quantization'),
                max_steps=exp_config.get('max_steps'),
                dataset_name=args.dataset,
                resume=resume_run,
            )
            
            logger.info("Distillation complete.")

        except Exception as e:
            logger.error(f"Distillation experiment failed for config: {description}. Error: {e}", exc_info=True)
            continue


    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info(f"--- Finished all {len(finetuning_experiment_configs)} Distillation Experiment(s) ---")
        logger.info("\n=== All Distillation Experiments Complete ===")