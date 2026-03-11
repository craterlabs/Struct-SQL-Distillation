#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Distillation Core Utilities.

Core training logic for LoRA-based knowledge distillation using SFTTrainer.
Includes model setup, training configuration, evaluation, and checkpoint management.

@author: Khushboo Thaker, Yony Bresler, Khalid Eidoo
@license: Crater Labs (C)
"""

import gc
import logging
import os
import shutil
from typing import Dict, Any, Optional

import torch
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, EarlyStoppingCallback, TrainerCallback
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator

from src.distillation_data_utils import prepare_static_distillation_dataset
from src.database_manager import DatabaseManager, DatabaseManagerError, CompareResult
from src.bird_schema_loader import BirdSchemaLoader, BirdSchemaLoaderError
from src.config_loader import ConfigLoader, ConfigurationError
from src.llm_initializer import create_bnb_config, _detect_attn_implementation

logger = logging.getLogger(__name__)

# Default HuggingFace dataset for knowledge distillation
DEFAULT_HF_DATASET = "craterlabs/struct-sql-data"

# Estimated wall-clock seconds per gradient step on an L4/H200 GPU for 15k-token sequences.
# Used only in the pre-flight "silence period" warning — not a hard constraint.
_PREFLIGHT_SECS_PER_STEP: int = 70

def _resolve_dataset_kwargs(
    dataset_name: Optional[str],
    curated_data_dir: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Return the kwargs dict for _load_or_prepare_datasets based on data availability.

    Priority:
        1. Explicit HuggingFace dataset name.
        2. All three local JSON files present.
        3. Fallback to the default HuggingFace dataset.

    Args:
        dataset_name: HuggingFace dataset identifier, or None to use local files.
        curated_data_dir: Directory containing train.json / val_*.json.
        output_dir: Destination directory for the static-baked artefacts.

    Returns:
        kwargs dict suitable for passing directly to _load_or_prepare_datasets.
    """
    baked_output = os.path.join(output_dir, "static_baked")
    if dataset_name:
        return {"output_dir": baked_output, "dataset_name": dataset_name}
    local_files = [
        os.path.join(curated_data_dir, n)
        for n in ("train.json", "val_in_domain.json", "val_out_of_domain.json")
    ]
    if all(os.path.exists(p) for p in local_files):
        logger.info("Using local JSON files from %s", curated_data_dir)
        return {"output_dir": baked_output, "raw_data_dir": curated_data_dir}
    missing = [os.path.basename(p) for p in local_files if not os.path.exists(p)]
    logger.warning(
        "Local JSON files not found: %s. Downloading from HuggingFace (%s) instead.",
        missing, DEFAULT_HF_DATASET,
    )
    return {"output_dir": baked_output, "dataset_name": DEFAULT_HF_DATASET}


def _load_or_prepare_datasets(kwargs: Dict[str, Any], output_dir: str, db_manager: DatabaseManager) -> Dict[str, Any]:
    """Prepare, concatenate, and save static distillation datasets to disk.

    Calls ``prepare_static_distillation_dataset`` with the provided kwargs, then
    combines the in-domain and out-of-domain validation splits into a single eval
    set and persists all splits under ``output_dir`` using HuggingFace's
    ``save_to_disk`` format for fast loading during training.

    Args:
        kwargs: Keyword arguments forwarded to ``prepare_static_distillation_dataset``
            (must include ``output_dir`` and either ``dataset_name`` or ``raw_data_dir``).
        output_dir: Root directory under which split subdirectories are written
            (``train_raw/``, ``all_raw_eval/``, ``val_in_domain/``, ``val_out_of_domain/``).
        db_manager: DatabaseManager used for schema-context baking.

    Returns:
        Dict with keys ``'train'`` and ``'val'``, each a HuggingFace ``Dataset``.
    """
    logger.info(f"Preparing static distillation datasets...")
    static_datasets = prepare_static_distillation_dataset(db_manager=db_manager, **kwargs)

    train_dataset = static_datasets["train"]
    val_in_domain_dataset = static_datasets["val_in_domain"]
    val_out_of_domain_dataset = static_datasets["val_out_of_domain"]

    logger.info("Combining in-domain and out-of-domain validation sets...")
    val_dataset = concatenate_datasets([val_in_domain_dataset, val_out_of_domain_dataset])

    train_dataset.save_to_disk(os.path.join(output_dir, "train_raw"))
    val_dataset.save_to_disk(os.path.join(output_dir, "all_raw_eval"))
    val_in_domain_dataset.save_to_disk(os.path.join(output_dir, "val_in_domain"))
    val_out_of_domain_dataset.save_to_disk(os.path.join(output_dir, "val_out_of_domain"))
    
    return {"train": train_dataset, "val": val_dataset}


def run_pre_flight_checks(trainer: SFTTrainer, tokenizer: AutoTokenizer, sft_config: SFTConfig) -> None:
    """Verify EOS token masking, batch size settings, and display training expectations.

    Args:
        trainer: SFTTrainer instance with configured dataloader.
        tokenizer: Tokenizer for decoding and EOS token lookup.
        sft_config: SFTConfig with batch size and context length settings.
    """
    logger.info("=" * 50)
    logger.info("STARTING PRE-FLIGHT SANITY CHECKS")
    logger.info("=" * 50)

    # Step 1: Verify EOS token leak & masking
    logger.info("CHECK 1: EOS Token & Completion Masking...")
    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))

    # Check the first sample in the batch
    input_ids = batch['input_ids'][0]
    labels = batch['labels'][0]

    # Find where the actual content ends (before padding)
    # Look for the EOS token to verify the model will learn to stop
    eos_id = tokenizer.eos_token_id
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    found_eos_with_label = False
    for i in range(len(input_ids)):
        # Check if EOS exists and has a valid label (not -100)
        if input_ids[i] == eos_id and labels[i] != -100:
            found_eos_with_label = True
            logger.info(f"Found EOS at index {i} with valid Label. Model WILL learn to stop.")

            # Log a snippet of the end of the SQL to verify
            context = tokens[max(0, i-5):i+1]
            logger.info(f"   Context: {' '.join(context)}")
            break

    if not found_eos_with_label:
        logger.warning("No EOS token found with a valid label! Model may hallucinate endlessly.")

    # Step 2: Batch size & gradient math
    logger.info("CHECK 2: Effective Batch Size & Throughput...")
    total_batch_size = sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps

    logger.info(f"Per-device Batch Size: {sft_config.per_device_train_batch_size}")
    logger.info(f"Gradient Accumulation Steps: {sft_config.gradient_accumulation_steps}")
    logger.info(f"Effective Batch Size (samples per update): {total_batch_size}")

    if 32 <= total_batch_size <= 128:
        logger.info("Effective Batch Size is in the 'Sweet Spot' for SQL Distillation.")
    else:
        logger.warning("Effective Batch Size is outside the usual 32-128 range.")

    # Step 3: The "silence" warning
    # Assuming ~70s per step on L4/H200 for large context
    wait_time_min = (sft_config.gradient_accumulation_steps * _PREFLIGHT_SECS_PER_STEP) / 60
    logger.info("CHECK 3: The 'Silence' Period...")
    logger.info(f"You will only see a Loss update every {wait_time_min:.1f} minutes.")
    logger.info("If Loss is 'NaN' or '0.0' after the first update, KILL the run.")

    logger.info("=" * 50)


def evaluate_with_execution_match(model, tokenizer, db_manager, dataset, max_new_tokens: int = 2000) -> Dict[str, Any]:
    """Evaluate model using execution accuracy by comparing predicted vs gold SQL.

    Args:
        model: Language model for SQL generation.
        tokenizer: Tokenizer for encoding prompts and decoding outputs.
        db_manager: DatabaseManager for executing and comparing SQL queries.
        dataset: Dataset with 'prompt', 'teacher_sql', and 'db_id' columns.
        max_new_tokens: Maximum new tokens to generate per sample.

    Returns:
        Dict with 'exact_match_accuracy', 'execution_accuracy', 'total_samples',
        'gold_query_execution_failures', 'predicted_query_execution_failures'.
    """
    model.eval()
    
    predictions = []
    references = []

    for example in tqdm(dataset, desc="Running Evaluation"):
        prompt = example['prompt']
        teacher_sql = example['teacher_sql']
        db_id = example['db_id']

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_sql = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        predictions.append({"db_id": db_id, "predicted_sql": generated_sql})
        references.append({"db_id": db_id, "teacher_sql": teacher_sql})

    exact_match_count = 0
    execution_match_count = 0
    gold_failed_count = 0
    prediction_failed_count = 0
    total_samples = len(predictions)

    for i in range(total_samples):
        predicted_sql = predictions[i]['predicted_sql']
        teacher_sql = references[i]['teacher_sql']
        db_id = predictions[i]['db_id']

        if predicted_sql.strip() == teacher_sql.strip():
            exact_match_count += 1

        try:
            cmp = db_manager.execute_and_compare(db_id, teacher_sql, predicted_sql)

            if not cmp.gold_exec_status:
                gold_failed_count += 1
            if not cmp.predict_exec_status:
                prediction_failed_count += 1
            if cmp.match_status:
                execution_match_count += 1
                
        except Exception as e:
            # Catches unexpected errors from DatabaseManager not raised at the call site
            logger.error(f"An unexpected error occurred during execute_and_compare for db '{db_id}': {e}")


    # Calculate final metrics
    exact_match_accuracy = (exact_match_count / total_samples) * 100 if total_samples > 0 else 0
    execution_accuracy = (execution_match_count / total_samples) * 100 if total_samples > 0 else 0

    return {
        "exact_match_accuracy": exact_match_accuracy,
        "execution_accuracy": execution_accuracy,
        "total_samples": total_samples,
        "gold_query_execution_failures": gold_failed_count,
        "predicted_query_execution_failures": prediction_failed_count,
    }

class CustomEvaluationCallback(TrainerCallback):
    """Trainer callback that runs execution-match evaluation at scheduled steps.

    Called automatically by HuggingFace's SFTTrainer at each `on_step_end`
    event. Uses `evaluate_with_execution_match` to compute SQL execution
    accuracy.

    Attributes:
        tokenizer: Tokenizer for decoding model outputs.
        db_manager: DatabaseManager for SQL execution and comparison.
        eval_datasets: Dict mapping split name to dataset for evaluation.
        eval_steps: Interval (in global steps) between evaluations.
        accelerator: Accelerator instance for distributed-training-awareness.
    """

    def __init__(self, tokenizer, db_manager, eval_datasets, eval_steps, accelerator):
        """Initialise the callback with evaluation dependencies and schedule.

        Args:
            tokenizer: Tokenizer matching the model being trained.
            db_manager: DatabaseManager for executing and comparing SQL queries.
            eval_datasets: Dict of split_name -> Dataset, each with 'prompt',
                'teacher_sql', and 'db_id' columns.
            eval_steps: Number of global training steps between evaluations.
            accelerator: Accelerator for checking if this is the main process.
        """
        self.tokenizer = tokenizer
        self.db_manager = db_manager
        self.eval_datasets = eval_datasets
        self.eval_steps = eval_steps
        self.accelerator = accelerator

    def on_step_end(self, args, state, control, **kwargs):
        """Run execution-match evaluation on the current model at scheduled steps.

        Called automatically by the HuggingFace Trainer after each step.
        Evaluation only runs on the main process and only when
        ``state.global_step`` is a non-zero multiple of ``self.eval_steps``.

        Args:
            args: ``TrainingArguments`` instance passed by the Trainer.
            state: ``TrainerState`` containing current step and epoch info.
            control: ``TrainerControl`` for modifying Trainer behaviour.
            **kwargs: Additional Trainer keyword arguments, including 'model'
                (the current model being trained) and optionally 'wandb_run'
                for logging metrics to Weights & Biases.
        """
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            if self.accelerator.is_main_process:
                model = kwargs["model"]
                logger.info(f"Step {state.global_step}: Running custom evaluation on best model so far...")
                
                # You'd need to load the best model here, which is complex.
                # A simpler approach is to use the current model.
                
                for name, dataset in self.eval_datasets.items():
                    metrics = evaluate_with_execution_match(
                        model,
                        self.tokenizer,
                        self.db_manager,
                        dataset
                    )
                    
                    # Log the metrics directly
                    if kwargs.get("wandb_run"):
                        kwargs["wandb_run"].log({f"custom_eval_{name}_metrics": metrics}, step=state.global_step)
                    else:
                        logger.info(f"Custom metrics for {name}: {metrics}")


def print_trainable_parameters_mine(model: AutoModelForCausalLM) -> None:
    """Log the number and percentage of trainable parameters.

    Args:
        model: PyTorch model to analyze.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )


def run_knowledge_distillation(
    accelerator: Accelerator,
    model_name: str,
    output_dir: str,
    config_file: str,
    lora_r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    learning_rate: Optional[float] = None,
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
    quantization: Optional[int] = None,
    max_steps: Optional[int] = None,
    dataset_name: Optional[str] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    """Run LoRA-based knowledge distillation training.

    Args:
        accelerator: Accelerator instance for distributed training.
        model_name: HuggingFace model identifier for the base model.
        output_dir: Directory for saving checkpoints and outputs.
        config_file: Path to configuration INI file.
        lora_r: LoRA rank (overrides config if provided).
        lora_alpha: LoRA alpha scaling factor (overrides config if provided).
        learning_rate: Training learning rate (overrides config if provided).
        batch_size: Per-device batch size (overrides config if provided).
        epochs: Number of training epochs (overrides config if provided).
        quantization: Bit quantization (4 or 8), or None for full precision.
        max_steps: Maximum training steps (overrides epochs when > 0). Use for debugging.
        dataset_name: HuggingFace dataset identifier (e.g. 'craterlabs/struct-sql-data').
            When provided, loads data from HuggingFace and bakes schema context.
            When None, falls back to local JSON files from kd_data_dir.
        resume: If True, resume from latest checkpoint if available. If False,
            delete existing output directory and start a fresh training run.

    Returns:
        Dict with keys: ``'status'`` ("success"), ``'output_dir'`` (the run
        directory passed in), ``'model_specific_output_dir'`` (subdirectory
        for this model's data and checkpoints), and ``'adapter_path'`` (path
        to the final saved LoRA adapter weights).

    Raises:
        ConfigurationError: If config file is invalid.
        BirdSchemaLoaderError: If schema loading fails.
        DatabaseManagerError: If database initialization fails.
    """
    # --- Load parameters from config file (All Processes) ---
    try:
        config_loader = ConfigLoader(config_file)
        general_paths = config_loader.get_general_paths()
        bird_training_paths = config_loader.get_bird_train_dataset_paths()
        db_train_settings = config_loader.get_database_train_settings()
        distillation_params_from_config = config_loader.get_distillation_parameters()
        training_params = config_loader.get_training_parameters()
    except ConfigurationError as e:
        logger.error(f"Configuration error during fine-tuning setup: {e}")
        raise

    try:
        bird_schema_loader_instance = BirdSchemaLoader(bird_training_paths['train_tables_json_path'])
        db_manager_instance = DatabaseManager(db_train_settings, bird_schema_loader=bird_schema_loader_instance)

    except (BirdSchemaLoaderError, DatabaseManagerError) as e:
        logger.error(f"Error initializing database/schema managers: {e}")
        raise

    model_specific_output_dir = os.path.join(output_dir, f"model_kd_data_{model_name.replace('/', '-')}")

    # --- Fresh start: wipe existing checkpoints when resume=False ---
    if not resume and accelerator.is_main_process:
        if os.path.exists(model_specific_output_dir):
            logger.info(f"resume=False: Deleting existing output directory {model_specific_output_dir} for a fresh run.")
            shutil.rmtree(model_specific_output_dir)

    # --- Variables used by all processes ---
    lora_r = lora_r if lora_r is not None else distillation_params_from_config['lora_r']
    lora_alpha = lora_alpha if lora_alpha is not None else distillation_params_from_config['lora_alpha']
    learning_rate = learning_rate if learning_rate is not None else distillation_params_from_config['learning_rate']
    batch_size = batch_size if batch_size is not None else distillation_params_from_config['batch_size']
    epochs = epochs if epochs is not None else distillation_params_from_config['epochs']
    quantization = quantization if quantization is not None else distillation_params_from_config.get('quantization', None)
    max_length = distillation_params_from_config.get('max_length', 15000)
    max_new_tokens = distillation_params_from_config.get('max_new_tokens', 2000)
    max_steps = max_steps if max_steps is not None else -1  # -1 means use epochs

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=general_paths['model_cache_dir'], use_fast=True)
    tokenizer.truncation_side = "left"

    # --- Data Prep & Caching (Main Process Only) ---
    if accelerator.is_main_process:
        os.makedirs(model_specific_output_dir, exist_ok=True)

        curated_data_dir = general_paths['kd_data_dir']
        dataset_kwargs = _resolve_dataset_kwargs(dataset_name, curated_data_dir, model_specific_output_dir)
        _load_or_prepare_datasets(dataset_kwargs, model_specific_output_dir, db_manager_instance)

    accelerator.wait_for_everyone() # All processes wait here

    # --- Load Data, Model, and Tokenizer (All Processes) ---
    train_dataset = load_from_disk(os.path.join(model_specific_output_dir, "train_raw"))
    val_dataset = load_from_disk(os.path.join(model_specific_output_dir, "all_raw_eval"))
    val_in_domain_dataset = load_from_disk(os.path.join(model_specific_output_dir, "val_in_domain"))
    val_out_of_domain_dataset = load_from_disk(os.path.join(model_specific_output_dir, "val_out_of_domain"))

    eval_datasets_dict = {
        "out_domain": val_out_of_domain_dataset,
        "in_domain": val_in_domain_dataset
    }
    if accelerator.is_main_process:
        logger.info("Loading base model and tokenizer (from cache)...")
    
    bnb_config = create_bnb_config(quantization)
    attn_impl = _detect_attn_implementation()
    logger.info("Training: using attention implementation: %s", attn_impl)

    # NOTE: trust_remote_code is required for Qwen — see LLMInitializer for details.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, cache_dir=general_paths['model_cache_dir'], trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )

    logger.info("Preparing model for Accelerator")
   
    # --- LoRA Adapter Setup (All Processes) ---
    if accelerator.is_main_process:
        logger.info("Setting up LoRA adapter configuration...")
    
    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    if accelerator.is_main_process:
        logger.info("Configuring Hugging Face Trainer...")

    sft_config = SFTConfig(
        output_dir=model_specific_output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=training_params['gradient_accumulation_steps'],
        learning_rate=learning_rate, num_train_epochs=epochs, max_steps=max_steps,
        logging_steps=training_params['logging_steps'],
        eval_strategy="steps",
        eval_steps=training_params['eval_steps'],
        save_strategy="steps",
        # Must exceed early_stopping_patience so the best checkpoint isn't evicted
        # before load_best_model_at_end can retrieve it. patience + 2 adds a small buffer.
        save_total_limit=training_params['early_stopping_patience'] + 2,

        save_steps=training_params['save_steps'],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim=training_params['optim'],
        warmup_steps=training_params['warmup_steps'],
        completion_only_loss=True,
        logging_strategy="steps",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        bf16=True,
        seed=training_params['seed'],
        max_length=max_length,
        remove_unused_columns=False,
        packing=False,
        eval_packing=False,
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=training_params['early_stopping_patience'],
        early_stopping_threshold=training_params['early_stopping_threshold'],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[early_stopping_callback],
        peft_config=lora_config,
    )
    
    if accelerator.is_main_process:
        logger.info("Checking data masking (Completion Only Loss)...")
        batch = next(iter(trainer.get_train_dataloader()))
        
        input_ids = batch['input_ids'][0]
        labels = batch['labels'][0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        logger.debug("\n" + "="*60)
        logger.debug("TRANSITION MASKING CHECK (Scanning 28k Block)")
        logger.debug("="*60)

        count = 0
        for i in range(1, len(input_ids)):
            # We are looking for the flip from -100 to a real ID
            if labels[i] != -100 and labels[i-1] == -100:
                count += 1
                logger.debug("\n[FOUND COMPLETION START #%d] at index %d", count, i)

                # Print the 5 tokens BEFORE (should be IGNORING) and 5 tokens AFTER (should be LEARNING)
                for j in range(i-5, i+5):
                    if j < 0 or j >= len(input_ids): continue
                    t = tokens[j]
                    l = labels[j].item()
                    s = "LEARNING" if l != -100 else "IGNORING"
                    marker = "===>" if j == i else "    "
                    logger.debug("%s Token: [%-15s] | Label: %-6s | Status: %s", marker, t, l, s)

            # Stop after showing 3 examples in the packed block
            if count >= 3:
                break

        if count == 0:
            logger.error("CRITICAL: No learning tokens found! Everything is masked.")
        logger.debug("="*60 + "\n")
    

    if accelerator.is_main_process:
        print_trainable_parameters_mine(trainer.model)

    if trainer.accelerator.is_main_process:
        run_pre_flight_checks(trainer, tokenizer, sft_config)

    gc.collect()
    torch.cuda.empty_cache()

    if resume and os.path.exists(model_specific_output_dir):
        latest_checkpoint = get_last_checkpoint(model_specific_output_dir)

        # Check if a checkpoint was found and if it's a valid directory
        if latest_checkpoint and os.path.isdir(latest_checkpoint):
            # Add a check for essential files within the checkpoint directory
            logger.info(f"Found checkpoint: {latest_checkpoint}")
            required_files = ["adapter_model.safetensors", "adapter_config.json"]
            is_valid_checkpoint = all(os.path.exists(os.path.join(latest_checkpoint, f)) for f in required_files)

            if is_valid_checkpoint:
                logger.info("Found and verified a valid checkpoint. Resuming training.")
                trainer.train(resume_from_checkpoint=latest_checkpoint)
            else:
                logger.warning("Found checkpoint directory, but it's invalid: %s. Starting a new training run.", latest_checkpoint)
                trainer.train()
        else:
            logger.info("No valid checkpoint found. Starting a new training run.")
            trainer.train()
    else:
        if not resume:
            logger.info("resume=False: Starting a fresh training run.")
        else:
            logger.info("Output directory does not exist. Starting a new training run.")
        trainer.train()
    
    adapter_output_dir = os.path.join(model_specific_output_dir, "lora_adapter")
    if accelerator.is_main_process:
        os.makedirs(adapter_output_dir, exist_ok=True)
        trainer.save_model(adapter_output_dir)
        logger.info(f"LoRA adapter weights saved to {adapter_output_dir}.")
    
    return {
        "status": "success",
        "output_dir": output_dir,
        "model_specific_output_dir": model_specific_output_dir,
        "adapter_path": adapter_output_dir
    }