#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Inference Runner for Text-to-SQL.

Runs batched inference using fine-tuned models on BIRD benchmark questions.
Supports checkpoint resumption and execution accuracy evaluation.

@author: Khushboo Thaker, Yony Bresler, Khalid Eidoo
@license: Crater Labs (C)
"""

import argparse
import os
import json
import logging
import sys
from typing import Any, Optional, Tuple
import torch
import re
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _setup_file_logging(log_file: str = "submission_run.log") -> None:
    """Add a file handler to the root logger. Called only from main().

    Args:
        log_file: Path of the log file to write. Appends to existing file.
    """
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

try:
    from src.database_manager import DatabaseManager
    from src.bird_schema_loader import BirdSchemaLoader
except ImportError as e:
    logger.critical(f"Could not import src modules: {e}", exc_info=True)
    sys.exit(1)

# Intentionally empty: the structsql.txt prompt template includes its own
# format instructions inline. This placeholder exists for compatibility with
# prompt templates that expose a {format_instructions} slot.
FORMAT_INSTRUCTIONS = ''
# Slight repetition penalty to discourage the model from looping on SQL clauses.
_REPETITION_PENALTY: float = 1.15

def validate_file(file_path: str, description: str) -> Optional[Any]:
    """Load and validate a JSON file, returning None on error.

    Args:
        file_path: Path to the JSON file to load.
        description: Human-readable label used in error log messages.

    Returns:
        Parsed JSON content (list or dict), or None if the file is
        missing or unreadable.
    """
    if not os.path.exists(file_path):
        logger.error("[MISSING] %s: %s", description, file_path)
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error("[ERROR] reading %s: %s", description, e)
        return None

def load_model(model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer from a HuggingFace path with GPU optimization.

    Args:
        model_path: HuggingFace model identifier or local path.

    Returns:
        Tuple of (model, tokenizer) loaded in bfloat16 with device_map="auto".

    Raises:
        Exception: If model loading fails (logged as FATAL before re-raising).
    """
    logger.info("[LOADING] Model from %s ...", model_path)

    try:
        # NOTE: trust_remote_code is required for Qwen — see LLMInitializer for details.
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        tokenizer.padding_side = "left"  # Crucial for batching with decoder models
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Configuring for GPU (bfloat16)")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16, # Better for Qwen3/L4 than float16
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        model.eval()
        return model, tokenizer
    except Exception as e:
        logger.critical("[FATAL] Failed to load model: %s", e)
        raise e

def extract_sql(response_content: Optional[str]) -> str:
    """Extract SQL from model response using markdown blocks or heuristic parsing.

    Args:
        response_content: Raw model output string, or None.

    Returns:
        Extracted SQL string terminated with ';', or empty string if no SQL
        found. Thinking tags (<think>...</think>) are stripped before extraction.
    """
    if not response_content: return ""
    # Strip thinking tags
    cleaned = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()
    raw_sql = ""

    # Strategy: Markdown Code Blocks
    match = re.search(r"```sql\n(.*?)\n```", cleaned, re.DOTALL | re.IGNORECASE)
    if match: 
        raw_sql = match.group(1).strip()
            
    # Fallback: Heuristic Scanning
    if not raw_sql:
        lines = cleaned.split('\n')
        sql_lines = []
        capture_mode = False
        for line in lines:
            normalized = line.strip().upper()
            if not capture_mode and (normalized.startswith("SELECT ") or normalized.startswith("WITH ")):
                capture_mode = True
            if capture_mode: sql_lines.append(line)
        if sql_lines: raw_sql = "\n".join(sql_lines).strip()
        elif "SELECT" in cleaned.upper(): raw_sql = cleaned

    raw_sql = raw_sql.replace("```", "").strip()
    if raw_sql and not raw_sql.endswith(';'): raw_sql += ';'
    return raw_sql

def main():
    """Run the batch inference pipeline for BIRD benchmark evaluation.

    Loads input questions and the fine-tuned model, constructs per-question
    prompts with semantic schema context, generates SQL in batches using greedy
    decoding, and saves results in BIRD submission format.  Supports checkpoint
    resumption: if ``--output_file`` already exists, completed question IDs are
    skipped automatically.  On CUDA OOM, the current batch is skipped and the
    pipeline continues.

    CLI arguments: ``--input_file``, ``--db_path``, ``--tables_file``,
    ``--model_path``, ``--prompt_file`` (all required), ``--output_file``,
    ``--batch_size``, ``--max_new_tokens``.  See ``guide/RUN_INFERENCE_GUIDE.txt``
    for the complete reference.
    """
    _setup_file_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--tables_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="predict_dev.json")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2) # L4 can handle 4-8 depending on context length
    parser.add_argument("--max_new_tokens", type=int, default=2000,
                        help="Maximum new tokens to generate per query (default: 2000).")
    args = parser.parse_args()

    raw_output_file = args.output_file.replace(".json", "_raw.jsonl")
    # A. Validate & Setup
    questions = validate_file(args.input_file, "Input Questions")
    if not questions: sys.exit(1)
    for idx, q in enumerate(questions):
        if 'question_id' not in q:
            q['question_id'] = idx

    results = {}
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info("Resuming from %d existing records.", len(results))
        except Exception as e:
            logger.warning(f"Failed to load existing results from {args.output_file}: {e}. Starting fresh.")

    # B. Load Model First (Claim VRAM)
    model, tokenizer = load_model(args.model_path)

    # C. Initialize Database Modules
    schema_loader = BirdSchemaLoader(args.tables_file)
    db_manager = DatabaseManager({'database_root_path': args.db_path, 'db_type': 'sqlite'}, schema_loader)
    if not os.path.exists(args.prompt_file):
        logger.error("[MISSING] Prompt file: %s", args.prompt_file)
        sys.exit(1)
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # D. Filter to remaining (unfinished) questions
    todo_questions = [q for q in questions if str(q['question_id']) not in results]

    logger.info("Processing %d of %d queries (%d already completed).", len(todo_questions), len(questions), len(results))

    # E. Batch Inference Loop
    for batch_start in range(0, len(todo_questions), args.batch_size):
        batch_items = todo_questions[batch_start : batch_start + args.batch_size]
        batch_prompts = []
        batch_meta = []

        # 1. Prepare Batch Prompts
        for item in batch_items:
            db_id = item['db_id']
            q_id = str(item['question_id'])
            
            # Generate schema on-the-fly to save RAM
            schema_context = db_manager.get_semantic_schema_for_llm_with_rows(db_id)
            
            user_prompt = prompt_template.format(
                db_schema_context_hmr=schema_context,
                nl_question=item['question'],
                hint=item.get('evidence', ''),
                format_instructions=FORMAT_INSTRUCTIONS
            )

            batch_prompts.append(user_prompt)
            batch_meta.append((q_id, db_id))

        # 2. Run Inference
        try:
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    repetition_penalty=_REPETITION_PENALTY,
                )

                generated_sequences = []

                for sample_idx in range(outputs.shape[0]):
                    prompt_len = inputs.input_ids.shape[1]
                    gen_ids = outputs[sample_idx, prompt_len:]
                    generated_sequences.append(gen_ids)

                decoded_responses = tokenizer.batch_decode(
                    generated_sequences,
                    skip_special_tokens=True
                )

            with open(raw_output_file, 'a', encoding='utf-8') as raw_f:
                for idx, response in enumerate(decoded_responses):
                    q_id, db_id = batch_meta[idx]
                    raw_entry = {
                        "question_id": q_id,
                        "raw_content": response
                    }
                    raw_f.write(json.dumps(raw_entry) + "\n")

                    sql = extract_sql(response)
                    results[q_id] = f"{sql}\t----- bird -----\t{db_id}"

            # Save after every batch so progress survives failures
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)

            logger.info("Progress saved. Total records: %d", len(results))

        except torch.cuda.OutOfMemoryError:
            logger.error("[OOM] CUDA out of memory: batch too large. Clearing cache and skipping batch.")
            torch.cuda.empty_cache()
            gc.collect()
            continue
            
    # F. Final Save
    logger.info("Saving final results to %s ...", args.output_file)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    logger.info("Run complete.")

if __name__ == "__main__":
    main()