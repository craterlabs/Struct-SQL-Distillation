#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distillation Data Utilities.

Data processing functions for knowledge distillation including dataset splitting,
prompt-completion formatting, tokenization, and data validation.

@author: Khushboo Thaker, Yony Bresler, Khalid Eidoo
@license: Crater Labs (C)
"""

import logging
import os
from typing import Optional

from datasets import load_dataset, DatasetDict

from src.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

def prepare_static_distillation_dataset(
    output_dir: str,
    db_manager: DatabaseManager,
    dataset_name: Optional[str] = None,
    raw_data_dir: Optional[str] = None,
    prompt_template_path: str = "prompts/structsql.txt"
) -> DatasetDict:
    """Bake schema context into train/val datasets and save to disk.

    Supports two data sources:
    - HuggingFace dataset (via dataset_name, e.g. 'craterlabs/struct-sql-data')
    - Local JSON files (via raw_data_dir containing train.json, val_in_domain.json, val_out_of_domain.json)

    Both sources must have columns: question, db_id, query_plan, predicted_sql.

    Args:
        output_dir: Directory to save the processed DatasetDict.
        db_manager: DatabaseManager instance for schema generation.
        dataset_name: HuggingFace dataset identifier (e.g. 'craterlabs/struct-sql-data').
        raw_data_dir: Directory containing raw JSON files (legacy path).
        prompt_template_path: Path to the prompt template file (default: prompts/structsql.txt).

    Returns:
        DatasetDict with splits 'train', 'val_in_domain', 'val_out_of_domain',
        each containing columns: prompt, completion, db_id, teacher_sql.
    """
    # Load prompt template (same one used at inference time)
    logger.info(f"Loading prompt template from: {prompt_template_path}")
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    if dataset_name:
        logger.info(f"Loading dataset from HuggingFace: {dataset_name}")
        hf_datasets = load_dataset(dataset_name)
        # Map HF split names to our internal names
        raw_datasets = DatasetDict({
            "train": hf_datasets["train"],
            "val_in_domain": hf_datasets["validation_in_domain"],
            "val_out_of_domain": hf_datasets["validation_out_of_domain"]
        })
    elif raw_data_dir:
        logger.info(f"Loading dataset from local JSON files: {raw_data_dir}")
        data_files = {
            "train": os.path.join(raw_data_dir, "train.json"),
            "val_in_domain": os.path.join(raw_data_dir, "val_in_domain.json"),
            "val_out_of_domain": os.path.join(raw_data_dir, "val_out_of_domain.json")
        }
        raw_datasets = load_dataset("json", data_files=data_files)
    else:
        raise ValueError("Either dataset_name or raw_data_dir must be provided.")

    # Drop the 'split' column if present (HF dataset includes it as metadata)
    for split_name in raw_datasets:
        if "split" in raw_datasets[split_name].column_names:
            raw_datasets[split_name] = raw_datasets[split_name].remove_columns(["split"])

    def format_row(example):
        # Render the 'Struct' schema context ONCE
        struct_context = db_manager.get_semantic_schema_for_llm_with_rows(
            db_id=example['db_id'],
            num_rows=3
        )

        # Use the same prompt template as inference (includes few-shot examples)
        # rstrip + "\n" ensures a clean token boundary (no trailing space before newline)
        prompt = prompt_template.format(
            db_schema_context_hmr=struct_context,
            nl_question=example['question'],
            hint=example.get('evidence', ''),
            format_instructions=''
        ).rstrip() + "\n"

        # Completion starts directly after the prompt's trailing newline
        completion = f"{example['query_plan']}\n\n## SQL Query:\n{example['predicted_sql']}"

        return {
            "prompt": prompt,
            "completion": completion,
            "db_id": example['db_id'],
            "teacher_sql": example['predicted_sql']
        }

    # Map across all splits (Train, ID-Val, OOD-Val)
    logger.info("Baking schema context into all dataset splits...")
    static_dataset = raw_datasets.map(format_row, remove_columns=raw_datasets['train'].column_names)

    # Save as a DatasetDict to disk
    logger.info(f"Saving all static splits to {output_dir}")
    static_dataset.save_to_disk(output_dir)

    return static_dataset