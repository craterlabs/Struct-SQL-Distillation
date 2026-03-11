#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distillation Data Generation Pipeline.

Entry point for generating curated training data through stratified sampling
based on SQL complexity levels. Coordinates SQL classification and data building.

@author: Khushboo Thaker, Yony Bresler, Khalid Eidoo
@license: Crater Labs (C)
"""

import argparse
import json
import logging
import os
import sys
from src.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from src.data_processing.sql_classifier import SQLClassifier
from src.data_processing.struct_data_builder import run_data_curation

def main() -> None:
    """Run two-step data generation: classify SQL complexity, then curate stratified datasets.

    Step 1 — SQL Complexity Classification:
        Reads raw BIRD training questions from the path configured in
        ``[bird_training_paths] > train_questions_json_path``, classifies each
        SQL query into a complexity level (L1–L4) via ``SQLClassifier``, and
        writes the enriched level file to
        ``[bird_training_paths] > train_questions_level_json_path``.
        If the level file already exists, this step is skipped.

    Step 2 — Stratified Data Curation:
        Delegates to ``run_data_curation``, which splits databases into
        in-domain / out-of-domain pools and runs a 4-stage LLM-assisted
        pipeline to produce ``train.json``, ``val_in_domain.json``, and
        ``val_out_of_domain.json`` under ``--output_dir``.

    CLI arguments: ``--config``, ``--output_dir`` (required), ``--train_size``,
    ``--val_id_size``, ``--val_ood_size``. See ``guide/DATA_GENERATION_GUIDE.txt``
    for the complete reference.
    """
    parser = argparse.ArgumentParser(description="Distillation Data Generation Pipeline")
    # Only x, y, z and basic paths are needed as arguments; others come from config
    parser.add_argument("--config", type=str, default="config.ini", help="Path to config.ini")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save curated sets")
    parser.add_argument("--train_size", type=int, default=1000, help="Target training size (x)")
    parser.add_argument("--val_id_size", type=int, default=150, help="Target in-domain val size (y)")
    parser.add_argument("--val_ood_size", type=int, default=150, help="Target out-of-domain val size (z)")
    args = parser.parse_args()

    # Initialize ConfigLoader
    config = ConfigLoader(args.config)
    
    # Get paths from config
    train_paths = config.get_bird_train_dataset_paths()
    raw_input_path = train_paths['train_questions_json_path']
    level_file_path = train_paths['train_questions_level_json_path']

    # Ensure the parent directory exists (guard against stale files blocking mkdir)
    level_dir = os.path.dirname(level_file_path)
    if os.path.exists(level_dir) and not os.path.isdir(level_dir):
        sys.exit(f"Error: '{level_dir}' exists as a file but needs to be a directory. "
                 "Remove it and re-run.")
    os.makedirs(level_dir, exist_ok=True)

    if not os.path.exists(level_file_path):
        logger.info("Level file not found at %s. Generating from %s...", level_file_path, raw_input_path)
        classifier = SQLClassifier()
        with open(raw_input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for idx, record in enumerate(data):
            # Assign a question_id if the raw data doesn't have one
            if 'question_id' not in record:
                record['question_id'] = idx
            # Assign complexity labels
            record['level'] = classifier.classify(record.get('SQL', ''))

        with open(level_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logger.info("Successfully created: %s", level_file_path)
    else:
        logger.info("Using existing level file found in config: %s", level_file_path)

    # All paths (including the level file) are resolved from config.ini
    run_data_curation(
        output_dir=args.output_dir,
        train_size=args.train_size,
        in_domain_val_size=args.val_id_size,
        out_of_domain_val_size=args.val_ood_size,
        config_file=args.config,
    )

if __name__ == "__main__":
    main()
