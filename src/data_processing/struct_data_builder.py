#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structured Data Builder for Knowledge Distillation.

Orchestrates the data curation pipeline: loads data, splits databases,
manages caches, and delegates to specialized modules for inference,
curation, and reporting.

@author: Khushboo Thaker, Yony Bresler, Khalid Eidoo
@license: Crater Labs (C)
"""

import concurrent.futures
import json
import logging
import os
import random
from pathlib import Path

from tqdm.auto import tqdm

from src.data_processing.utils import (
    load_prompt_template,
    is_successful_inference,
    get_target_counts,
)
from src.data_processing.curation import (
    curate_in_domain_datasets,
    curate_single_dataset,
)
from src.data_processing.sql_classifier import (
    SQL_LEVEL_L1,
    SQL_LEVEL_L2,
    SQL_LEVEL_L3,
    SQL_LEVEL_L4,
    SQL_LEVEL_OTHER,
)

logger = logging.getLogger(__name__)

from src.config_loader import ConfigLoader, ConfigurationError
from src.llm_initializer import initialize_azure_openai_llm
from src.database_manager import DatabaseManager
from src.bird_schema_loader import BirdSchemaLoader, BirdSchemaLoaderError


def run_data_curation(
    output_dir='./data/BIRD/active_split_100',
    db_split_ratio=0.8,
    train_size=2000,
    in_domain_val_size=150,
    out_of_domain_val_size=350,
    seed=42,
    prompt_template='structsql',
    config_file='config.ini',
    model_type='gpt4o',
    model_name='gpt-4o',
    api_delay_seconds=4,
):
    """Split a text-to-SQL dataset by database, run multi-stage curation, and save the result.

    Splits databases into in-domain / out-of-domain pools, then runs the 4-stage curation
    pipeline to produce train, in-domain validation, and out-of-domain validation sets.

    Args:
        output_dir: Directory where train.json, val_in_domain.json, and
            val_out_of_domain.json are written.
        db_split_ratio: Fraction of databases assigned to the in-domain pool (0–1).
        train_size: Target number of samples for the training set.
        in_domain_val_size: Target number of samples for the in-domain validation set.
        out_of_domain_val_size: Target number of samples for the out-of-domain validation set.
        seed: Random seed for reproducible database splitting and sampling.
        prompt_template: Name of the prompt template to load (matches a file in prompts/).
        config_file: Path to the config.ini file.
        model_type: LLM backend type passed to the inference state initializer.
        model_name: Model deployment name used in API calls.
        api_delay_seconds: Seconds to wait between API calls to avoid rate limits.

    Note:
        Saves train.json, val_in_domain.json, and val_out_of_domain.json to
        output_dir. Per-question inference results are cached incrementally in
        output_dir/individual_results/.
    """
    random.seed(seed)

    individual_results_dir = os.path.join(output_dir, "individual_results")
    os.makedirs(individual_results_dir, exist_ok=True)
    logger.info(f"Individual results directory created at: {individual_results_dir}")
    logger.info("Script initialized. Starting data loading and splitting process.")

    # --- Initialize LLM / DB components ---
    state_components, config_paths = _initialize_state_components(
        config_file, prompt_template, model_name, model_type, api_delay_seconds, individual_results_dir
    )

    # --- Resolve input file path from config ---
    input_file = config_paths.get('train_questions_level_json_path')
    if not input_file:
        logger.error("'train_questions_level_json_path' not found in config. Cannot proceed.")
        return
    logger.info(f"Using level file from config: {input_file}")

    # --- Load and validate input data ---
    validated_data = _load_validated_data(input_file)
    if validated_data is None:
        return

    # --- Classify cached results vs. candidates ---
    successful_cache, failed_cache, candidate_pool = _load_caches(
        validated_data, individual_results_dir
    )

    # --- Split databases into in-domain / out-of-domain ---
    in_domain_dbs, out_of_domain_dbs = _split_databases(validated_data, db_split_ratio)

    in_domain_successful_cache = [s for s in successful_cache if s.get('db_id') in in_domain_dbs]
    in_domain_candidates = [s for s in candidate_pool if s.get('db_id') in in_domain_dbs]
    out_of_domain_successful_cache = [s for s in successful_cache if s.get('db_id') in out_of_domain_dbs]
    out_of_domain_candidates = [s for s in candidate_pool if s.get('db_id') in out_of_domain_dbs]

    # Log split statistics
    in_domain_failed_cache = [s for s in failed_cache if s.get('db_id') in in_domain_dbs]
    out_of_domain_failed_cache = [s for s in failed_cache if s.get('db_id') in out_of_domain_dbs]
    logger.info(f"- In-Domain Successful Cache: {len(in_domain_successful_cache)} records")
    logger.info(f"- In-Domain Failed Cache: {len(in_domain_failed_cache)} records")
    logger.info(f"- In-Domain Candidates: {len(in_domain_candidates)} records")
    logger.info(f"- Out-of-Domain Successful Cache: {len(out_of_domain_successful_cache)} records")
    logger.info(f"- Out-of-Domain Failed Cache: {len(out_of_domain_failed_cache)} records")
    logger.info(f"- Out-of-Domain Candidates: {len(out_of_domain_candidates)} records")

    TARGET_DISTRIBUTION = {
        SQL_LEVEL_L4: 0.25,
        SQL_LEVEL_L3: 0.25,
        SQL_LEVEL_L2: 0.25,
        SQL_LEVEL_L1: 0.25,
        SQL_LEVEL_OTHER: 0.0,
    }

    # --- Load existing data for resumability ---
    curated_train = _load_existing_json(output_dir, 'train.json')
    curated_val_id = _load_existing_json(output_dir, 'val_in_domain.json')
    curated_val_ood = _load_existing_json(output_dir, 'val_out_of_domain.json')

    # Global set of IDs to skip during new sampling
    locked_ids = {str(s['question_id']) for s in (curated_train + curated_val_id + curated_val_ood)}

    logger.info(f"Loaded {len(curated_train)} training, {len(curated_val_id)} ID-val, and {len(curated_val_ood)} OOD-val samples.")

    # --- Curate In-Domain Datasets ---
    logger.info("\n--- Starting curation for IN-DOMAIN datasets ---")
    train_targets = get_target_counts(train_size, TARGET_DISTRIBUTION)
    val_in_domain_targets = get_target_counts(in_domain_val_size, TARGET_DISTRIBUTION)

    train_set, val_in_domain_set = curate_in_domain_datasets(
        train_targets, val_in_domain_targets,
        in_domain_successful_cache, in_domain_candidates,
        in_domain_dbs,
        state_components,
        existing_train=curated_train,
        existing_val=curated_val_id,
        locked_ids=locked_ids
    )

    # --- Curate Out-of-Domain Dataset ---
    logger.info("\n--- Starting curation for OUT-OF-DOMAIN dataset ---")
    val_out_of_domain_targets = get_target_counts(out_of_domain_val_size, TARGET_DISTRIBUTION)

    val_out_of_domain_set = curate_single_dataset(
        val_out_of_domain_targets,
        out_of_domain_successful_cache, out_of_domain_candidates,
        out_of_domain_dbs,
        state_components,
        initial_set=curated_val_ood,
        locked_ids=locked_ids
    )

    # --- Save final datasets ---
    _save_datasets(output_dir, train_set, val_in_domain_set, val_out_of_domain_set)

    logger.info("\nData curation complete. The final datasets are ready for use.")


def _initialize_state_components(
    config_file: str,
    prompt_template: str,
    model_name: str,
    model_type: str,
    api_delay_seconds: int,
    individual_results_dir: str,
) -> tuple:
    """Initialize config, LLM, database, and prompt components.

    Args:
        config_file: Path to the config.ini file.
        prompt_template: Prompt template name (stem without .txt) inside prompts/.
        model_name: Model deployment name used in LLM API calls.
        model_type: LLM backend type identifier (e.g. 'gpt4o').
        api_delay_seconds: Base delay in seconds between API calls for rate limiting.
        individual_results_dir: Directory for per-question inference cache files.

    Returns:
        (state_components, config_paths) where state_components is None on
        LLM/DB init failure and config_paths always contains the resolved
        bird training paths from config.ini.
    """
    config = ConfigLoader(config_file)
    bird_train_paths = config.get_bird_train_dataset_paths()

    try:
        db_settings = config.get_database_train_settings()
        conn_settings = config.get_azure_openai_connection_settings()
        deploy_names = config.get_azure_openai_deployment_names()
        llm_parameters = config.get_llm_parameters()

        llm_instance = initialize_azure_openai_llm(conn_settings, deploy_names, llm_parameters)
        schema_loader = BirdSchemaLoader(bird_train_paths['train_tables_json_path'])
        db_manager = DatabaseManager(db_settings, bird_schema_loader=schema_loader)
        prompt_template_string = load_prompt_template(prompt_template)
        state_components = {
            'llm': llm_instance, 'db_manager': db_manager,
            'prompt_template_string': prompt_template_string,
            'model_name': model_name, 'model_type': model_type,
            'individual_results_dir': individual_results_dir, 'api_delay_seconds': api_delay_seconds
        }
        return state_components, bird_train_paths
    except Exception as e:
        logger.error(f"Initialization of LLM/DB components failed: {e}. Inference will be skipped.")
        return None, bird_train_paths


def _load_validated_data(input_file: str) -> list | None:
    """Load and validate the input JSON dataset. Returns None on failure."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: The input file '{input_file}' was not found.")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error: The file '{input_file}' is not a valid JSON file.")
        return None

    if not isinstance(full_data, list):
        logger.error(f"Error: Expected a JSON array, got {type(full_data).__name__}.")
        return None

    # Assign question_id if missing (BIRD raw data doesn't include one)
    missing_id_count = 0
    for idx, record in enumerate(full_data):
        if 'question_id' not in record:
            record['question_id'] = idx
            missing_id_count += 1
    if missing_id_count:
        logger.info(f"Assigned question_id to {missing_id_count} records that were missing it.")

    required_keys = ['question_id', 'db_id', 'SQL', 'level']
    validated_data = [
        r for r in full_data
        if all(k in r for k in required_keys) and r.get('level')
    ]

    if len(validated_data) == 0 and len(full_data) > 0:
        # Diagnostic: figure out why records are failing validation
        sample = full_data[0]
        missing = [k for k in required_keys if k not in sample]
        logger.error(
            f"0 of {len(full_data)} records passed validation. "
            f"First record keys: {sorted(sample.keys())}. "
            f"Missing required keys: {missing or 'none'}. "
            f"level value: {repr(sample.get('level'))}"
        )

    logger.info(f"Loaded {len(validated_data)} valid records from {len(full_data)} total in the dataset.")
    return validated_data


def _load_caches(validated_data: list, individual_results_dir: str) -> tuple:
    """Classify validated records into successful cache, failed cache, and candidates
    by checking the individual results directory.
    """
    successful_cache = []
    failed_cache = []
    candidate_pool = []

    # Optimization: Pre-fetch existing filenames to avoid thousands of syscalls
    existing_files = set(os.listdir(individual_results_dir))

    def load_cache_entry(sample):
        filename = f"{sample['question_id']}.json"
        if filename in existing_files:
            filepath = os.path.join(individual_results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    result = json.load(f)
                return result, "loaded"
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Corrupted cache file found: {filepath}. Error: {e}")
                return sample, "error"
        return sample, "missing"

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(load_cache_entry, validated_data), total=len(validated_data), desc="Checking existing results"))

    for data, status in results:
        if status == "loaded":
            if is_successful_inference(data):
                successful_cache.append(data)
            else:
                failed_cache.append(data)
        else:
            candidate_pool.append(data)

    logger.info(f"Found {len(successful_cache)} records with successful cached inferences.")
    logger.info(f"Found {len(failed_cache)} records with failed or corrupted cached inferences.")
    logger.info(f"Starting with {len(candidate_pool)} records that need new inference.")

    return successful_cache, failed_cache, candidate_pool


def _split_databases(validated_data: list, db_split_ratio: float) -> tuple:
    """Split unique database IDs into in-domain and out-of-domain sets."""
    all_db_ids = sorted(list(set(s['db_id'] for s in validated_data)))
    random.shuffle(all_db_ids)
    split_point = int(len(all_db_ids) * db_split_ratio)
    in_domain_dbs = set(all_db_ids[:split_point])
    out_of_domain_dbs = set(all_db_ids[split_point:])

    logger.info(f"Divided {len(all_db_ids)} unique databases into two pools:")
    logger.info(f"- In-Domain Databases: {len(in_domain_dbs)} databases")
    logger.info(f"- Out-of-Domain Databases: {len(out_of_domain_dbs)} databases")

    return in_domain_dbs, out_of_domain_dbs


def _load_existing_json(output_dir: str, filename: str) -> list:
    """Load an existing JSON file for resumability, filtering invalid prompt records."""
    path = Path(output_dir) / filename
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(data, list):
                filtered_data = [
                    r for r in data
                    if r.get("generation_status") == "Success"
                ]
                if len(filtered_data) < len(data):
                    logger.warning(
                        "Removed %d records without successful generation from %s",
                        len(data) - len(filtered_data), filename,
                    )
                return filtered_data
            return data
        except json.JSONDecodeError as e:
            logger.warning("The existing file '%s' is corrupted or not valid JSON. Error: %s", path, e)
            return []
    return []


def _save_datasets(output_dir: str, train_set: list, val_in_domain_set: list, val_out_of_domain_set: list) -> None:
    """Save the final curated datasets to JSON files."""
    logger.info("\nSaving final curated datasets to files...")
    base = Path(output_dir)

    for filename, dataset in [
        ('train.json', train_set),
        ('val_in_domain.json', val_in_domain_set),
        ('val_out_of_domain.json', val_out_of_domain_set),
    ]:
        dest = base / filename
        dest.write_text(json.dumps(dataset, indent=4), encoding='utf-8')
        logger.info("Saved %s (%d samples) to %s", filename, len(dataset), dest)
