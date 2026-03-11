#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-stage data curation strategies.

Implements the stratified sampling pipeline for both in-domain (train + validation)
and out-of-domain (single dataset) curation. Each stage is a standalone function:

    Stage 1: Stratified fill from successful cache
    Stage 2: Targeted inference on unfilled categories
    Stage 3: Backfill from remaining cache + database coverage (3.1)
    Stage 4: Active inference on candidate leftovers

@author: Khushboo Thaker, Yony Bresler, Khalid Eidoo
@license: Crater Labs (C)
"""

import logging
import random
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Set

from tqdm.auto import tqdm

from src.data_processing.inference import run_inference_for_sample
from src.data_processing.reporting import (
    print_target_vs_filled_summary,
    print_database_coverage_summary,
)
from src.data_processing.utils import is_successful_inference

logger = logging.getLogger(__name__)

# DB-Coverage Tuning Parameters (used in stage 3.1)
# Fraction of the train target size used as an extra budget for DB coverage samples.
_DB_COVERAGE_BUDGET_RATIO: float = 0.2
# Minimum number of samples per database before it's considered "covered".
_DB_COVERAGE_MIN_DENSITY: int = 2


def _stage1_fill_from_cache_dual(
    train_set: List[Dict], val_set: List[Dict],
    train_counts: Counter, val_counts: Counter,
    train_targets: Dict[str, int], val_targets: Dict[str, int],
    successful_cache: List[Dict], locked_ids: Set[str],
):
    """Stage 1: Stratified fill from successful cache into train + val sets."""
    logger.info("Starting Stage 1: Stratified filling from in-domain successful cache.")
    cache_by_level = defaultdict(list)
    for sample in successful_cache:
        q_id = str(sample.get('question_id'))
        if q_id in locked_ids:
            continue
        cache_by_level[sample.get('level')].append(sample)

    for category, samples in cache_by_level.items():
        random.shuffle(samples)
        while ((train_counts[category] < train_targets.get(category, 0) or
                val_counts[category] < val_targets.get(category, 0)) and samples):

            train_filled_ratio = train_counts[category] / train_targets[category] if train_targets[category] > 0 else 1
            val_filled_ratio = val_counts[category] / val_targets[category] if val_targets[category] > 0 else 1

            if train_filled_ratio <= val_filled_ratio and train_counts[category] < train_targets[category] and len(train_set) < sum(train_targets.values()):
                train_set.append(samples.pop(0))
                train_counts[category] += 1
            elif val_counts[category] < val_targets[category] and len(val_set) < sum(val_targets.values()):
                val_set.append(samples.pop(0))
                val_counts[category] += 1
            else:
                break

    print_target_vs_filled_summary("In-Domain Training Set (After Stage 1)", train_set, train_targets)
    print_target_vs_filled_summary("In-Domain Validation Set (After Stage 1)", val_set, val_targets)


def _stage2_targeted_inference_dual(
    train_set: List[Dict], val_set: List[Dict],
    train_counts: Counter, val_counts: Counter,
    train_targets: Dict[str, int], val_targets: Dict[str, int],
    candidates: List[Dict], locked_ids: Set[str],
    state_components: Dict[str, Any],
):
    """Stage 2: Targeted inference on candidates for unfilled categories."""
    unfilled_categories = {
        cat for cat, target in train_targets.items() if train_counts[cat] < target
    }.union(
        {cat for cat, target in val_targets.items() if val_counts[cat] < target}
    )

    if not (state_components and unfilled_categories):
        print_target_vs_filled_summary("In-Domain Training Set (After Stage 2)", train_set, train_targets)
        print_target_vs_filled_summary("In-Domain Validation Set (After Stage 2)", val_set, val_targets)
        return

    logger.info(f"In-domain targets for {unfilled_categories} are not fully filled. Starting targeted inference...")

    pending_counts = {c: (train_targets[c] - train_counts[c]) + (val_targets[c] - val_counts[c]) for c in unfilled_categories}
    logger.info(f"Pending counts by category: {pending_counts}")

    inference_candidates = [
        s for s in candidates
        if s.get('level') in unfilled_categories and str(s.get('question_id')) not in locked_ids
    ]
    random.shuffle(inference_candidates)

    with tqdm(total=len(inference_candidates), desc="Running Inference (In-Domain)") as pbar:
        for candidate in inference_candidates:
            if len(train_set) >= sum(train_targets.values()) and len(val_set) >= sum(val_targets.values()):
                break

            inferred_result = run_inference_for_sample(candidate, state_components)
            pbar.update(1)

            if is_successful_inference(inferred_result):
                category = inferred_result.get('level')
                train_needed = train_counts[category] < train_targets.get(category, 0)
                val_needed = val_counts[category] < val_targets.get(category, 0)

                if train_needed or val_needed:
                    train_filled_ratio = train_counts[category] / train_targets[category] if train_targets[category] > 0 else 1
                    val_filled_ratio = val_counts[category] / val_targets[category] if val_targets[category] > 0 else 1

                    if train_filled_ratio <= val_filled_ratio and train_counts[category] < train_targets[category]:
                        train_set.append(inferred_result)
                        train_counts[category] += 1
                    elif val_counts[category] < val_targets[category]:
                        val_set.append(inferred_result)
                        val_counts[category] += 1

    print_target_vs_filled_summary("In-Domain Training Set (After Stage 2)", train_set, train_targets)
    print_target_vs_filled_summary("In-Domain Validation Set (After Stage 2)", val_set, val_targets)


def _stage3_backfill_and_db_coverage_dual(
    train_set: List[Dict], val_set: List[Dict],
    train_targets: Dict[str, int], val_targets: Dict[str, int],
    successful_cache: List[Dict], candidates: List[Dict],
    in_domain_db_ids: Set[str], locked_ids: Set[str],
    state_components: Dict[str, Any],
):
    """Stage 3 + 3.1: Backfill from remaining cache, then ensure multi-sample DB coverage."""
    # --- Stage 3: Backfilling from remaining successful cache ---
    if len(train_set) < sum(train_targets.values()) or len(val_set) < sum(val_targets.values()):
        used_q_ids = {str(s.get('question_id')) for s in train_set + val_set}
        used_q_ids.update(locked_ids)
        remaining_cache = [s for s in successful_cache if str(s.get('question_id')) not in used_q_ids]
        random.shuffle(remaining_cache)

        logger.info(f"Starting backfilling Stage 3 for IN-DOMAIN with {len(remaining_cache)} remaining successful cache samples.")

        # Backfill validation set first to guarantee its target size
        samples_needed_val = sum(val_targets.values()) - len(val_set)
        samples_to_add_val = remaining_cache[:samples_needed_val]
        val_set.extend(samples_to_add_val)
        remaining_cache = remaining_cache[len(samples_to_add_val):]

        # Backfill training set with the rest
        samples_needed_train = sum(train_targets.values()) - len(train_set)
        samples_to_add_train = remaining_cache[:samples_needed_train]
        train_set.extend(samples_to_add_train)

    print_target_vs_filled_summary("In-Domain Training Set (After Stage 3)", train_set, train_targets)
    print_target_vs_filled_summary("In-Domain Validation Set (After Stage 3)", val_set, val_targets)

    # Build used IDs set for Stage 3.1
    used_q_ids = {str(s.get('question_id')) for s in train_set + val_set}
    used_q_ids.update(locked_ids)

    # --- Stage 3.1: Multi-Sample Database Coverage ---
    # Budget: allow up to 20% extra samples beyond train target for DB diversity
    total_train_target = sum(train_targets.values())
    coverage_budget = max(0, int(total_train_target * _DB_COVERAGE_BUDGET_RATIO) - max(0, len(train_set) - total_train_target))

    if coverage_budget <= 0:
        logger.info("Skipping Stage 3.1: No remaining budget for DB coverage samples.")
    else:
        # Scale target density based on available budget and DB count
        target_density = _DB_COVERAGE_MIN_DENSITY if coverage_budget >= len(in_domain_db_ids) else 1
        logger.info(f"Starting Stage 3.1: Ensuring DB coverage (density={target_density}, budget={coverage_budget})...")

        combined_samples = train_set + val_set
        db_counts = Counter(s.get('db_id') for s in combined_samples)
        underfilled_dbs = {db_id for db_id in in_domain_db_ids if db_counts[db_id] < target_density}

        if underfilled_dbs:
            logger.info(f"Found {len(underfilled_dbs)} databases with fewer than {target_density} samples.")

            coverage_cands = [
                s for s in candidates
                if s['db_id'] in underfilled_dbs and str(s['question_id']) not in used_q_ids
            ]
            random.shuffle(coverage_cands)

            added = 0
            for cand in coverage_cands:
                if added >= coverage_budget:
                    logger.info(f"Stage 3.1 budget exhausted ({added}/{coverage_budget}).")
                    break

                db_id = cand['db_id']
                if db_counts[db_id] >= target_density:
                    continue

                res = run_inference_for_sample(cand, state_components)
                if is_successful_inference(res):
                    train_set.append(res)
                    db_counts[db_id] += 1
                    used_q_ids.add(str(res['question_id']))
                    added += 1

                    if db_counts[db_id] >= target_density:
                        underfilled_dbs.remove(db_id)
                        logger.info(f"Reached target density for DB: {db_id}")

            logger.info(f"Stage 3.1 added {added} samples for DB coverage.")
        else:
            logger.info("All databases already meet the minimum density requirement.")

    print_database_coverage_summary("In-Domain (Train)", train_set, in_domain_db_ids)
    print_database_coverage_summary("In-Domain (Validation)", val_set, in_domain_db_ids)
    print_target_vs_filled_summary("In-Domain Training Set (After Stage 3.1)", train_set, train_targets)
    print_target_vs_filled_summary("In-Domain Validation Set (After Stage 3.1)", val_set, val_targets)


def _stage4_active_inference_dual(
    train_set: List[Dict], val_set: List[Dict],
    train_targets: Dict[str, int], val_targets: Dict[str, int],
    candidates: List[Dict], locked_ids: Set[str],
    state_components: Dict[str, Any],
):
    """Stage 4: Active inference on remaining candidate leftovers."""
    if len(train_set) >= sum(train_targets.values()) and len(val_set) >= sum(val_targets.values()):
        print_target_vs_filled_summary("In-Domain Training Set (After Stage 4)", train_set, train_targets)
        print_target_vs_filled_summary("In-Domain Validation Set (After Stage 4)", val_set, val_targets)
        return

    used_q_ids = {str(s.get('question_id')) for s in train_set + val_set}
    used_q_ids.update(locked_ids)
    remaining_candidates = [s for s in candidates if str(s.get('question_id')) not in used_q_ids]
    random.shuffle(remaining_candidates)
    logger.info(f"Starting backfilling Stage 4 for IN-DOMAIN with {len(remaining_candidates)} remaining candidate samples. This is now an active inference step.")

    with tqdm(total=len(remaining_candidates), desc="Running Inference (In-Domain Backfill)") as pbar:
        for candidate in remaining_candidates:
            if len(train_set) >= sum(train_targets.values()) and len(val_set) >= sum(val_targets.values()):
                break

            inferred_result = run_inference_for_sample(candidate, state_components)
            pbar.update(1)

            if is_successful_inference(inferred_result):
                # Prefer filling the smaller, more critical set first
                if len(val_set) < sum(val_targets.values()):
                    val_set.append(inferred_result)
                elif len(train_set) < sum(train_targets.values()):
                    train_set.append(inferred_result)

    print_target_vs_filled_summary("In-Domain Training Set (After Stage 4)", train_set, train_targets)
    print_target_vs_filled_summary("In-Domain Validation Set (After Stage 4)", val_set, val_targets)


def _stage1_fill_from_cache_single(
    final_set: List[Dict], filled_counts: Counter,
    target_counts: Dict[str, int], target_size: int,
    successful_cache: List[Dict], locked_ids: Set[str],
):
    """Stage 1: Stratified fill from successful cache into a single set."""
    logger.info("Starting Stage 1: Stratified filling from cache.")
    cache_by_level = defaultdict(list)
    for sample in successful_cache:
        if str(sample.get('question_id')) in locked_ids:
            continue
        cache_by_level[sample.get('level')].append(sample)

    for category, samples in cache_by_level.items():
        random.shuffle(samples)
        while filled_counts[category] < target_counts.get(category, 0) and samples:
            if len(final_set) >= target_size:
                break
            final_set.append(samples.pop(0))
            filled_counts[category] += 1

    print_target_vs_filled_summary("Out-of-Domain Set (After Stage 1)", final_set, target_counts)


def _stage2_targeted_inference_single(
    final_set: List[Dict], filled_counts: Counter,
    target_counts: Dict[str, int], target_size: int,
    candidate_pool: List[Dict],
    state_components: Dict[str, Any],
):
    """Stage 2: Targeted inference on candidates for unfilled categories."""
    unfilled_categories = {
        cat for cat, target in target_counts.items() if filled_counts[cat] < target
    }

    if not (state_components and unfilled_categories and len(final_set) < target_size):
        print_target_vs_filled_summary("Out-of-Domain Set (After Stage 2)", final_set, target_counts)
        return

    logger.info(f"Targets for {unfilled_categories} are not fully filled. Starting targeted inference...")

    pending_counts = {c: target_counts[c] - filled_counts[c] for c in unfilled_categories}
    logger.info(f"Pending counts by category: {pending_counts}")

    inference_candidates = [
        s for s in candidate_pool if s.get('level') in unfilled_categories
    ]
    random.shuffle(inference_candidates)

    with tqdm(total=len(inference_candidates), desc="Running Inference") as pbar:
        for candidate in inference_candidates:
            if len(final_set) >= target_size:
                break

            inferred_result = run_inference_for_sample(candidate, state_components)
            pbar.update(1)

            if is_successful_inference(inferred_result):
                category = inferred_result.get('level')
                if filled_counts[category] < target_counts[category]:
                    final_set.append(inferred_result)
                    filled_counts[category] += 1

    print_target_vs_filled_summary("Out-of-Domain Set (After Stage 2)", final_set, target_counts)


def _stage3_backfill_and_db_coverage_single(
    final_set: List[Dict],
    target_counts: Dict[str, int],
    successful_cache: List[Dict], candidate_pool: List[Dict],
    out_of_domain_db_ids: Set[str], locked_ids: Set[str],
    state_components: Dict[str, Any],
):
    """Stage 3 + 3.1: Backfill from remaining cache, then ensure OOD DB coverage."""
    target_size = sum(target_counts.values())

    # --- Stage 3: Backfilling from remaining successful cache ---
    if len(final_set) < target_size:
        used_q_ids = {str(s.get('question_id')) for s in final_set}
        remaining_cache = [s for s in successful_cache if str(s.get('question_id')) not in used_q_ids]
        random.shuffle(remaining_cache)

        logger.info(f"Starting backfilling Stage 3 with {len(remaining_cache)} remaining successful cache samples.")

        samples_needed = target_size - len(final_set)
        samples_to_add = remaining_cache[:samples_needed]
        final_set.extend(samples_to_add)

    print_target_vs_filled_summary("Out-of-Domain Set (After Stage 3)", final_set, target_counts)

    # --- Stage 3.1: OOD Database Coverage ---
    # Budget: allow up to 20% extra samples beyond target for DB coverage
    coverage_budget = max(0, int(target_size * _DB_COVERAGE_BUDGET_RATIO) - max(0, len(final_set) - target_size))

    if coverage_budget <= 0:
        logger.info("Skipping Stage 3.1: No remaining budget for OOD DB coverage.")
    else:
        logger.info(f"Starting Stage 3.1: Ensuring Database Coverage for Out-of-Domain (budget={coverage_budget})...")
        covered_ood_dbs = {s.get('db_id') for s in final_set}
        missing_ood_dbs = set(out_of_domain_db_ids) - covered_ood_dbs

        if missing_ood_dbs:
            coverage_cands = [s for s in candidate_pool if s['db_id'] in missing_ood_dbs and str(s['question_id']) not in locked_ids]
            random.shuffle(coverage_cands)
            added = 0
            for cand in coverage_cands:
                if added >= coverage_budget:
                    logger.info(f"Stage 3.1 OOD budget exhausted ({added}/{coverage_budget}).")
                    break
                if cand['db_id'] not in missing_ood_dbs: continue
                res = run_inference_for_sample(cand, state_components)
                if is_successful_inference(res):
                    final_set.append(res)
                    missing_ood_dbs.remove(cand['db_id'])
                    locked_ids.add(str(res['question_id']))
                    added += 1
                    logger.info(f"Covered missing OOD database: {cand['db_id']}")

            logger.info(f"Stage 3.1 OOD added {added} samples for DB coverage.")

    print_database_coverage_summary("Out-of-Domain", final_set, out_of_domain_db_ids)


def _stage4_active_inference_single(
    final_set: List[Dict],
    target_counts: Dict[str, int],
    candidate_pool: List[Dict],
    state_components: Dict[str, Any],
):
    """Stage 4: Active inference on remaining candidate leftovers."""
    target_size = sum(target_counts.values())

    if len(final_set) >= target_size:
        print_target_vs_filled_summary("Out-of-Domain Set (After Stage 4)", final_set, target_counts)
        return

    used_q_ids = {str(s.get('question_id')) for s in final_set}
    remaining_candidates = [s for s in candidate_pool if str(s.get('question_id')) not in used_q_ids]
    random.shuffle(remaining_candidates)
    logger.info(f"Starting backfilling Stage 4 with {len(remaining_candidates)} remaining samples. This is now an active inference step.")

    with tqdm(total=len(remaining_candidates), desc="Running Inference (OOD Backfill)") as pbar:
        for sample in remaining_candidates:
            if len(final_set) >= target_size:
                break

            inferred_result = run_inference_for_sample(sample, state_components)
            pbar.update(1)

            if is_successful_inference(inferred_result):
                final_set.append(inferred_result)

    print_target_vs_filled_summary("Out-of-Domain Set (After Stage 4)", final_set, target_counts)


def curate_in_domain_datasets(
    train_targets: Dict[str, int],
    val_in_domain_targets: Dict[str, int],
    in_domain_successful_cache: List[Dict[str, Any]],
    in_domain_candidates: List[Dict[str, Any]],
    in_domain_db_ids: Set[str],
    state_components: Dict[str, Any],
    existing_train: List[Dict[str, Any]] = None,
    existing_val: List[Dict[str, Any]] = None,
    locked_ids: Set[str] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Curate in-domain training and validation datasets through a 4-stage pipeline.

    Stages: (1) fill from cache, (2) targeted inference on unfilled categories,
    (3) backfill + DB coverage, (4) active inference on remaining leftovers.

    Args:
        train_targets: Per-complexity-level target counts for the training set.
        val_in_domain_targets: Per-level target counts for the in-domain validation set.
        in_domain_successful_cache: Previously successful inference results for in-domain samples.
        in_domain_candidates: Full pool of in-domain samples available for inference.
        in_domain_db_ids: Set of database IDs assigned to the in-domain split.
        state_components: Shared inference state (LLM, db_manager, prompt template, paths).
        existing_train: Existing training records to resume from (default: empty list).
        existing_val: Existing validation records to resume from (default: empty list).
        locked_ids: Question IDs already committed to another split; excluded from sampling.

    Returns:
        Tuple of (train_set, val_set) as lists of curated sample dicts.
    """
    train_set = existing_train if existing_train else []
    val_set = existing_val if existing_val else []
    train_counts = Counter(sample.get('level') for sample in train_set)
    val_counts = Counter(sample.get('level') for sample in val_set)
    locked_ids = locked_ids if locked_ids else set()

    _stage1_fill_from_cache_dual(
        train_set, val_set, train_counts, val_counts,
        train_targets, val_in_domain_targets,
        in_domain_successful_cache, locked_ids,
    )

    _stage2_targeted_inference_dual(
        train_set, val_set, train_counts, val_counts,
        train_targets, val_in_domain_targets,
        in_domain_candidates, locked_ids, state_components,
    )

    _stage3_backfill_and_db_coverage_dual(
        train_set, val_set,
        train_targets, val_in_domain_targets,
        in_domain_successful_cache, in_domain_candidates,
        in_domain_db_ids, locked_ids, state_components,
    )

    _stage4_active_inference_dual(
        train_set, val_set,
        train_targets, val_in_domain_targets,
        in_domain_candidates, locked_ids, state_components,
    )

    return train_set, val_set


def curate_single_dataset(
    target_counts: Dict[str, int],
    successful_cache: List[Dict[str, Any]],
    candidate_pool: List[Dict[str, Any]],
    out_of_domain_db_ids: Set[str],
    state_components: Dict[str, Any],
    initial_set: List[Dict[str, Any]] = None,
    locked_ids: Set[str] = None
) -> List[Dict[str, Any]]:
    """Curate a single dataset (e.g. out-of-domain validation) through a 4-stage pipeline.

    Stages: (1) fill from cache, (2) targeted inference on unfilled categories,
    (3) backfill + OOD DB coverage, (4) active inference on remaining leftovers.

    Args:
        target_counts: Per-complexity-level target counts for the dataset.
        successful_cache: Previously successful inference results available for reuse.
        candidate_pool: Full pool of samples eligible for inference.
        out_of_domain_db_ids: Set of database IDs assigned to the out-of-domain split.
        state_components: Shared inference state (LLM, db_manager, prompt template, paths).
        initial_set: Existing records to resume from (default: empty list).
        locked_ids: Question IDs already committed to another split; excluded from sampling.

    Returns:
        Curated list of sample dicts meeting the target counts.
    """
    final_set = list(initial_set) if initial_set else []
    locked_ids = locked_ids if locked_ids else set()
    filled_counts = Counter(sample.get('level') for sample in final_set)
    target_size = sum(target_counts.values())

    _stage1_fill_from_cache_single(
        final_set, filled_counts,
        target_counts, target_size,
        successful_cache, locked_ids,
    )

    _stage2_targeted_inference_single(
        final_set, filled_counts,
        target_counts, target_size,
        candidate_pool, state_components,
    )

    _stage3_backfill_and_db_coverage_single(
        final_set, target_counts,
        successful_cache, candidate_pool,
        out_of_domain_db_ids, locked_ids, state_components,
    )

    _stage4_active_inference_single(
        final_set, target_counts,
        candidate_pool, state_components,
    )

    return final_set
