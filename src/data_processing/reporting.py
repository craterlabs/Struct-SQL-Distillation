#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reporting utilities for data curation.

Provides functions for printing distribution summaries, target vs. filled
comparisons, and database coverage reports.

@author: Khushboo Thaker, Yony Bresler, Khalid Eidoo
@license: Crater Labs (C)
"""

from collections import Counter
from typing import List, Dict, Any, Set


def print_distribution_summary(dataset_name: str, dataset: List[Dict[str, Any]]) -> None:
    """Print a table showing how samples are distributed across SQL complexity levels.

    Args:
        dataset_name: Human-readable label shown in the table header (e.g. "Train Set").
        dataset: List of sample dicts, each expected to contain a 'level' key.
    """
    total_samples = len(dataset)
    if not dataset:
        print(f"\n--- {dataset_name} Distribution ---")
        print("Dataset is empty.")
        return

    counts = Counter(sample.get('level', 'Other/Uncategorized') for sample in dataset)
    print(f"\n--- {dataset_name} Distribution ---")
    print(f"{'Category':<45} | {'Count':>7} | {'Percentage':>12}")
    print("-" * 75)
    for category, count in sorted(counts.items()):
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"{category:<45} | {count:>7} | {percentage:11.2f}%")
    print("-" * 75)
    print(f"{'Total':<45} | {total_samples:>7} | {'100.00%':>12}")
    print("-" * 75)


def print_target_vs_filled_summary(dataset_name: str, filled_dataset: List[Dict[str, Any]], target_counts: Dict[str, int]) -> None:
    """Print a table comparing how many samples were collected vs. the target for each level.

    Args:
        dataset_name: Human-readable label for the dataset (e.g. "In-Domain Validation").
        filled_dataset: List of sample dicts that were actually collected.
        target_counts: Mapping of complexity level to the integer target for that level.
    """
    total_filled = len(filled_dataset)
    total_target = sum(target_counts.values())
    filled_counts = Counter(sample.get('level', 'Other/Uncategorized') for sample in filled_dataset)

    print(f"\n--- {dataset_name} (Target vs. Filled) ---")
    print(f"{'Category':<45} | {'Target':>7} | {'Filled':>7} | {'% Filled':>12}")
    print("-" * 75)
    for category, target in sorted(target_counts.items()):
        filled = filled_counts.get(category, 0)
        percentage_filled = (filled / target) * 100 if target > 0 else 0
        print(f"{category:<45} | {target:>7} | {filled:>7} | {percentage_filled:11.2f}%")
    print("-" * 75)

    total_percentage_achieved = (total_filled / total_target) * 100 if total_target > 0 else 0
    print(f"{'Total':<45} | {total_target:>7} | {total_filled:>7} | {total_percentage_achieved:11.2f}%")
    print("-" * 75)


def print_database_coverage_summary(split_name: str, dataset: List[Dict[str, Any]], expected_db_ids: Set[str]) -> None:
    """Print a per-database coverage table showing which databases have zero samples.

    Args:
        split_name: Label for the data split (e.g. "Out-of-Domain Validation").
        dataset: List of sample dicts, each expected to contain a 'db_id' key.
        expected_db_ids: Complete set of database IDs that should be represented.
    """
    actual_counts = Counter(s.get('db_id') for s in dataset)

    print(f"\n{'='*60}")
    print(f" DATABASE COVERAGE DETAILS: {split_name}")
    print(f"{'='*60}")
    print(f"{'Database Name':<40} | {'Samples':>8}")
    print(f"{'-'*40}-|-{'-'*8}")

    # Sort by expected DBs alphabetically so it's easy to read
    for db_id in sorted(expected_db_ids):
        count = actual_counts.get(db_id, 0)
        status = " " if count > 0 else "❌ MISSING"
        print(f"{db_id:<40} | {count:>8} {status}")

    print(f"{'-'*60}")
    covered = len([c for c in actual_counts.values() if c > 0])
    total = len(expected_db_ids)
    pct = (covered / total * 100) if total > 0 else 0
    print(f"Summary: {covered}/{total} Databases Covered ({pct:.1f}%)")
    print(f"{'='*60}\n")
