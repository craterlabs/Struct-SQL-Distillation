#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BIRD Dataset Schema Loader.

Loads and provides structured schema metadata from BIRD benchmark's
dev_tables.json for use in text-to-SQL prompt construction.

@author: Khushboo Thaker, Yony Bresler, Khalid Eidoo
@license: Crater Labs (C)
"""

import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BirdSchemaLoaderError(Exception):
    """Exception raised for BIRD schema loading errors."""
    pass


class BirdSchemaLoader:
    """Loads and formats database schema metadata from BIRD's dev_tables.json.

    Abstracts schema loading to provide flexible formatting (e.g., schema-only
    or schema-with-descriptions) for different LLM prompt strategies.

    Attributes:
        db_schemas_metadata: Dict mapping db_id to schema metadata.
    """

    def __init__(self, dev_tables_json_path: str):
        """Initialize loader with path to dev_tables.json.

        Args:
            dev_tables_json_path: Path to BIRD's dev_tables.json file.

        Raises:
            BirdSchemaLoaderError: If file not found or invalid JSON.
        """
        self.dev_tables_json_path = dev_tables_json_path
        self.db_schemas_metadata: Dict[str, Dict[str, Any]] = {}
        self._load_schemas()

    def _load_schemas(self):
        """Load and parse dev_tables.json into internal schema metadata."""
        if not os.path.exists(self.dev_tables_json_path):
            raise BirdSchemaLoaderError(f"BIRD dev_tables.json file not found at: {self.dev_tables_json_path}")
        if not os.path.isfile(self.dev_tables_json_path):
            raise BirdSchemaLoaderError(f"BIRD dev_tables.json path is not a file: {self.dev_tables_json_path}")

        try:
            with open(self.dev_tables_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for db_info in data:
                db_id = db_info.get('db_id')
                if not db_id:
                    logger.warning(f"Skipping entry in {self.dev_tables_json_path} due to missing 'db_id'.")
                    continue
                self.db_schemas_metadata[db_id] = db_info

            logger.info(f"Loaded schema metadata for {len(self.db_schemas_metadata)} databases.")
        except json.JSONDecodeError as e:
            raise BirdSchemaLoaderError(f"Error decoding JSON from {self.dev_tables_json_path}: {e}") from e
        except Exception as e:
            raise BirdSchemaLoaderError(f"Unexpected error loading BIRD schema: {e}") from e

    def get_db_schema_metadata(self, db_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve schema metadata for a database.

        Args:
            db_id: The database identifier.

        Returns:
            Schema metadata dict, or None if not found.
        """
        return self.db_schemas_metadata.get(db_id)
