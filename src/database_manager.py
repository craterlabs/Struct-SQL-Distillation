#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Manager for BIRD Benchmark.

Manages SQLite database connections, query execution, schema generation,
and result comparison for text-to-SQL evaluation.

@author: Khushboo Thaker, Yony Bresler, Khalid Eidoo
@license: Crater Labs (C)
"""

import os
import glob
import logging
import sqlite3
import pandas as pd
import sqlparse
import re
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from func_timeout import func_timeout, FunctionTimedOut

logger = logging.getLogger(__name__)


class DatabaseManagerError(Exception):
    """Custom exception for database manager related errors."""
    pass


class CompareResult(NamedTuple):
    """Result of comparing two SQL query executions."""
    gold_exec_status: bool
    predict_exec_status: bool
    match_status: bool
    num_gold_rows: int
    num_predicted_rows: int
    num_gold_cols: int
    num_predicted_cols: int
    gold_cols: List[str]
    predicted_cols: List[str]
    error_msg: Optional[str]


def nice_look_table(column_names: list, values: list) -> str:
    """Format table data into an aligned string representation.

    Args:
        column_names: List of column header names.
        values: List of row tuples containing cell values.

    Returns:
        Formatted string with right-aligned columns and header.
    """
    rows = []
    # Ensure all data (including column names) is stringified for width calculation
    all_data_for_widths = [[str(item) for item in row] for row in values] + [[str(col) for col in column_names]]
    widths = [max(len(col_val) for col_val in [row[i] for row in all_data_for_widths]) for i in range(len(column_names))]

    header = ''.join(f'{column.rjust(width)} ' for column, width in zip(column_names, widths))
    for value in values:
        row = ''.join(f'{str(v).rjust(width)} ' for v, width in zip(value, widths))
        rows.append(row)
    final_output = header + '\n' + "\n".join(rows)
    return final_output


class DatabaseManager:
    """Manages connections to multiple databases organized in a BIRD-like structure.

    Discovers DB file paths and executes queries directly using sqlite3.
    """

    _QUERY_TIMEOUT_SEC: int = 40

    def __init__(self, db_settings: Dict[str, Any], bird_schema_loader: Any = None):
        """Initialize the database manager and discover SQLite databases.

        Args:
            db_settings: Dict with 'db_type' (must be 'sqlite') and
                'database_root_path' pointing to the directory containing
                per-database subdirectories.
            bird_schema_loader: Optional BirdSchemaLoader instance for
                semantic schema generation. If None, DDL-only methods still
                work but semantic schema methods return placeholder strings.

        Raises:
            DatabaseManagerError: If settings are invalid, root path is
                missing, or no SQLite databases are found.
        """
        self.db_type = db_settings.get('db_type')
        self.database_root_path = db_settings.get('database_root_path')
        self.bird_schema_loader = bird_schema_loader

        # Stores db_id -> SQLite file path
        self.db_file_paths: Dict[str, str] = {}

        self._validate_init_settings()
        self._load_all_databases()

    def _preprocess_query_for_execution(self, query: str) -> str:
        """Preprocess a raw SQL query string for execution.

        Args:
            query: Raw SQL query, possibly with markdown blocks or comments.

        Returns:
            Cleaned SQL string ready for execution.
        """
        processed_query = query
        try:
            if not query or not query.strip():
                return ""

            processed_query = query.strip()

            # Strip markdown code fences: ```sql ... ``` or ``` ... ```
            processed_query = re.sub(r'^```(?:sql)?\s*', '', processed_query, flags=re.IGNORECASE)
            processed_query = re.sub(r'\s*```$', '', processed_query, flags=re.IGNORECASE).strip()

            if not processed_query.startswith("SELECT") and not processed_query.startswith("WITH"):
                processed_query = 'SELECT ' + processed_query

            # Remove SQL comments (both -- and /* */ styles)
            processed_query = re.sub(r'--.*$', '', processed_query, flags=re.MULTILINE).strip()
            processed_query = re.sub(r'/\*.*?\*/', '', processed_query, flags=re.DOTALL).strip()
        except Exception as e:
            logger.warning(f"[preprocessing failure]: '{processed_query[:50]}...': {e}. Using raw cleaned SQL.", exc_info=True)

        # Re-remove trailing semicolons
        processed_query = processed_query.rstrip(';')
        return processed_query

    def _validate_init_settings(self):
        """Internal method to validate initial settings."""
        if not self.db_type:
            raise DatabaseManagerError("Database type (db_type) is not specified in settings.")
        if self.db_type != 'sqlite':
            raise DatabaseManagerError(f"Unsupported database type: {self.db_type}. This DatabaseManager supports only 'sqlite'.")

        if not self.database_root_path:
            raise DatabaseManagerError("Database root path is not specified in settings.")
        if not os.path.exists(self.database_root_path):
            raise DatabaseManagerError(f"Database root path '{self.database_root_path}' does not exist.")
        if not os.path.isdir(self.database_root_path):
            raise DatabaseManagerError(f"Database root path '{self.database_root_path}' is not a directory.")
        if self.bird_schema_loader and not hasattr(self.bird_schema_loader, 'get_db_schema_metadata'):
            raise DatabaseManagerError("Provided bird_schema_loader does not have 'get_db_schema_metadata' method.")

    def _load_all_databases(self):
        """Scans the database_root_path and stores direct SQLite file paths."""
        logger.info(f"Scanning for databases in: {self.database_root_path}")
        db_folders = [f for f in os.listdir(self.database_root_path) if os.path.isdir(os.path.join(self.database_root_path, f))]
        if not db_folders:
            logger.warning(f"No database folders found in '{self.database_root_path}'.")

        for db_id in db_folders:
            db_folder_path = os.path.join(self.database_root_path, db_id)
            sqlite_files = glob.glob(os.path.join(db_folder_path, '*.sqlite'))
            if not sqlite_files:
                logger.warning(f"No .sqlite file found in database folder '{db_id}'. Skipping.")
                continue
            db_file_path = sqlite_files[0]
            self.db_file_paths[db_id] = db_file_path

        if not self.db_file_paths:
            raise DatabaseManagerError(f"No SQLite databases were successfully loaded from '{self.database_root_path}'.")

    def _get_db_file_path(self, db_id: str) -> str:
        """Internal helper to get the full file path for a database."""
        db_path = self.db_file_paths.get(db_id)
        if not db_path:
            raise DatabaseManagerError(f"SQLite database file path not found for db_id: {db_id}.")
        return db_path

    def get_ddl(self, db_id: str) -> str:
        """Retrieve CREATE TABLE DDL statements for a database.

        Args:
            db_id: Database identifier.

        Returns:
            Concatenated DDL statements as a string, or an inline SQL comment
            string on error.
        """
        db_file_path = self._get_db_file_path(db_id)

        full_schema_prompt_list = []
        conn = None
        try:
            conn = sqlite3.connect(f'file:{db_file_path}?mode=ro', uri=True)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()

            for table_info in tables:
                table_name = table_info[0]
                if table_name == 'sqlite_sequence':
                    continue

                cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
                create_prompt = cursor.fetchone()[0]
                full_schema_prompt_list.append(create_prompt)

            schema_prompt = "\n".join(full_schema_prompt_list)
            return schema_prompt
        except Exception as e:
            logger.error(f"Error generating DDL schema for '{db_id}': {e}", exc_info=True)
            return f"/* Error loading DDL schema for '{db_id}': {e} */"
        finally:
            if conn: conn.close()

    def execute_query(self, db_id: str, query: str) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame.

        Args:
            db_id: Database identifier.
            query: SQL query string to execute.

        Returns:
            Query results as a pandas DataFrame.

        Raises:
            DatabaseManagerError: If query is empty, times out, or fails execution.
        """
        db_file_path = self._get_db_file_path(db_id)
        timeout_sec = self._QUERY_TIMEOUT_SEC
        if not query or not query.strip():
            raise DatabaseManagerError("Cannot execute empty or whitespace-only query.")

        preprocessed_query = self._preprocess_query_for_execution(query)

        if not preprocessed_query:
            raise DatabaseManagerError("Preprocessed query is empty or whitespace-only after cleaning.")

        def _execute_sqlite_query_in_thread():
            conn = None
            try:
                conn = sqlite3.connect(f'file:{db_file_path}?mode=ro', uri=True)
                cursor = conn.cursor()
                cursor.execute(preprocessed_query)
                column_names = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                return column_names, rows
            finally:
                if conn:
                    conn.close()

        try:
            column_names_raw, rows_raw = func_timeout(timeout_sec, _execute_sqlite_query_in_thread)
        except FunctionTimedOut:
            raise DatabaseManagerError(f"Query execution timed out after {timeout_sec} seconds for db '{db_id}'.")
        except Exception as e:
            raise DatabaseManagerError(f"Error during query execution: {e}") from e

        seen_names = {}
        unique_column_names = []
        for name in column_names_raw:
            original_name = name
            count = seen_names.get(original_name, 0)
            if count > 0:
                unique_column_names.append(f"{original_name}_{count}")
            else:
                unique_column_names.append(original_name)
            seen_names[original_name] = count + 1

        df = pd.DataFrame(rows_raw, columns=unique_column_names)
        return df

    def execute_and_compare(self, db_id: str, query1: str, query2: str) -> CompareResult:
        """Execute two queries and compare results for set equality.

        Args:
            db_id: Database identifier.
            query1: First SQL query (typically gold/reference).
            query2: Second SQL query (typically predicted).

        Returns:
            CompareResult with execution statuses, match flag, row/col counts, and error info.
        """
        gold_exec_status = False
        predict_exec_status = False
        try:
            results1_df = self.execute_query(db_id, query1)
            gold_exec_status = True
        except DatabaseManagerError as e:
            logger.warning(f"Error during running query1 in execute_and_compare for db '{db_id}' for query {query1} - {e}")
            return CompareResult(gold_exec_status, predict_exec_status, False, 0, 0, 0, 0, [], [], None)

        try:
            results2_df = self.execute_query(db_id, query2)
            predict_exec_status = True
        except DatabaseManagerError as e:
            logger.warning(f"Error during running query2 in execute_and_compare for db '{db_id}' for query {query2} - {e}")
            return CompareResult(gold_exec_status, predict_exec_status, False, 0, 0, 0, 0, [], [], str(e))

        try:
            num_gold_rows = len(results1_df)
            num_gold_cols = len(results1_df.columns) if not results1_df.empty else 0
            gold_cols = list(results1_df.columns)
            num_predicted_rows = len(results2_df)
            num_predicted_cols = len(results2_df.columns) if not results2_df.empty else 0
            predicted_cols = list(results2_df.columns)
            results1_raw = [tuple(row) for row in results1_df.values.tolist()] if not results1_df.empty else []
            results2_raw = [tuple(row) for row in results2_df.values.tolist()] if not results2_df.empty else []
            match_status = set(results1_raw) == set(results2_raw)
            return CompareResult(
                gold_exec_status, predict_exec_status, match_status,
                num_gold_rows, num_predicted_rows, num_gold_cols, num_predicted_cols,
                gold_cols, predicted_cols, None,
            )
        except DatabaseManagerError as e:
            logger.warning(f"Error during comparison in execute_and_compare for db '{db_id}' for query {query2} - {e}")
            return CompareResult(gold_exec_status, False, False, 0, 0, 0, 0, [], [], str(e))

    def list_database_ids(self) -> List[str]:
        """Get all loaded database identifiers.

        Returns:
            List of database ID strings.
        """
        return list(self.db_file_paths.keys())

    @staticmethod
    def _canonicalize_col_name(name: str) -> str:
        """Removes quotes and converts to lowercase for consistent lookup."""
        return name.strip().replace('`', '').replace('"', '').replace("'", "").lower()

    @staticmethod
    def _quote_identifier(name: str) -> str:
        """Double-quote a SQLite identifier, escaping any embedded double-quotes."""
        return '"' + name.replace('"', '""') + '"'

    def _get_friendly_names_for_db(self, db_id: str) -> Dict[str, Any]:
        """Helper to retrieve human-friendly names/descriptions for tables and columns from BirdSchemaLoader.

        Stores column names as canonicalized (lowercase, no quotes) keys for robust lookup.
        """
        if not self.bird_schema_loader:
            return {'tables': {}, 'columns': {}}

        schema_metadata = self.bird_schema_loader.get_db_schema_metadata(db_id)
        if not schema_metadata:
            return {'tables': {}, 'columns': {}}

        friendly_names_map = {'tables': {}, 'columns': {}}

        if 'table_names_original' in schema_metadata and 'table_names' in schema_metadata:
            for original, friendly in zip(schema_metadata['table_names_original'], schema_metadata['table_names']):
                friendly_names_map['tables'][original] = friendly

        if 'column_names_original' in schema_metadata and 'column_names' in schema_metadata:
            table_idx_to_original = {i: name for i, name in enumerate(schema_metadata.get('table_names_original', []))}

            for i in range(len(schema_metadata['column_names_original'])):
                table_idx = schema_metadata['column_names_original'][i][0]
                col_original = schema_metadata['column_names_original'][i][1]
                col_friendly = schema_metadata['column_names'][i][1]

                original_table_name = table_idx_to_original.get(table_idx)

                if original_table_name and col_original != '*':
                    if original_table_name not in friendly_names_map['columns']:
                        friendly_names_map['columns'][original_table_name] = {}
                    friendly_names_map['columns'][original_table_name][self._canonicalize_col_name(col_original)] = col_friendly

        return friendly_names_map

    def get_ddl_with_friendly_names(self, db_id: str) -> str:
        """Retrieve DDL with human-friendly names as inline comments.

        Args:
            db_id: Database identifier.

        Returns:
            DDL string with friendly table/column names as SQL comments.
        """
        db_file_path = self._get_db_file_path(db_id)
        friendly_names = self._get_friendly_names_for_db(db_id)

        full_schema_prompt_list = []
        conn = None
        try:
            conn = sqlite3.connect(f'file:{db_file_path}?mode=ro', uri=True)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()

            for table_info in tables:
                table_name = table_info[0]
                if table_name == 'sqlite_sequence':
                    continue

                cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
                create_prompt_raw = cursor.fetchone()[0]

                create_prompt_formatted = sqlparse.format(create_prompt_raw, reindent=True, keyword_case='upper', indent_width=4)

                table_comment = ""
                if table_name in friendly_names['tables'] and friendly_names['tables'][table_name] != table_name:
                    table_comment = f" -- {friendly_names['tables'][table_name]}"

                column_def_pattern = r"^(\s*)([a-zA-Z0-9_`\"'\[\]]+)\s+(.+?),?$"

                lines = create_prompt_formatted.splitlines()
                processed_lines = []
                for line in lines:
                    match = re.match(column_def_pattern, line)
                    if match:
                        indent = match.group(1)
                        col_name_in_ddl = match.group(2).strip().replace('`', '').replace('"', '').replace("'", "")
                        type_and_constraints = match.group(3).strip()

                        col_comment = ""
                        if (table_name in friendly_names['columns']
                                and col_name_in_ddl in friendly_names['columns'][table_name]
                                and friendly_names['columns'][table_name][col_name_in_ddl] != col_name_in_ddl):
                            col_comment = f" -- {friendly_names['columns'][table_name][col_name_in_ddl]}"

                        processed_lines.append(f"{indent}{col_name_in_ddl} {type_and_constraints}{col_comment}")
                    else:
                        processed_lines.append(line)

                final_create_prompt_lines = []
                for i, line in enumerate(processed_lines):
                    if i == 0 and line.strip().upper().startswith("CREATE TABLE"):
                        table_name_in_regex = re.escape(table_name)
                        if re.search(r"CREATE TABLE [`\"']?" + table_name_in_regex + r"[`\"']?", line, re.IGNORECASE):
                            final_create_prompt_lines.append(f"{line}{table_comment}")
                        else:
                            final_create_prompt_lines.append(line)
                    else:
                        final_create_prompt_lines.append(line)

                full_schema_prompt_list.append("\n".join(final_create_prompt_lines))

            schema_prompt = "\n\n".join(full_schema_prompt_list)
            return schema_prompt
        except Exception as e:
            logger.error(f"Error generating DDL schema for '{db_id}': {e}", exc_info=True)
            return f"/* Error loading DDL schema for '{db_id}': {e} */"
        finally:
            if conn: conn.close()

    def _build_schema_preamble(
        self, db_id: str
    ) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict], Optional[Dict]]:
        """Build shared components needed for semantic schema generation.

        Returns:
            Tuple of (schema_metadata, friendly_names_map, column_type_lookup,
                     table_idx_to_original_name), or a tuple of Nones if unavailable.
        """
        if not self.bird_schema_loader:
            logger.warning(f"No BirdSchemaLoader initialized. Cannot generate semantic schema for {db_id}.")
            return None, None, None, None
        schema_metadata = self.bird_schema_loader.get_db_schema_metadata(db_id)
        if not schema_metadata:
            logger.warning(f"No schema metadata found for {db_id} in BirdSchemaLoader.")
            return None, None, None, None
        friendly_names_map = self._get_friendly_names_for_db(db_id)
        column_type_lookup: Dict[str, Dict[str, str]] = {}
        table_idx_to_original_name = {i: name for i, name in enumerate(schema_metadata.get('table_names_original', []))}
        if 'column_names_original' in schema_metadata and 'column_types' in schema_metadata:
            for i, (table_idx, col_original_name) in enumerate(schema_metadata['column_names_original']):
                original_table_name = table_idx_to_original_name.get(table_idx)
                if original_table_name:
                    if original_table_name not in column_type_lookup:
                        column_type_lookup[original_table_name] = {}
                    column_type_lookup[original_table_name][self._canonicalize_col_name(col_original_name)] = schema_metadata['column_types'][i]
        return schema_metadata, friendly_names_map, column_type_lookup, table_idx_to_original_name

    def _build_table_column_description(
        self,
        schema_metadata: Dict[str, Any],
        friendly_names_map: Dict[str, Any],
        column_type_lookup: Dict[str, Dict[str, str]],
        table_idx: int,
        original_table_name: str,
    ) -> List[str]:
        """Return semantic description lines for a single table's header and columns."""
        parts: List[str] = []
        friendly_table_name = friendly_names_map['tables'].get(original_table_name, original_table_name)
        parts.append(f"Table: {original_table_name}")
        if friendly_table_name and friendly_table_name.strip() and friendly_table_name != original_table_name:
            parts.append(f"  Human-friendly Table name: {friendly_table_name}")
        col_definitions: List[str] = []
        for col_table_idx, col_original_name in schema_metadata.get('column_names_original', []):
            if col_table_idx == table_idx:
                canonical = self._canonicalize_col_name(col_original_name)
                col_friendly = friendly_names_map['columns'].get(original_table_name, {}).get(canonical)
                col_type = column_type_lookup.get(original_table_name, {}).get(canonical, "UNKNOWN")
                col_info = [f"Column {col_original_name}: Type -> {col_type}"]
                if col_friendly and col_friendly.strip():
                    col_info.append(f"Desc -> {col_friendly}")
                col_definitions.append(f"  {', '.join(col_info)}")
        if col_definitions:
            parts.extend(col_definitions)
        return parts

    def get_semantic_schema_for_llm(self, db_id: str) -> str:
        """Generate a semantic schema description for LLM prompts.

        Args:
            db_id: Database identifier.

        Returns:
            Human-readable schema with table/column descriptions, types, PKs, and FKs.
        """
        schema_metadata, friendly_names_map, column_type_lookup, table_idx_to_original_name = \
            self._build_schema_preamble(db_id)
        if schema_metadata is None:
            if not self.bird_schema_loader:
                return "/* Semantic schema generation not available. */"
            return f"/* No schema metadata available for {db_id}. */"

        schema_description_parts: List[str] = []

        for table_idx, original_table_name in enumerate(schema_metadata.get('table_names_original', [])):
            schema_description_parts.extend(
                self._build_table_column_description(
                    schema_metadata, friendly_names_map, column_type_lookup, table_idx, original_table_name
                )
            )
            schema_description_parts.append("")

        # Add Primary Keys
        pk_info = []
        if 'primary_keys' in schema_metadata:
            for pk_def in schema_metadata['primary_keys']:
                if isinstance(pk_def, list):
                    pk_table_idx = pk_def[0]
                    pk_col_original_indices = pk_def[1:]
                    pk_table_name = table_idx_to_original_name.get(pk_table_idx)
                    if pk_table_name:
                        current_table_all_cols = [
                            col_original for tbl_idx, col_original in schema_metadata['column_names_original']
                            if tbl_idx == pk_table_idx
                        ]
                        col_names_for_composite_pk = [
                            current_table_all_cols[idx]
                            for idx in pk_col_original_indices
                            if idx < len(current_table_all_cols)
                        ]
                        if col_names_for_composite_pk:
                            pk_info.append(f"  {pk_table_name}.({', '.join(col_names_for_composite_pk)})")
                elif isinstance(pk_def, int):
                    if pk_def < len(schema_metadata['column_names_original']):
                        pk_table_idx, pk_col_original = schema_metadata['column_names_original'][pk_def]
                        pk_table_name = table_idx_to_original_name.get(pk_table_idx)
                        if pk_table_name:
                            pk_info.append(f"  {pk_table_name}.{pk_col_original}")
            if pk_info:
                schema_description_parts.append("Primary Keys (PKs):")
                schema_description_parts.extend(pk_info)
                schema_description_parts.append("")

        # Add Foreign Keys
        fk_info = []
        if 'foreign_keys' in schema_metadata:
            for fk_pair in schema_metadata['foreign_keys']:
                if (len(fk_pair) == 2
                        and fk_pair[0] < len(schema_metadata['column_names_original'])
                        and fk_pair[1] < len(schema_metadata['column_names_original'])):
                    from_col_info = schema_metadata['column_names_original'][fk_pair[0]]
                    to_col_info = schema_metadata['column_names_original'][fk_pair[1]]
                    from_table = table_idx_to_original_name.get(from_col_info[0])
                    from_col = from_col_info[1]
                    to_table = table_idx_to_original_name.get(to_col_info[0])
                    to_col = to_col_info[1]
                    if from_table and from_col and to_table and to_col:
                        fk_info.append(f"  {from_table}.{from_col} -> {to_table}.{to_col}")
            if fk_info:
                schema_description_parts.append("Foreign Keys (FKs):")
                schema_description_parts.extend(fk_info)
                schema_description_parts.append("")

        # Add Legend
        schema_description_parts.append("Legend:")
        schema_description_parts.append("  Human-friendly name: Real-world name for table/column.")
        schema_description_parts.append("  Desc: Description or human-friendly name of the column.")
        schema_description_parts.append("  PKs: Primary Keys.")
        schema_description_parts.append("  FKs: Foreign Keys.")

        return "\n".join(schema_description_parts).strip()

    def get_semantic_schema_for_llm_with_rows(self, db_id: str, num_rows: int = 3) -> str:
        """Generate semantic schema with example rows for LLM prompts.

        Args:
            db_id: Database identifier.
            num_rows: Number of sample rows to include per table.

        Returns:
            Schema description with types, descriptions, PKs, FKs, and sample data.
        """
        schema_metadata, friendly_names_map, column_type_lookup, _ = \
            self._build_schema_preamble(db_id)
        if schema_metadata is None:
            if not self.bird_schema_loader:
                return "/* Semantic schema with rows generation not available. */"
            return f"/* No schema metadata available for {db_id}. */"

        schema_description_parts: List[str] = []
        conn = None
        db_file_path = self._get_db_file_path(db_id)

        try:
            conn = sqlite3.connect(f'file:{db_file_path}?mode=ro', uri=True)
            cursor_for_rows = conn.cursor()

            for table_idx, original_table_name in enumerate(schema_metadata.get('table_names_original', [])):
                schema_description_parts.extend(
                    self._build_table_column_description(
                        schema_metadata, friendly_names_map, column_type_lookup, table_idx, original_table_name
                    )
                )

                if num_rows > 0:
                    cur_table_for_query = original_table_name
                    try:
                        table_columns_metadata = [
                            (col_original_name, schema_metadata['column_types'][col_entry_idx])
                            for col_entry_idx, (col_table_idx, col_original_name) in enumerate(schema_metadata.get('column_names_original', []))
                            if col_table_idx == table_idx
                        ]
                        non_blob_columns = [
                            col_name for col_name, col_type in table_columns_metadata if col_type.upper() != 'BLOB'
                        ]
                        if non_blob_columns:
                            column_list_for_query = ', '.join(self._quote_identifier(col) for col in non_blob_columns)
                            query = f"SELECT {column_list_for_query} FROM {self._quote_identifier(cur_table_for_query)} LIMIT {num_rows}"
                            cursor_for_rows.execute(query)
                            column_names_for_rows = [description[0] for description in cursor_for_rows.description]
                            values_for_rows = cursor_for_rows.fetchall()
                            if values_for_rows:
                                truncated_values_for_rows = []
                                for row in values_for_rows:
                                    truncated_row = []
                                    for cell in row:
                                        if isinstance(cell, str) and len(cell) > 50:
                                            truncated_row.append(cell[:50] + "...")
                                        else:
                                            truncated_row.append(cell)
                                    truncated_values_for_rows.append(tuple(truncated_row))
                                rows_prompt = nice_look_table(column_names=column_names_for_rows, values=truncated_values_for_rows)
                                schema_description_parts.append("/*")
                                schema_description_parts.append(f"3 rows from {cur_table_for_query} table:")
                                indented_rows_prompt = rows_prompt.replace('\n', '\n  ')
                                schema_description_parts.append(f"  {indented_rows_prompt}")
                                schema_description_parts.append("*/")
                    except Exception as table_e:
                        logger.warning(f"Could not fetch example rows for table '{cur_table_for_query}' in DB '{db_id}': {table_e}")

                schema_description_parts.append("")

            # Add Legend
            schema_description_parts.append("Legend:")
            schema_description_parts.append("  Human-friendly name: Real-world name for table/column.")
            schema_description_parts.append("  Desc: Description or human-friendly name of the column.")
            schema_description_parts.append("  PKs: Primary Keys.")
            schema_description_parts.append("  FKs: Foreign Keys.")

            return "\n".join(schema_description_parts).strip()
        except Exception as e:
            logger.error(f"Error generating semantic schema with rows for '{db_id}': {e}", exc_info=True)
            return f"/* Error generating semantic schema with rows for '{db_id}': {e} */"
        finally:
            if conn: conn.close()
