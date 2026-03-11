#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQL Query Complexity Classifier.

Classifies SQL queries into complexity levels based on JOINs, subqueries,
set operations, aggregations, and ordering clauses.

@author: Khushboo Thaker, Yony Bresler, Khalid Eidoo
@license: Crater Labs (C)
"""

import re
import logging

logger = logging.getLogger(__name__)

# SQL complexity level constants
SQL_LEVEL_L4 = "Level 4: JOINs / Set Ops with Subquery"
SQL_LEVEL_L3 = "Level 3: JOINs / Set Ops without Subquery"
SQL_LEVEL_L2 = "Level 2: Subquery"
SQL_LEVEL_L1 = "Level 1: Basic Query"
SQL_LEVEL_OTHER = "Other/Uncategorized"


class SQLClassifier:
    """
    Classifies SQL queries into complexity levels for stratified sampling.

    Levels (most to least complex):
        L4: JOINs/Set Ops with subquery
        L3: JOINs/Set Ops without subquery
        L2: Subquery only
        L1: Basic (simple SELECT, ORDER BY, GROUP BY/HAVING)
    """

    def __init__(self):
        """Initialize complexity level definitions."""
        self.levels = {
            "L4":    SQL_LEVEL_L4,
            "L3":    SQL_LEVEL_L3,
            "L2":    SQL_LEVEL_L2,
            "L1":    SQL_LEVEL_L1,
            "OTHER": SQL_LEVEL_OTHER,
        }

    def classify(self, sql: str) -> str:
        """Classify a SQL query based on structural complexity.

        Args:
            sql: The SQL query string to classify.

        Returns:
            Human-readable complexity level description.
        """
        if not sql:
            return self.levels["OTHER"]

        sql_upper = sql.upper()
        select_count = len(re.findall(r"\bSELECT\b", sql_upper))
        has_join = "JOIN" in sql_upper
        has_set_op = any(op in sql_upper for op in ["UNION", "INTERSECT", "EXCEPT"])

        # Check from most complex to least complex
        if (has_join or has_set_op) and select_count > 1:
            return self.levels["L4"]
        if has_join or has_set_op:
            return self.levels["L3"]
        if select_count > 1:
            return self.levels["L2"]

        # Basic queries: simple SELECT, ORDER BY, GROUP BY/HAVING all map to Level 1
        return self.levels["L1"] if "SELECT" in sql_upper else self.levels["OTHER"]
