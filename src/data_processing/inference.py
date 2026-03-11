#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM inference and SQL execution for data curation.

Handles direct LLM API invocation, SQL generation from natural language,
execution against databases, and retry logic with exponential backoff.

@author: Khushboo Thaker, Yony Bresler, Khalid Eidoo
@license: Crater Labs (C)
"""

import json
import logging
import os
import re
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Character limit for prompt text (≈ 50k tokens at ~4 chars/token).
# Prompts exceeding this are retried with fewer schema sample rows.
_MAX_PROMPT_CHARS: int = 200_000
# HTTP request timeout for LLM API calls (seconds).
_API_REQUEST_TIMEOUT_SEC: int = 120
# Exponential-backoff base delay between inference retries (seconds).
_RETRY_BASE_DELAY_SEC: int = 4

# Fields written to each per-sample cache file. Explicit whitelist avoids
# accidentally serializing non-JSON-safe objects (db_manager, llm, etc.).
_RESULT_FIELDS = frozenset({
    'question_id', 'db_id', 'question', 'SQL', 'level', 'evidence',
    'predicted_sql', 'final_reasoning', 'generation_status',
    'execution_match', 'predicted_exec_status', 'error_occurred',
    'raw_output', 'prompt_template_string', 'error_message',
})


def invoke_llm_direct(llm_instance: Any, prompt_text: str, model_name: str = "gpt-4o") -> str:
    """Invoke the LLM chat-completions API and return the raw response text.

    Supports both native Azure OpenAI (api-key header) and any OpenAI-compatible
    endpoint (Authorization: Bearer header) based on the base URL.

    Args:
        llm_instance: An LLMConnection or compatible object with base_url, api_key,
            and temperature attributes.
        prompt_text: The user prompt to send as a single-turn message.
        model_name: Deployment or model identifier to pass in the request payload.

    Returns:
        Raw text content of the model's response.

    Raises:
        ValueError: If credentials cannot be extracted, or if the context length is exceeded.
        requests.exceptions.HTTPError: On non-2xx API responses.
    """
    # Extract credentials
    api_key = None
    if hasattr(llm_instance, 'openai_api_key'):
        val = llm_instance.openai_api_key
        api_key = val.get_secret_value() if hasattr(val, 'get_secret_value') else val
    elif hasattr(llm_instance, 'api_key'):
        val = llm_instance.api_key
        api_key = val.get_secret_value() if hasattr(val, 'get_secret_value') else val

    base_url = None
    if hasattr(llm_instance, 'openai_api_base'):
        base_url = llm_instance.openai_api_base
    elif hasattr(llm_instance, 'azure_endpoint'):
        base_url = llm_instance.azure_endpoint
    elif hasattr(llm_instance, 'base_url'):
        base_url = llm_instance.base_url

    if not api_key or not base_url:
        raise ValueError("Could not extract API key or Base URL from LLM instance.")

    # Native Azure OpenAI uses "api-key" header; all other OpenAI-compatible endpoints
    # (Lightning AI, standard OpenAI, etc.) use "Authorization: Bearer".
    if "openai.azure.com" in base_url:
        headers = {"api-key": api_key, "Content-Type": "application/json"}
    else:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    logger.debug(f"Using model: {model_name}")
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": llm_instance.temperature,
    }

    # Ensure URL is correct for chat completions
    url = base_url if "chat/completions" in base_url else f"{base_url.rstrip('/')}/chat/completions"

    try:
        response = requests.post(url=url, headers=headers, json=payload, timeout=_API_REQUEST_TIMEOUT_SEC)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        logger.info(f"API Request: Input Size={len(prompt_text)} chars, Output Size={len(content)} chars")
        return content
    except requests.exceptions.HTTPError as e:
        logger.error(f"API HTTP Error: {e}")
        if e.response is not None:
            logger.error(f"API Error Response Body: {e.response.text}")
            if "context length" in e.response.text.lower():
                raise ValueError("ContextLengthExceeded")
        raise


def generate_and_execute_sample(sample: dict, state_components: dict[str, Any]) -> dict:
    """Generate SQL for a single sample via the LLM and execute it against the database.

    Adaptively reduces schema verbosity (3 rows → 1 row → 0 rows) if the prompt
    exceeds the character limit. Returns a result dict regardless of success or failure.

    Args:
        sample: A data record with keys 'question_id', 'db_id', 'question', 'SQL',
            and optionally 'evidence'.
        state_components: Shared inference state containing 'db_manager', 'llm',
            'prompt_template_string', and 'model_name'.

    Returns:
        Dict with keys 'predicted_sql', 'generation_status', and 'raw_output'.
        On execution success also includes 'final_reasoning', 'execution_match',
        and 'predicted_exec_status'. On execution exception also includes
        'error_occurred'.
    """
    question_id = sample.get("question_id")
    db_id = sample.get("db_id")
    nl_question = sample.get("question")

    db_manager = state_components['db_manager']
    llm = state_components['llm']
    prompt_template = state_components['prompt_template_string']
    model_name = state_components.get('model_name', 'gpt-4o')

    # Step 1: Prepare prompt
    schema_ddl = db_manager.get_ddl(db_id)
    format_instructions = ""

    prompt_text = None

    # Adaptive schema sizing: Try 3 rows, then 1 row, then 0 rows (schema only)
    for num_rows in [3, 1, 0]:
        schema_semantic = db_manager.get_semantic_schema_for_llm_with_rows(db_id, num_rows=num_rows)

        current_prompt_text = prompt_template.format(
            db_schema_context=schema_ddl,
            db_schema_context_hm=schema_ddl,
            nl_question=nl_question,
            db_schema_context_hmr=schema_semantic,
            hint=sample.get('evidence') or '',
            query1='',
            query2='',
            format_instructions=format_instructions
        )

        # Check length heuristic (approx 3-4 chars per token). 128k tokens ~ 400k-500k chars.
        if len(current_prompt_text) <= _MAX_PROMPT_CHARS:
            prompt_text = current_prompt_text
            logger.debug(f"Prompt length: {len(prompt_text)} chars")
            break
        else:
            logger.info(f"Prompt too long ({len(current_prompt_text)} chars) with {num_rows} rows for {question_id}. Reducing context...")
    if not prompt_text:
        logger.warning(f"Prompt too long even with 0 rows for {question_id}. Skipping API call.")
        return {
            "predicted_sql": "ERROR: PROMPT TOO LONG",
            "generation_status": "ContextLimit",
            "raw_output": "Prompt exceeded character limit"
        }

    # Step 2: Invoke API
    raw_output = invoke_llm_direct(llm, prompt_text, model_name)

    # Step 3: Parse output
    predicted_sql = None
    reasoning = None

    # Extract SQL from markdown code block
    sql_match = re.search(r"```(?:sql|sqlite)?\s*(.*?)```", raw_output, re.DOTALL | re.IGNORECASE)

    if sql_match:
        predicted_sql = sql_match.group(1).strip()
        # Everything before the SQL block is treated as the Query Plan / Reasoning
        reasoning = raw_output[:sql_match.start()].strip()
    else:
        return {
            "predicted_sql": "ERROR: NO SQL BLOCK FOUND",
            "generation_status": "Failure",
            "raw_output": raw_output
        }

    # Step 4: Execute SQL
    try:
        cmp = db_manager.execute_and_compare(db_id=db_id, query1=sample.get('SQL'), query2=predicted_sql)

        return {
            "predicted_sql": predicted_sql,
            "final_reasoning": reasoning,
            "generation_status": "Success",
            "execution_match": cmp.match_status,
            "predicted_exec_status": cmp.predict_exec_status,
            "raw_output": raw_output
        }
    except Exception as e:
        logger.error(f"Execution failed for {question_id}: {e}")
        return {
            "predicted_sql": predicted_sql,
            "generation_status": "Success",  # Generation worked, execution failed
            "execution_match": False,
            "predicted_exec_status": "Failure",
            "error_occurred": str(e),
            "raw_output": raw_output
        }


def run_inference_for_sample(sample: dict, state_components: dict[str, Any]) -> dict:
    """
    Run inference for a single sample with per-result file caching and exponential backoff.

    This is the main entry point for producing one labelled result. It wraps
    ``generate_and_execute_sample`` with three layers of resilience:

    1. **Cache hit** — if a result file already exists for this question_id, it is
       loaded and returned immediately without an API call.
    2. **Retry loop** — on transient failures (network errors, rate limits), up to
       ``max_retries`` attempts are made with exponential backoff
       (base_delay × 2^attempt seconds between tries).
    3. **Hard stops** — context-length errors are never retried; a structured error
       dict is returned instead so the calling loop can continue safely.

    On success the result is written to ``<individual_results_dir>/<question_id>.json``
    before returning, making every successful call idempotent on resume.

    Args:
        sample: A data record with at minimum 'question_id', 'db_id', 'question',
            and 'SQL' keys.
        state_components: Shared inference state containing 'individual_results_dir',
            'api_delay_seconds', and all fields required by
            ``generate_and_execute_sample``.

    Returns:
        Dict merging the original sample fields with inference results. Always contains
        'generation_status' ('Success', 'Error', or 'ContextLimit'). On success also
        contains 'predicted_sql', 'execution_match', and 'raw_output'.

    Raises:
        ValueError: If question_id resolves to a path outside the cache directory
            (path traversal guard).
    """

    question_id = sample.get("question_id")
    safe_id = re.sub(r'[^\w\-]', '_', str(question_id))
    cache_base = os.path.realpath(state_components['individual_results_dir'])
    output_filepath = os.path.join(cache_base, f"{safe_id}.json")
    if not os.path.realpath(output_filepath).startswith(cache_base + os.sep):
        raise ValueError(f"question_id '{question_id}' resolves outside cache directory")
    if os.path.exists(output_filepath):
        logger.info(f"Result file for question_id {question_id} already exists. Loading from cache.")
        try:
            with open(output_filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not load cached file {output_filepath}: {e}. Proceeding with inference.")
    max_retries = 5
    base_delay = state_components.get("api_delay_seconds", _RETRY_BASE_DELAY_SEC)

    for attempt in range(max_retries):
        if attempt > 0:
            # Exponential backoff: 4, 8, 16, 32, 64 seconds
            sleep_time = base_delay * (2 ** attempt)
            logger.info(f"Retrying inference for {question_id} in {sleep_time}s (Attempt {attempt + 1}/{max_retries})...")
            time.sleep(sleep_time)

        try:
            result_dict = generate_and_execute_sample(sample, state_components)

            if result_dict.get("generation_status") == "ContextLimit":
                logger.error(f"Context limit reached for {question_id}. Not retrying.")
                return {**sample, "generation_status": "Error", "error_message": "Context length exceeded"}

            if result_dict.get("generation_status") == "Failure":

                raise ValueError(f"Generation failed: {result_dict.get('predicted_sql')}")

            full_result = {**sample, **result_dict}
            serializable_result = {k: v for k, v in full_result.items() if k in _RESULT_FIELDS}
            with open(output_filepath, 'w', encoding='utf-8') as f_out:
                json.dump(serializable_result, f_out, indent=4)

            return serializable_result
        except Exception as e:
            if "ContextLengthExceeded" in str(e):
                logger.error(f"Context length exceeded for {question_id} (API rejected). Not retrying.")
                return {**sample, "generation_status": "Error", "error_message": "Context length exceeded"}

            logger.warning(f"Inference attempt {attempt + 1}/{max_retries} failed for question_id {question_id}: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Inference permanently failed for question_id {question_id}: {e}")
                # Return a failure object so the main loop doesn't crash
                return {**sample, "generation_status": "Error", "error_message": str(e)}

    return {**sample, "generation_status": "Error", "error_message": "Unknown error"}
