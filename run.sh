#!/bin/bash

# --- BIRD-SQL Submission Script ---
# 1. Install Dependencies
# BIRD guidelines require a requirements.txt file. 
# This command ensures the evaluation environment has all your necessary libraries.
pip install -r requirements.txt

# 2. Configure Python Path
# This is CRITICAL. It tells Python to look in the current directory (.) 
# so that "from src.database_manager import ..." works correctly inside run_inference.py.
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3. Run Inference
# We use the standard BIRD test paths here.
# The evaluators will map the actual hidden test set to these locations.

echo "Starting Inference..."

# --- Configuration ---
# Adjust these paths to match your actual folder structure
INPUT_FILE="./data/BIRD/minidev/MINIDEV/mini_dev_sqlite.json"
DB_PATH="./data/BIRD/dev/dev_databases/"
TABLES_FILE="./data/BIRD/dev/dev_tables.json"
MODEL_PATH="craterlabs/Struct-SQL"
PROMPT_FILE="prompts/structsql.txt"
OUTPUT_FILE="exp_results/predict_dev.json"

# --- Execution ---
echo "Starting BIRD Benchmark Inference..."
python run_inference.py \
  --input_file "$INPUT_FILE" \
  --db_path "$DB_PATH" \
  --tables_file "$TABLES_FILE" \
  --model_path "$MODEL_PATH" \
  --prompt_file "$PROMPT_FILE" \
  --output_file "$OUTPUT_FILE"

echo "Done! Results saved to $OUTPUT_FILE"