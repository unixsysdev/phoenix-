#!/bin/bash

# Script to evaluate trained LoRA adapters

set -euo pipefail

# Default values
MODEL_PATH="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
ADAPTER_PATH=""
CONFIG_PATH="configs/lora_configs.yaml"
TEST_DATA_FILE=""
OUTPUT_FILE=""
HEADS="all"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --adapter_path)
            ADAPTER_PATH="$2"
            shift 2
            ;;
        --config_path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --test_data_file)
            TEST_DATA_FILE="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --heads)
            HEADS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$ADAPTER_PATH" ]; then
    echo "Error: --adapter_path is required"
    echo "Usage: $0 --adapter_path <path> [options]"
    exit 1
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Build evaluation command
EVAL_CMD="python -m src.evaluation.evaluator \
    --model_path $MODEL_PATH \
    --adapter_path $ADAPTER_PATH \
    --config_path $CONFIG_PATH"

# Add optional arguments
if [ -n "$TEST_DATA_FILE" ]; then
    EVAL_CMD="$EVAL_CMD --test_data_file $TEST_DATA_FILE"
fi

if [ -n "$OUTPUT_FILE" ]; then
    EVAL_CMD="$EVAL_CMD --output_file $OUTPUT_FILE"
fi

echo "Running evaluation..."
echo "Model: $MODEL_PATH"
echo "Adapters: $ADAPTER_PATH"
echo "Heads: $HEADS"

# Run evaluation
eval $EVAL_CMD

echo "Evaluation completed!"