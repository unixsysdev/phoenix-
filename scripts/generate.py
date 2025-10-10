#!/bin/bash

# Script to generate code using trained LoRA adapters

set -euo pipefail

# Default values
MODEL_PATH="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
ADAPTER_PATH=""
CONFIG_PATH="configs/lora_configs.yaml"
MODE="function"
OUTPUT_FILE=""
STEPS=50
TEMPERATURE=0.7
NUM_CANDIDATES=5

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
        --mode)
            MODE="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --signature)
            SIGNATURE="$2"
            shift 2
            ;;
        --docstring)
            DOCSTRING="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --num_candidates)
            NUM_CANDIDATES="$2"
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

# Build generation command
GEN_CMD="python -m src.inference.generator \
    --model_path $MODEL_PATH \
    --adapter_path $ADAPTER_PATH \
    --config_path $CONFIG_PATH \
    --mode $MODE \
    --steps $STEPS \
    --temperature $TEMPERATURE"

# Add mode-specific arguments
if [ -n "${PROMPT:-}" ]; then
    GEN_CMD="$GEN_CMD --prompt \"$PROMPT\""
fi

if [ -n "${SIGNATURE:-}" ]; then
    GEN_CMD="$GEN_CMD --signature \"$SIGNATURE\""
fi

if [ -n "${DOCSTRING:-}" ]; then
    GEN_CMD="$GEN_CMD --docstring \"$DOCSTRING\""
fi

if [ -n "$OUTPUT_FILE" ]; then
    GEN_CMD="$GEN_CMD --output_file $OUTPUT_FILE"
fi

if [ "$MODE" = "candidates" ]; then
    GEN_CMD="$GEN_CMD --num_candidates $NUM_CANDIDATES"
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

echo "Running code generation..."
echo "Mode: $MODE"
echo "Model: $MODEL_PATH"
echo "Adapters: $ADAPTER_PATH"
echo "Steps: $STEPS"
echo "Temperature: $TEMPERATURE"

# Run generation
eval $GEN_CMD

echo "Generation completed!"