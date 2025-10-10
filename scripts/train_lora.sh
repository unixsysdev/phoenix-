#!/bin/bash

# Training script for Qwen3 Coder 30B MoE with LoRA adapters

set -euo pipefail

# Default values
CONFIG_FILE="configs/qwen3_coder_30b_moe.yaml"
DATA_DIR="data"
OUTPUT_DIR="logs"
RESUME_FROM=""
WANDB_PROJECT="qwen-diffusion-training"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --resume_from)
            RESUME_FROM="$2"
            shift 2
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="$WANDB_PROJECT"

# Check if data exists
if [ ! -f "$DATA_DIR/train_samples.json" ]; then
    echo "Training data not found at $DATA_DIR/train_samples.json"
    echo "Run: python scripts/prepare_data.py --output_dir $DATA_DIR"
    exit 1
fi

# Build training command
TRAIN_CMD="python -m src.training.train \
    --config_file $CONFIG_FILE \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR"

# Add resume flag if specified
if [ -n "$RESUME_FROM" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume_from $RESUME_FROM"
fi

echo "Starting training with command:"
echo "$TRAIN_CMD"
echo "Configuration: $CONFIG_FILE"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"

# Run training
eval $TRAIN_CMD

echo "Training completed successfully!"
echo "Check results in: $OUTPUT_DIR"