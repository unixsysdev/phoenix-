#!/bin/bash

# Training script optimized for H100 GPU with QLoRA

set -euo pipefail

# Configuration
MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
CONFIG_FILE="configs/qwen3_coder_30b_moe.yaml"
DATA_DIR="data"
OUTPUT_DIR="logs/h100_training"
WANDB_PROJECT="qwen-diffusion-h100"

# H100 optimized settings
MICRO_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=1
MAX_STEPS=40000
LEARNING_RATE=1e-4
WARMUP_STEPS=500

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
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
        --batch_size)
            MICRO_BATCH_SIZE="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
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

# Set environment variables for H100
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="$WANDB_PROJECT"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Check for H100
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)
    echo "Detected GPU: $GPU_NAME"
    
    if [[ $GPU_NAME == *"H100"* ]]; then
        echo "H100 detected - applying optimizations"
        export TORCH_CUDA_ARCH_LIST="8.0;9.0"
    else
        echo "Warning: H100 not detected. Consider using H100 for optimal performance."
    fi
else
    echo "Warning: nvidia-smi not found. GPU optimizations may not be applied."
fi

# Check if data exists
if [ ! -f "$DATA_DIR/train_samples.json" ]; then
    echo "Training data not found at $DATA_DIR/train_samples.json"
    echo "Running data preparation..."
    
    # Create synthetic data if no data provided
    python scripts/prepare_data.py \
        --output_dir "$DATA_DIR" \
        --synthetic_samples 5000 \
        --max_length 2048 \
        --model_name "$MODEL_NAME"
fi

# Create H100-optimized config
H100_CONFIG_FILE="$OUTPUT_DIR/h100_config.yaml"
cat > "$H100_CONFIG_FILE" << EOF
model:
  name: "$MODEL_NAME"
  type: "moe"
  active_params: "3B"

training:
  use_lora: true
  lora_rank: 128
  lora_alpha: 256
  lora_dropout: 0.05
  multi_head: true
  micro_batch_size: $MICRO_BATCH_SIZE
  gradient_accumulation_steps: $GRADIENT_ACCUMULATION_STEPS
  max_steps: $MAX_STEPS
  warmup_steps: $WARMUP_STEPS
  learning_rate: $LEARNING_RATE
  weight_decay: 0.01
  max_grad_norm: 1.0
  gradient_checkpointing: true
  mixed_precision: "bf16"
  use_flash_attn: true
  save_steps: 5000
  
  diffusion:
    mask_ratio_range: [0.1, 0.9]
    steps: 100
    inference_steps: 50
    scheduler: "cosine"
    
  length_prediction:
    enabled: true
    num_buckets: 20
    max_length: 1000

data:
  train_path: "$DATA_DIR/train_samples.json"
  val_path: "$DATA_DIR/val_samples.json"
  max_length: 2048
  mask_token: "<|mask|>"
  eos_token: "<|endoftext|>"
EOF

# Estimate training time
echo "Training configuration:"
echo "  Model: $MODEL_NAME"
echo "  Batch size: $MICRO_BATCH_SIZE"
echo "  Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "  Effective batch size: $((MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "  Max steps: $MAX_STEPS"
echo "  Learning rate: $LEARNING_RATE"

# Approximate training time calculation
EFFECTIVE_BATCH_SIZE=$((MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
if [ $EFFECTIVE_BATCH_SIZE -ge 16 ]; then
    ESTIMATED_HOURS=7
    ESTIMATED_MINUTES=$((7 * 60))
else
    # Scale up time for smaller batch sizes
    SCALE_FACTOR=$((16 / EFFECTIVE_BATCH_SIZE))
    ESTIMATED_HOURS=$((7 * SCALE_FACTOR))
    ESTIMATED_MINUTES=$((ESTIMATED_HOURS * 60))
fi

echo "  Estimated training time: ~${ESTIMATED_HOURS} hours (${ESTIMATED_MINUTES} minutes)"

# Build training command
TRAIN_CMD="python -m src.training.train \
    --config_file $H100_CONFIG_FILE \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR"

echo "Starting training with command:"
echo "$TRAIN_CMD"

# Run training with timing
START_TIME=$(date +%s)
eval $TRAIN_CMD
END_TIME=$(date +%s)

DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo "Training completed in ${HOURS}h ${MINUTES}m!"
echo "Results saved to: $OUTPUT_DIR"

# Save training summary
cat > "$OUTPUT_DIR/training_summary.txt" << EOF
Training Summary
===============
Model: $MODEL_NAME
GPU: $GPU_NAME
Batch size: $MICRO_BATCH_SIZE
Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS
Effective batch size: $EFFECTIVE_BATCH_SIZE
Max steps: $MAX_STEPS
Learning rate: $LEARNING_RATE
Training time: ${HOURS}h ${MINUTES}m
Start time: $(date -d @$START_TIME)
End time: $(date -d @$END_TIME)
EOF

echo "Training summary saved to: $OUTPUT_DIR/training_summary.txt"