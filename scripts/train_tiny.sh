#!/usr/bin/env bash
set -euo pipefail

# Tiny smoke-test training on the bundled mini dataset

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
CONFIG_FILE="${ROOT_DIR}/configs/qwen3_coder_30b_moe_tiny.yaml"
DATA_DIR="${ROOT_DIR}/data"
OUTPUT_DIR="${ROOT_DIR}/logs/tiny"

mkdir -p "${OUTPUT_DIR}"

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "Using config: ${CONFIG_FILE}"
echo "Data dir:    ${DATA_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"

python3 -m src.training.train \
  --config_file "${CONFIG_FILE}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}"

echo "Tiny run complete. Check: ${OUTPUT_DIR}"

