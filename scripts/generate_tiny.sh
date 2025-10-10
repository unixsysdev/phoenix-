#!/usr/bin/env bash
set -euo pipefail

# Tiny generator script using adapters from the tiny training run

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
MODEL_PATH="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
ADAPTER_PATH="${ROOT_DIR}/logs/tiny/final/model"
CONFIG_PATH="${ROOT_DIR}/configs/lora_configs.yaml"
OUT_DIR="${ROOT_DIR}/logs/tiny"
OUT_FILE="${OUT_DIR}/generated_function.py"

mkdir -p "${OUT_DIR}"

if [[ ! -d "${ADAPTER_PATH}" ]]; then
  echo "Adapters not found at: ${ADAPTER_PATH}"
  echo "Run tiny training first: bash scripts/train_tiny.sh"
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

SIG="def calculate_sum(numbers):"
DOC="Calculate the sum of a list of numbers."

echo "Generating function using adapters at: ${ADAPTER_PATH}"

python3 -m src.inference.generator \
  --model_path "${MODEL_PATH}" \
  --adapter_path "${ADAPTER_PATH}" \
  --config_path "${CONFIG_PATH}" \
  --mode function \
  --signature "${SIG}" \
  --docstring "${DOC}" \
  --steps 20 \
  --temperature 0.7 \
  --output_file "${OUT_FILE}"

echo "Generated function saved to: ${OUT_FILE}"

