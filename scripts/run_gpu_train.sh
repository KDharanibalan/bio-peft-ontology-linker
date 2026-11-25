#!/usr/bin/env bash
set -euo pipefail

# Reproducible Linux GPU launcher for HuggingFace + PyTorch + PEFT QLoRA fine-tuning
CONFIG_PATH="${1:-configs/lora_sft_config.yaml}"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

echo "[INFO] Validating dataset and experiment config via dry-run..."
python3 -m bio_peft.train_lora --config "${CONFIG_PATH}" --dry-run

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] GPU detected:"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  if command -v accelerate >/dev/null 2>&1; then
    accelerate launch --mixed_precision bf16 -m bio_peft.train_lora --config "${CONFIG_PATH}"
  else
    python3 -m bio_peft.train_lora --config "${CONFIG_PATH}"
  fi
else
  echo "[WARN] No NVIDIA GPU detected. Stopping after dry-run validation."
fi
