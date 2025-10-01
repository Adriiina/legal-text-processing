#!/usr/bin/env bash
set -euo pipefail

BATCHES_DIR="data/batches"
OUT_DIR="data/processed/embeddings"
MODEL="bert-base-uncased"
POOLING="mean"
WINDOW=256
STRIDE=200
BATCH_SIZE=8
MAX_LENGTH=256
MAX_DOCS=-1
DEVICE="auto"
FP16_FLAG=""

# Enable fp16 on CUDA if you want:
if command -v nvidia-smi >/dev/null 2>&1; then
  FP16_FLAG="--fp16"
fi

python -m src.ltm.embeddings \
  --batches-dir "$BATCHES_DIR" \
  --out-dir "$OUT_DIR" \
  --model "$MODEL" \
  --pooling "$POOLING" \
  --window "$WINDOW" \
  --stride "$STRIDE" \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --max-docs "$MAX_DOCS" \
  --device "$DEVICE" \
  $FP16_FLAG

