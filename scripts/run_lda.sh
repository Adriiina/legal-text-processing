#!/usr/bin/env bash
set -euo pipefail

BATCHES_DIR="data/batches"
OUT_DIR="models/lda"
K=50
VOCAB=30000
MINFREQ=3
TOPW=15

python -m src.ltm.topic_lda \
  --batches-dir "$BATCHES_DIR" \
  --out-dir "$OUT_DIR" \
  --n-topics "$K" \
  --vocab-size "$VOCAB" \
  --min-freq "$MINFREQ" \
  --n-top-words "$TOPW" \
  --random-state 42

