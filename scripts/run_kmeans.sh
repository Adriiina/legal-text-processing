#!/usr/bin/env bash
set -euo pipefail

EMB_DIR="data/processed/embeddings"
OUT_DIR="models/kmeans"

# Try a few K values; adjust based on inertia/silhouette + 
interpretability
python -m src.ltm.clustering \
  --emb-dir "$EMB_DIR" \
  --out-dir "$OUT_DIR" \
  --k 50 75 100 \
  --init k-means++ \
  --max-iter 300 \
  --n-init 10 \
  --random-state 42

