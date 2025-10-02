#!/usr/bin/env bash
set -euo pipefail

EMB_DIR="data/processed/embeddings"
BATCHES_DIR="data/batches"
OUT_DIR="models/bertopic"

# Tunables
MIN_CLUSTER_SIZE=30
MIN_SAMPLES=0            # use 0 to let HDBSCAN infer; or set None by 
removing the flag
NR_TOPICS="auto"         # or an integer like "100"
UMAP_N_NEIGHBORS=15
UMAP_N_COMPONENTS=5
UMAP_MIN_DIST=0.0
CALC_PROB=              # set to "--calc-prob" if you want per-doc 
probabilities saved

python -m src.ltm.topic_bertopic \
  --emb-dir "$EMB_DIR" \
  --batches-dir "$BATCHES_DIR" \
  --out-dir "$OUT_DIR" \
  --min-cluster-size "$MIN_CLUSTER_SIZE" \
  --umap-n-neighbors "$UMAP_N_NEIGHBORS" \
  --umap-n-components "$UMAP_N_COMPONENTS" \
  --umap-min-dist "$UMAP_MIN_DIST" \
  --nr-topics "$NR_TOPICS" \
  --random-state 42 \
  ${CALC_PROB:+--calc-prob}

