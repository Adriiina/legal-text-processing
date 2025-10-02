#!/usr/bin/env bash
set -euo pipefail

KMEANS_DIR="models/kmeans"
OUT_DIR="models/kmeans"   # write summary alongside metrics

python -m src.ltm.kmeans_report \
  --kmeans-dir "$KMEANS_DIR" \
  --out-dir "$OUT_DIR"

