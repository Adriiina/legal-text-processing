#!/usr/bin/env bash
set -euo pipefail

K=50
IN_DIR="models/lda"
OUT_CSV="models/lda/lda_gloss_timelines_k${K}.csv"
TOPN=10

python -m src.ltm.lda_gloss \
  --topics-csv "$IN_DIR/lda_topics_k${K}.csv" \
  --timelines-csv "$IN_DIR/lda_timelines_k${K}.csv" \
  --out-csv "$OUT_CSV" \
  --top-n "$TOPN"

