#!/usr/bin/env bash
set -euo pipefail

IN_DIR="models/bertopic"
OUT_CSV="models/bertopic/bertopic_gloss_timelines.csv"
TOPN=10

python -m src.ltm.bertopic_gloss \
  --topics-csv "$IN_DIR/bertopic_topics.csv" \
  --timelines-csv "$IN_DIR/bertopic_timelines.csv" \
  --out-csv "$OUT_CSV" \
  --top-n "$TOPN"

