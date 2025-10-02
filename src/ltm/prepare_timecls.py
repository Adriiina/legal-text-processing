import argparse, gzip, json, random
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional
import pandas as pd
from tqdm import tqdm

def stream_jsonl_gz(path: str) -> Iterator[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

def iter_docs(batches_dir: Path):
    for fp in sorted(batches_dir.glob("*.jsonl.gz")):
        for row in stream_jsonl_gz(str(fp)):
            tokens = row.get("full_text", [])
            if not isinstance(tokens, list) or len(tokens) == 0:
                continue
            doc_id = row.get("doc_id")
            year = row.get("rights_year")
            yield doc_id, year, tokens

def decade_bin(year: Optional[int]) -> Optional[str]:
    if year is None:
        return None
    # e.g., 1934 -> "1930s"
    dec = int(year) // 10 * 10
    return f"{dec}s"

def main():
    ap = argparse.ArgumentParser(description="Prepare time-period 
classification dataset (decades)")
    ap.add_argument("--batches-dir", required=True, help="data/batches 
with *.jsonl.gz")
    ap.add_argument("--out-dir", required=True, 
help="data/processed/timecls")
    ap.add_argument("--min-docs-per-class", type=int, default=50, 
help="drop sparse decades")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-docs", type=int, default=-1, help="optional 
limit for quick runs")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, 
exist_ok=True)

    rows = []
    for i, (doc_id, year, tokens) in 
enumerate(tqdm(iter_docs(Path(args.batches_dir)), desc="Scanning")):
        if args.max_docs > 0 and i >= args.max_docs:
            break
        label = decade_bin(year)
        if label is None:
            continue
        text = " ".join(tokens)  # simple join; fine for baselines & BERT
        rows.append({"doc_id": doc_id, "rights_year": year, "label": 
label, "text": text})

    if not rows:
        raise SystemExit("No labeled rows (check rights_year values).")

    df = pd.DataFrame(rows)
    # Drop rare decades
    counts = df["label"].value_counts()
    keep = counts[counts >= args.min_docs_per_class].index
    df = df[df["label"].isin(keep)].reset_index(drop=True)

    # Stratified split by label
    train_idx, val_idx = [], []
    for lab, dfg in df.groupby("label"):
        idx = list(dfg.index)
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * args.val_ratio))
        val_idx.extend(idx[:n_val]); train_idx.extend(idx[n_val:])

    df_train = df.loc[sorted(train_idx)].reset_index(drop=True)
    df_val = df.loc[sorted(val_idx)].reset_index(drop=True)

    df_train.to_csv(out_dir / "train.csv", index=False)
    df_val.to_csv(out_dir / "valid.csv", index=False)

    # Small README
    with open(out_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write(
            f"Time classification dataset\n"
            f"- Labels (decades): {sorted(df['label'].unique())}\n"
            f"- Train: {len(df_train)}  Valid: {len(df_val)}  
(val_ratio={args.val_ratio})\n"
            f"- Dropped classes < {args.min_docs_per_class} docs\n"
        )
    print(f"✅ Wrote {len(df_train)} train / {len(df_val)} valid to 
{out_dir.resolve()}")
if __name__ == "__main__":
    main()

