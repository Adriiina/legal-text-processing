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
            toks = row.get("full_text", [])
            if not isinstance(toks, list) or not toks:
                continue
            yield {
                "doc_id": row.get("doc_id"),
                "rights_year": row.get("rights_year"),
                "domain": row.get("domain"),      # may be None if using 
external labels
                "text": " ".join(toks),
            }

def main():
    ap = argparse.ArgumentParser(description="Prepare legal-domain 
classification dataset")
    ap.add_argument("--batches-dir", required=True, help="data/batches 
with *.jsonl.gz")
    ap.add_argument("--out-dir", required=True, 
help="data/processed/domaincls")
    ap.add_argument("--labels-csv", default=None, help="Optional external 
labels: CSV with doc_id,label")
    ap.add_argument("--min-docs-per-class", type=int, default=50)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-docs", type=int, default=-1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, 
exist_ok=True)

    # 1) Load texts (and maybe embedded domain)
    rows: List[Dict[str, Any]] = []
    for i, row in enumerate(tqdm(iter_docs(Path(args.batches_dir)), 
desc="Scanning")):
        if args.max_docs > 0 and i >= args.max_docs:
            break
        rows.append(row)
    if not rows:
        raise SystemExit("No documents found.")

    df = pd.DataFrame(rows)

    # 2) Merge labels
    if args.labels_csv:
        lab = pd.read_csv(args.labels_csv)[["doc_id","label"]]
        df = df.merge(lab, on="doc_id", how="left")
    else:
        # Expect 'domain' column in batches
        if "domain" not in df.columns:
            raise SystemExit("No labels-csv provided and no 'domain' in 
batches. Provide --labels-csv.")
        df["label"] = df["domain"]

    # 3) Clean & filter
    df = df.dropna(subset=["text","label"]).reset_index(drop=True)
    counts = df["label"].value_counts()
    keep = counts[counts >= args.min_docs_per_class].index
    df = df[df["label"].isin(keep)].reset_index(drop=True)
    if df.empty:
        raise SystemExit("All classes filtered by min-docs-per-class; 
lower the threshold.")

    # 4) Stratified split
    train_idx, val_idx = [], []
    for lab, dfg in df.groupby("label"):
        idx = list(dfg.index)
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * args.val_ratio))
        val_idx.extend(idx[:n_val]); train_idx.extend(idx[n_val:])
    df_train = df.loc[sorted(train_idx)].reset_index(drop=True)
    df_val   = df.loc[sorted(val_idx)].reset_index(drop=True)

    df_train.to_csv(out_dir / "train.csv", index=False)
    df_val.to_csv(out_dir / "valid.csv", index=False)

    with open(out_dir / "README.txt","w",encoding="utf-8") as f:
        f.write(
            f"Domain classification dataset\n"
            f"- Classes: {sorted(df['label'].unique())}\n"
            f"- Train: {len(df_train)}  Valid: {len(df_val)} 
(val_ratio={args.val_ratio})\n"
            f"- Dropped classes < {args.min_docs_per_class} docs\n"
        )
    print(f"✅ Wrote train/valid to {out_dir.resolve()}")

if __name__ == "__main__":
    main()

