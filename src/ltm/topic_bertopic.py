import argparse
import gzip
import json
from pathlib import Path
from typing import Dict, Any, Iterator, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

from bertopic import BERTopic
from umap import UMAP
import hdbscan


# ---------- I/O ----------
def stream_jsonl_gz(path: str) -> Iterator[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def load_embeddings(emb_dir: Path) -> Tuple[np.ndarray, np.ndarray, 
np.ndarray, List[Path]]:
    files = sorted(emb_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz found in {emb_dir}")
    Xs, ids, years = [], [], []
    for fp in tqdm(files, desc=f"Loading embeddings ({len(files)} 
batches)"):
        arr = np.load(fp, allow_pickle=True)
        Xs.append(arr["X"].astype(np.float32, copy=False))
        ids.append(arr["doc_ids"])
        years.append(arr["rights_year"])
    X = np.vstack(Xs).astype(np.float32, copy=False)
    doc_ids = np.concatenate(ids, axis=0)
    rights_year = np.concatenate(years, axis=0)
    return X, doc_ids, rights_year, files


def build_doc_texts_for_ids(
    needed_ids: np.ndarray,
    batches_dir: Path,
    max_docs: int = -1,
) -> List[str]:
    """
    Join token lists into simple space-separated strings for c-TF-IDF.
    Streams batches and collects texts for the doc_ids in 'needed_ids',
    preserving the order of 'needed_ids'.
    """
    need: Set[str] = set(map(str, needed_ids))
    buf: Dict[str, str] = {}

    batch_files = sorted(batches_dir.glob("*.jsonl.gz"))
    for fp in tqdm(batch_files, desc="Reconstructing doc strings from 
tokens"):
        for row in stream_jsonl_gz(str(fp)):
            did = str(row.get("doc_id", ""))
            if did in need and did not in buf:
                toks = row.get("full_text", [])
                if isinstance(toks, list):
                    # light normalization: join tokens
                    buf[did] = " ".join(toks)
                if max_docs > 0 and len(buf) >= max_docs:
                    break
        if max_docs > 0 and len(buf) >= max_docs:
            break

    # Now emit in the exact order of needed_ids
    docs: List[str] = []
    missing = 0
    for did in needed_ids:
        s = buf.get(str(did))
        if s is None:
            docs.append("")  # placeholder to keep alignment; topic words 
may be weaker
            missing += 1
        else:
            docs.append(s)
    if missing:
        print(f"⚠️ Missing {missing} doc texts (kept empty placeholders to 
preserve alignment).")
    return docs


def main():
    ap = argparse.ArgumentParser(description="BERTopic over precomputed 
embeddings + token-joined docs")
    ap.add_argument("--emb-dir", type=str, required=True, help="Directory 
with *.npz from embeddings.py")
    ap.add_argument("--batches-dir", type=str, required=True, 
help="Directory with *.jsonl.gz for doc text rebuild")
    ap.add_argument("--out-dir", type=str, required=True, help="Output 
directory for BERTopic artifacts")
    ap.add_argument("--min-cluster-size", type=int, default=30)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--nr-topics", type=str, default=None, help="e.g. 
'auto' or an integer like '100'")
    ap.add_argument("--umap-n-neighbors", type=int, default=15)
    ap.add_argument("--umap-n-components", type=int, default=5)
    ap.add_argument("--umap-min-dist", type=float, default=0.0)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--max-docs", type=int, default=-1, help="Limit docs 
(debug)")
    ap.add_argument("--calc-prob", action="store_true", help="Calculate 
topic probabilities (slower, more RAM)")
    args = ap.parse_args()

    emb_dir = Path(args.emb_dir)
    batches_dir = Path(args.batches_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load embeddings + metadata
    X, doc_ids, years, files_loaded = load_embeddings(emb_dir)
    if args.max_docs > 0 and X.shape[0] > args.max_docs:
        X = X[: args.max_docs]
        doc_ids = doc_ids[: args.max_docs]
        years = years[: args.max_docs]

    # 2) Build lightweight documents from tokens (for c-TF-IDF topic 
words)
    docs = build_doc_texts_for_ids(doc_ids, batches_dir=batches_dir, 
max_docs=args.max_docs)

    # 3) Configure UMAP + HDBSCAN (memory-friendly defaults)
    umap_model = UMAP(
        n_neighbors=args.umap_n_neighbors,
        n_components=args.umap_n_components,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=args.random_state,
        verbose=False,
    )
    hdb_model = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
        core_dist_n_jobs=1,
    )

    # 4) Instantiate BERTopic
    nr_topics = args.nr_topics
    if nr_topics is not None and nr_topics.isdigit():
        nr_topics = int(nr_topics)

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdb_model,
        nr_topics=nr_topics,         # 'auto' or int or None
        calculate_probabilities=args.calc_prob,
        verbose=True,
        low_memory=True,
    )

    # 5) Fit
    topics, probs = topic_model.fit_transform(docs, embeddings=X)

    # 6) Save model + artifacts
    # Model (can be large). If too big, skip to save space.
    model_dir = out_dir / "bertopic_model"
    try:
        topic_model.save(model_dir, save_embedding_model=False)
    except Exception as e:
        print(f"⚠️ Could not save full BERTopic model: {e}")

    # Topic info table
    df_info = topic_model.get_topic_info()
    df_info.to_csv(out_dir / "bertopic_topics.csv", index=False)

    # Per-doc assignments
    df_doc = pd.DataFrame({
        "doc_id": doc_ids,
        "rights_year": years,
        "topic": topics
    })
    df_doc.to_csv(out_dir / "bertopic_doc_topics.csv", index=False)

    # Optional: probabilities (can be huge). Save only if requested.
    if args.calc_prob and probs is not None:
        np.savez_compressed(out_dir / "bertopic_doc_probs.npz", 
P=probs.astype(np.float32))

    # 7) Yearly timelines (mean prob per topic per year)
    # If we didn't compute probs, synthesize a 1-hot approximation from 
hard labels
    if probs is not None:
        P = probs
    else:
        # Build [N, T] one-hot on observed topics (excluding outliers = 
-1)
        T = df_info["Topic"].max() + 1 if not df_info.empty else 0
        P = np.zeros((len(topics), max(T, 0)), dtype=np.float32)
        for i, t in enumerate(topics):
            if t >= 0 and t < P.shape[1]:
                P[i, t] = 1.0

    dfP = pd.DataFrame(P)
    dfP["rights_year"] = years
    df_year = (
        dfP.dropna(subset=["rights_year"])
           .groupby("rights_year")
           .mean()
           .reset_index()
           .sort_values("rights_year")
    )
    df_year.to_csv(out_dir / "bertopic_timelines.csv", index=False)

    # 8) Small README
    with open(out_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write(
            "BERTopic outputs\n"
            f"- Embeddings from: {emb_dir}\n"
            f"- Batches used for doc texts: {batches_dir}\n"
            f"- Topics table: bertopic_topics.csv\n"
            f"- Per-doc topics: bertopic_doc_topics.csv\n"
            f"- Timelines (mean probs by year): bertopic_timelines.csv\n"
            f"- Model dir (if saved): {model_dir.name}\n"
        )

    print(f"✅ BERTopic complete. Outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()

