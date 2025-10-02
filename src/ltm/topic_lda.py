import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import vstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# ---------- I/O ----------
def stream_jsonl_gz(path: str) -> Iterator[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def iter_docs(batches_dir: Path) -> Iterator[Tuple[str, Optional[int], 
List[str]]]:
    files = sorted(batches_dir.glob("*.jsonl.gz"))
    for fp in files:
        for row in stream_jsonl_gz(str(fp)):
            tokens = row.get("full_text", None)
            if not isinstance(tokens, list) or len(tokens) == 0:
                continue
            yield (row.get("doc_id", None), row.get("rights_year", None), 
tokens)


# ---------- Vocab building (1st pass) ----------
def build_vocab(batches_dir: Path, vocab_size: int, min_freq: int) -> 
List[str]:
    cnt = Counter()
    for _, _, tokens in tqdm(iter_docs(batches_dir), desc="Scanning tokens 
for vocab"):
        cnt.update(tokens)
    # filter
    if min_freq > 1:
        cnt = Counter({w: c for w, c in cnt.items() if c >= min_freq})
    # keep most common
    vocab = [w for w, _ in cnt.most_common(vocab_size)]
    return vocab


# ---------- Vectorization (2nd pass) ----------
def make_vectorizer(vocab: List[str]) -> CountVectorizer:
    # Identity tokenizer/preprocessor to accept pre-tokenized lists
    return CountVectorizer(
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        lowercase=False,
        vocabulary=vocab,
        dtype=np.float32,
        min_df=1,
    )


def vectorize_corpus(batches_dir: Path, vectorizer: CountVectorizer) -> 
Tuple[csr_matrix, List[str], List[Optional[int]]]:
    X_blocks = []
    doc_ids: List[str] = []
    years: List[Optional[int]] = []
    for did, yr, tokens in tqdm(iter_docs(batches_dir), 
desc="Vectorizing"):
        doc_ids.append(did if did is not None else f"doc_{len(doc_ids)}")
        years.append(yr)
        X_blocks.append(vectorizer.transform([tokens]))
        # occasional progress concat to limit Python list growth
        if len(X_blocks) >= 2000:
            X_blocks = [vstack(X_blocks, format="csr")]
    if len(X_blocks) == 0:
        raise RuntimeError("No documents vectorized.")
    X = vstack(X_blocks, format="csr") if len(X_blocks) > 1 else 
X_blocks[0]
    return X, doc_ids, years


# ---------- Topic labeling ----------
def top_words(components: np.ndarray, feature_names: List[str], n_top: 
int) -> List[List[str]]:
    topics = []
    for k in range(components.shape[0]):
        idx = np.argsort(components[k])[::-1][:n_top]
        topics.append([feature_names[j] for j in idx])
    return topics


def main():
    ap = argparse.ArgumentParser(description="LDA topic modeling over 
pre-tokenized JSONL.GZ batches")
    ap.add_argument("--batches-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--n-topics", type=int, default=50)
    ap.add_argument("--vocab-size", type=int, default=30000)
    ap.add_argument("--min-freq", type=int, default=3)
    ap.add_argument("--n-top-words", type=int, default=15)
    ap.add_argument("--max-docs", type=int, default=-1, help="Limit docs 
(debug)")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    batches_dir = Path(args.batches_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Vocabulary
    vocab = build_vocab(batches_dir, vocab_size=args.vocab_size, 
min_freq=args.min_freq)
    vec = make_vectorizer(vocab)
    feature_names = list(vec.get_feature_names_out())

    # 2) Vectorize
    X, doc_ids, years = vectorize_corpus(batches_dir, vec)
    if args.max_docs > 0 and X.shape[0] > args.max_docs:
        X = X[: args.max_docs]
        doc_ids = doc_ids[: args.max_docs]
        years = years[: args.max_docs]

    # 3) Fit LDA (online for better scalability)
    lda = LatentDirichletAllocation(
        n_components=args.n_topics,
        learning_method="online",
        batch_size=1024,
        doc_topic_prior=None,
        topic_word_prior=None,
        random_state=args.random_state,
        n_jobs=-1,
        evaluate_every=-1,
    )
    doc_topic = lda.fit_transform(X)  # [N, K], row-normalized

    # 4) Save top words
    topics = top_words(lda.components_, feature_names, 
n_top=args.n_top_words)
    with open(out_dir / f"lda_topics_k{args.n_topics}.csv", "w", 
encoding="utf-8") as f:
        f.write("topic,rank,word\n")
        for k, words in enumerate(topics):
            for r, w in enumerate(words, start=1):
                f.write(f"{k},{r},{w}\n")

    # 5) Save per-doc topic distribution (argmax + optional probs)
    top_idx = np.argmax(doc_topic, axis=1).astype(int)
    df_doc = pd.DataFrame({
        "doc_id": doc_ids,
        "rights_year": years,
        "top_topic": top_idx,
    })
    df_doc.to_csv(out_dir / f"lda_doc_topics_k{args.n_topics}.csv", 
index=False)

    # optional: save dense probs compressed
    np.savez_compressed(out_dir / f"lda_doc_topic_k{args.n_topics}.npz", 
doc_topic=doc_topic.astype(np.float32))

    # 6) Yearly timelines (mean topic probability per year)
    df = pd.DataFrame(doc_topic)
    df["rights_year"] = years
    df_year = (
        df.dropna(subset=["rights_year"])
          .groupby("rights_year")
          .mean()
          .reset_index()
          .sort_values("rights_year")
    )
    df_year.to_csv(out_dir / f"lda_timelines_k{args.n_topics}.csv", 
index=False)

    # 7) Small README
    with open(out_dir / f"lda_k{args.n_topics}_README.txt", "w", 
encoding="utf-8") as f:
        f.write(
            f"LDA results (K={args.n_topics})\n"
            f"- Docs: {len(doc_ids)}\n"
            f"- Vocab: {len(vocab)} (min_freq={args.min_freq})\n"
            f"- Outputs:\n"
            f"  * lda_topics_k{args.n_topics}.csv (top words per topic)\n"
            f"  * lda_doc_topics_k{args.n_topics}.csv (top topic per 
doc)\n"
            f"  * lda_doc_topic_k{args.n_topics}.npz (full doc-topic 
matrix)\n"
            f"  * lda_timelines_k{args.n_topics}.csv (mean topic prob by 
year)\n"
        )

    print(f"✅ LDA complete. Outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()

