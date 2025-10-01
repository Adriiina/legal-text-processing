import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_npz_batches(emb_dir: Path) -> Tuple[np.ndarray, np.ndarray, 
np.ndarray, List[Path]]:
    """
    Load and stack *.npz batches saved by embeddings.py
    Returns:
      X: [N, D] float32
      doc_ids: [N] object
      years: [N] object (may include None)
      files: list of file paths loaded (for provenance)
    """
    files = sorted(emb_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {emb_dir}")

    Xs, ids, years = [], [], []
    for fp in tqdm(files, desc=f"Loading {len(files)} batch file(s)"):
        data = np.load(fp, allow_pickle=True)
        Xi = data["X"]
        di = data["doc_ids"]
        yi = data["rights_year"]
        # sanity
        assert Xi.shape[0] == di.shape[0] == yi.shape[0], f"Row mismatch 
in {fp}"
        Xs.append(Xi.astype(np.float32, copy=False))
        ids.append(di)
        years.append(yi)

    X = np.vstack(Xs).astype(np.float32, copy=False)
    doc_ids = np.concatenate(ids, axis=0)
    rights_year = np.concatenate(years, axis=0)
    return X, doc_ids, rights_year, files


def run_kmeans(
    X: np.ndarray,
    k: int,
    init: str = "k-means++",
    max_iter: int = 300,
    n_init: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    km = KMeans(
        n_clusters=k,
        init=init,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
        verbose=0,
    )
    labels = km.fit_predict(X)
    inertia = float(km.inertia_)
    # Silhouette is expensive; compute on a subset if N is huge
    sil = None
    try:
        if X.shape[0] > 50000:
            # sample to ~20k for speed, stratified-ish by stepping
            idx = np.arange(0, X.shape[0], max(1, X.shape[0] // 20000))
            sil = float(silhouette_score(X[idx], labels[idx], 
metric="euclidean"))
        else:
            sil = float(silhouette_score(X, labels, metric="euclidean"))
    except Exception as e:
        sil = None

    return {
        "labels": labels.astype(np.int32),
        "inertia": inertia,
        "silhouette": sil,
        "centers": km.cluster_centers_.astype(np.float32),
    }


def save_outputs(
    out_dir: Path,
    k: int,
    doc_ids: np.ndarray,
    years: np.ndarray,
    labels: np.ndarray,
    inertia: float,
    silhouette: float | None,
    files_loaded: List[Path],
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Labels table (CSV)
    labels_csv = out_dir / f"kmeans_k{k}_labels.csv"
    with open(labels_csv, "w", encoding="utf-8") as f:
        f.write("doc_id,rights_year,cluster\n")
        for did, yr, lab in zip(doc_ids, years, labels):
            f.write(f"{did},{'' if yr is None else yr},{int(lab)}\n")

    # Metrics JSON
    metrics_json = out_dir / f"kmeans_k{k}_metrics.json"
    meta = {
        "k": k,
        "inertia": inertia,
        "silhouette": silhouette,
        "n_docs": int(doc_ids.shape[0]),
        "sources": [str(p.name) for p in files_loaded],
    }
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # (Optional) cluster centers for later inspection
    centers_npz = out_dir / f"kmeans_k{k}_centers.npz"
    # Save as npz for compactness
    np.savez_compressed(centers_npz, k=k, centers=None)  # centers omitted 
by default to save space

    # Small README for this run
    readme = out_dir / f"kmeans_k{k}_README.txt"
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            f"KMeans results (k={k})\n"
            f"- Docs: {doc_ids.shape[0]}\n"
            f"- Inertia: {inertia:.2f}\n"
            f"- Silhouette: {silhouette if silhouette is not None else 
'n/a'}\n"
            f"- Labels CSV: {labels_csv.name}\n"
            f"- Metrics JSON: {metrics_json.name}\n"
        )


def main():
    ap = argparse.ArgumentParser(description="Cluster BERT embeddings with 
KMeans")
    ap.add_argument("--emb-dir", type=str, required=True, help="Directory 
with *.npz from embeddings.py")
    ap.add_argument("--out-dir", type=str, required=True, help="Output 
directory for labels/metrics")
    ap.add_argument("--k", type=int, nargs="+", required=True, help="One 
or more K values, e.g., --k 50 75 100")
    ap.add_argument("--init", type=str, default="k-means++")
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--n-init", type=int, default=10)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    emb_dir = Path(args.emb_dir)
    out_dir = Path(args.out_dir)

    X, doc_ids, years, files_loaded = load_npz_batches(emb_dir)

    for k in args.k:
        print(f"\n== KMeans: k={k} ==")
        res = run_kmeans(
            X=X,
            k=k,
            init=args.init,
            max_iter=args.max_iter,
            n_init=args.n_init,
            random_state=args.random_state,
        )
        save_outputs(
            out_dir=out_dir,
            k=k,
            doc_ids=doc_ids,
            years=years,
            labels=res["labels"],
            inertia=res["inertia"],
            silhouette=res["silhouette"],
            files_loaded=files_loaded,
        )
        print(f"Saved labels/metrics for k={k} to {out_dir}")

    print("✅ Done.")


if __name__ == "__main__":
    main()

