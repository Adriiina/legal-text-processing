import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_metrics(out_dir: Path) -> pd.DataFrame:
    rows = []
    for fp in sorted(out_dir.glob("kmeans_k*_metrics.json")):
        with open(fp, "r", encoding="utf-8") as f:
            meta = json.load(f)
        rows.append({
            "k": int(meta["k"]),
            "inertia": float(meta["inertia"]),
            "silhouette": (None if meta.get("silhouette") in [None, "n/a"] 
else float(meta["silhouette"])),
            "n_docs": int(meta["n_docs"]),
            "file": fp.name,
        })
    if not rows:
        raise FileNotFoundError(f"No kmeans_k*_metrics.json files found in 
{out_dir}")
    df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    return df


def elbow_k(k: np.ndarray, inertia: np.ndarray) -> int:
    """
    Simple 'knee' heuristic:
    - Draw line from first (k0, I0) to last (kN, IN)
    - Compute perpendicular distance of each point to this line
    - Pick k with max distance
    """
    x1, y1 = k[0], inertia[0]
    x2, y2 = k[-1], inertia[-1]
    dx, dy = (x2 - x1), (y2 - y1)
    denom = np.sqrt(dx*dx + dy*dy)
    if denom == 0:
        return int(k[0])

    dists = []
    for xi, yi in zip(k, inertia):
        num = abs(dy*xi - dx*yi + x2*y1 - y2*x1)
        dists.append(num / denom)
    idx = int(np.argmax(dists))
    return int(k[idx])


def best_silhouette_k(k: np.ndarray, sil: np.ndarray) -> int:
    valid = ~np.isnan(sil)
    if not valid.any():
        return int(k[len(k)//2])  # fallback
    kk = k[valid]
    ss = sil[valid]
    return int(kk[np.argmax(ss)])


def plot_elbow(df: pd.DataFrame, savepath: Path) -> None:
    plt.figure()
    plt.plot(df["k"], df["inertia"], marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("KMeans Elbow")
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()


def plot_silhouette(df: pd.DataFrame, savepath: Path) -> None:
    if df["silhouette"].notna().any():
        plt.figure()
        plt.plot(df["k"], df["silhouette"], marker="o")
        plt.xlabel("k")
        plt.ylabel("Silhouette")
        plt.title("KMeans Silhouette")
        plt.tight_layout()
        plt.savefig(savepath, dpi=150)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Summarize KMeans metrics and 
recommend k")
    ap.add_argument("--kmeans-dir", type=str, required=True, 
help="Directory with kmeans_k*_metrics.json")
    ap.add_argument("--out-dir", type=str, required=False, help="Where to 
write summary files (default: same dir)")
    args = ap.parse_args()

    kdir = Path(args.kmeans_dir)
    out_dir = Path(args.out_dir) if args.out_dir else kdir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(kdir)

    # Save CSV summary
    summary_csv = out_dir / "kmeans_summary.csv"
    df.to_csv(summary_csv, index=False)

    # Compute recommendations
    k_arr = df["k"].to_numpy()
    inertia_arr = df["inertia"].to_numpy()
    sil_arr = df["silhouette"].to_numpy(dtype=float)
    sil_arr = np.where(pd.isna(sil_arr), np.nan, sil_arr)

    k_elbow = elbow_k(k_arr, inertia_arr)
    k_sil = best_silhouette_k(k_arr, sil_arr)

    # A simple compromise: pick k between elbow and silhouette (median of 
the two)
    k_compromise = int(np.median([k_elbow, k_sil]))

    rec = {
        "n_candidates": int(len(df)),
        "k_elbow": k_elbow,
        "k_best_silhouette": k_sil,
        "k_compromise": k_compromise,
    }

    with open(out_dir / "kmeans_recommendation.json", "w", 
encoding="utf-8") as f:
        json.dump(rec, f, indent=2)

    # Plots
    plot_elbow(df, out_dir / "kmeans_elbow.png")
    plot_silhouette(df, out_dir / "kmeans_silhouette.png")

    print("=== KMeans Summary ===")
    print(df.to_string(index=False))
    print("\nRecommendations:", rec)
    print(f"\nSaved: {summary_csv.name}, kmeans_recommendation.json, 
kmeans_elbow.png, kmeans_silhouette.png in {out_dir.resolve()}")


if __name__ == "__main__":
    main()

