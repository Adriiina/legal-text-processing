import argparse
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def parse_top_words_from_representation(rep: str, top_n: int) -> 
List[str]:
    """
    BERTopic's get_topic_info() -> 'Representation' column contains a 
string of
    top terms joined by commas (implementation-dependent). We try a few 
splits.
    """
    if not isinstance(rep, str) or not rep.strip():
        return []
    # Prefer comma split first
    parts = [p.strip() for p in rep.split(",") if p.strip()]
    if len(parts) < 2:  # fallback to whitespace split
        parts = [p for p in re.split(r"\s+", rep.strip()) if p]
    return parts[:top_n]


def summarize_timelines(df_tl: pd.DataFrame, topic_col_prefix: str = "") 
-> pd.DataFrame:
    """
    Input df_tl has columns: rights_year, <topic cols 0..T-1 or arbitrary 
ints>
    Returns long-form summary with topic stats.
    """
    df = df_tl.copy()
    year_col = "rights_year"
    if year_col not in df.columns:
        raise ValueError("Timelines file must contain 'rights_year' 
column.")
    topic_cols = [c for c in df.columns if c != year_col]

    # Long format: year, topic, value
    df_long = df.melt(id_vars=[year_col], value_vars=topic_cols,
                      var_name="topic", value_name="prevalence")
    # Normalize topic ids to int if possible
    try:
        df_long["topic"] = df_long["topic"].astype(int)
    except Exception:
        pass

    # Summaries
    grouped = df_long.groupby("topic")
    stats = grouped.apply(lambda g: pd.Series({
        "mean_prevalence": float(np.nanmean(g["prevalence"].to_numpy())),
        "peak_year": int(g.loc[g["prevalence"].idxmax()][year_col]) if 
g["prevalence"].notna().any() else np.nan,
        "first_year": int(g[year_col].min()) if g.shape[0] else np.nan,
        "last_year": int(g[year_col].max()) if g.shape[0] else np.nan,
    })).reset_index()

    return stats


def main():
    ap = argparse.ArgumentParser(description="BERTopic topic gloss + 
timeline summary exporter")
    ap.add_argument("--topics-csv", type=str, required=True, help="Path to 
bertopic_topics.csv")
    ap.add_argument("--timelines-csv", type=str, required=True, help="Path 
to bertopic_timelines.csv")
    ap.add_argument("--out-csv", type=str, required=True, help="Output CSV 
path")
    ap.add_argument("--top-n", type=int, default=10, help="Top-N words to 
keep in gloss")
    args = ap.parse_args()

    topics_csv = Path(args.topics_csv)
    timelines_csv = Path(args.timelines_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load
    df_topics = pd.read_csv(topics_csv)
    df_tl = pd.read_csv(timelines_csv)

    # Expected columns in topics CSV from BERTopic.get_topic_info():
    # ['Topic', 'Count', 'Name', 'Representation', ...]
    if "Topic" not in df_topics.columns:
        raise ValueError("topics CSV must contain 'Topic' column")
    if "Representation" not in df_topics.columns:
        # Some versions store c-TF-IDF words spread across columns; try to 
synthesize
        rep_cols = [c for c in df_topics.columns if c not in {"Topic", 
"Count", "Name"}]
        df_topics["Representation"] = 
df_topics[rep_cols].astype(str).agg(", ".join, axis=1)

    # Parse gloss
    df_topics["top_words"] = df_topics["Representation"].apply(
        lambda s: ", ".join(parse_top_words_from_representation(s, 
args.top_n))
    )

    # Summaries from timelines
    df_stats = summarize_timelines(df_tl)  # topic, mean_prevalence, 
peak_year, first_year, last_year

    # Merge
    merged = df_topics.merge(df_stats, left_on="Topic", right_on="topic", 
how="left")
    merged = merged.drop(columns=["topic"]).rename(columns={
        "Topic": "topic_id",
        "Count": "doc_count",
        "Name": "topic_name"
    })

    # Sort by doc_count desc, then topic_id
    if "doc_count" in merged.columns:
        merged = merged.sort_values(["doc_count", "topic_id"], 
ascending=[False, True])
    else:
        merged = merged.sort_values(["topic_id"])

    # Save
    merged.to_csv(out_csv, index=False)
    print(f"✅ Saved: {out_csv.resolve()}")


if __name__ == "__main__":
    main()

