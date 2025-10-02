import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def load_topics(topics_csv: Path, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(topics_csv)
    # Expect columns: topic,rank,word
    if not {"topic", "rank", "word"}.issubset(df.columns):
        raise ValueError("Expected columns: topic, rank, word")
    df = df.sort_values(["topic", "rank"])
    gloss = (df.groupby("topic")["word"]
               .apply(lambda s: ", ".join(list(s)[:top_n]))
               .reset_index()
               .rename(columns={"topic": "topic_id", "word": 
"top_words"}))
    return gloss

def summarize_timelines(tl_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(tl_csv)
    if "rights_year" not in df.columns:
        raise ValueError("Timelines CSV must contain 'rights_year'")
    topic_cols = [c for c in df.columns if c != "rights_year"]
    # Melt to long
    long = df.melt(id_vars=["rights_year"], value_vars=topic_cols,
                   var_name="topic_id", value_name="prevalence")
    # Ensure ints where possible
    try:
        long["topic_id"] = long["topic_id"].astype(int)
    except Exception:
        pass
    grp = long.groupby("topic_id")
    stats = grp.apply(lambda g: pd.Series({
        "mean_prevalence": float(np.nanmean(g["prevalence"].to_numpy())),
        "peak_year": int(g.loc[g["prevalence"].idxmax()]["rights_year"]) 
if g["prevalence"].notna().any() else np.nan,
        "first_year": int(g["rights_year"].min()) if g.shape[0] else 
np.nan,
        "last_year": int(g["rights_year"].max()) if g.shape[0] else 
np.nan,
    })).reset_index()
    return stats

def main():
    ap = argparse.ArgumentParser(description="LDA topic gloss + timeline 
summary")
    ap.add_argument("--topics-csv", required=True, 
help="models/lda/lda_topics_k{K}.csv")
    ap.add_argument("--timelines-csv", required=True, 
help="models/lda/lda_timelines_k{K}.csv")
    ap.add_argument("--out-csv", required=True, help="Output CSV path")
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()

    topics_csv = Path(args.topics_csv)
    tl_csv = Path(args.timelines_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    gloss = load_topics(topics_csv, args.top_n)
    stats = summarize_timelines(tl_csv)
    merged = gloss.merge(stats, on="topic_id", 
how="left").sort_values("topic_id")
    merged.to_csv(out_csv, index=False)
    print(f"✅ Saved {out_csv.resolve()}")

if __name__ == "__main__":
    main()

