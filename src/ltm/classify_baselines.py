import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, 
classification_report, confusion_matrix

def load_split(train_csv: Path, valid_csv: Path):
    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(valid_csv)
    # ensure no NaNs
    df_tr = df_tr.dropna(subset=["text", "label"]); df_va = 
df_va.dropna(subset=["text", "label"])
    Xtr, ytr = df_tr["text"].tolist(), df_tr["label"].tolist()
    Xva, yva = df_va["text"].tolist(), df_va["label"].tolist()
    labels = sorted(df_tr["label"].unique().tolist())
    return (Xtr, ytr, Xva, yva, labels)

def train_vectorizer(Xtr, max_features=100000, min_df=2, 
ngram_range=(1,2)):
    vec = TfidfVectorizer(max_features=max_features, min_df=min_df, 
ngram_range=ngram_range)
    Xtr_t = vec.fit_transform(Xtr)
    return vec, Xtr_t

def evaluate(model, Xva_t, yva, labels, out_dir: Path, tag: str):
    yp = model.predict(Xva_t)
    acc = accuracy_score(yva, yp)
    f1m = f1_score(yva, yp, average="macro")
    report = classification_report(yva, yp, labels=labels, 
output_dict=True, zero_division=0)
    cm = confusion_matrix(yva, yp, labels=labels)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{tag}_metrics.json", "w", encoding="utf-8") as 
f:
        json.dump({"accuracy": acc, "macro_f1": f1m, "labels": labels, 
"report": report}, f, indent=2)
    np.savetxt(out_dir / f"{tag}_confusion_matrix.csv", cm, delimiter=",", 
fmt="%d")
    print(f"[{tag}] acc={acc:.4f}  macroF1={f1m:.4f}")

def main():
    ap = argparse.ArgumentParser(description="Time period baselines 
(TF-IDF → LR/MLP)")
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--valid-csv", required=True)
    ap.add_argument("--out-dir", required=True, 
help="models/timecls_baselines")
    ap.add_argument("--max-features", type=int, default=100000)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--ngram", type=int, nargs=2, default=(1,2))
    args = ap.parse_args()

    Xtr, ytr, Xva, yva, labels = load_split(Path(args.train_csv), 
Path(args.valid_csv))
    print(f"Labels: {labels}  | train={len(ytr)}  valid={len(yva)}")

    vec, Xtr_t = train_vectorizer(Xtr, max_features=args.max_features, 
min_df=args.min_df, ngram_range=tuple(args.ngram))
    Xva_t = vec.transform(Xva)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, 
exist_ok=True)
    # Save vectorizer vocabulary (optional)
    import joblib; joblib.dump(vec, out_dir / "tfidf_vectorizer.joblib")

    # Logistic Regression (saga, multi_class='multinomial')
    lr = LogisticRegression(
        solver="saga", penalty="l2", C=2.0,
        max_iter=2000, n_jobs=-1, verbose=0, multi_class="multinomial"
    ).fit(Xtr_t, ytr)
    evaluate(lr, Xva_t, yva, labels, out_dir, tag="lr")

    # MLP (simple)
    mlp = MLPClassifier(hidden_layer_sizes=(512,), activation="relu", 
batch_size=256, max_iter=20, random_state=42)
    mlp.fit(Xtr_t, ytr)
    evaluate(mlp, Xva_t, yva, labels, out_dir, tag="mlp")

if __name__ == "__main__":
    main()

