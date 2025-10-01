import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Dict, Any, Iterator, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel


# ---------- I/O (reuses your JSONL.GZ format) ----------
def stream_jsonl_gz(path: str) -> Iterator[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# ---------- Device helpers ----------
def pick_device(device_flag: str) -> torch.device:
    if device_flag == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # Apple Silicon (M-series) — Metal backend
        if getattr(torch.backends, "mps", None) and 
torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_flag)


# ---------- Windowing over pre-tokenized words ----------
def iter_windows(tokens: List[str], window: int, stride: int) -> 
Iterator[List[str]]:
    if window <= 0:
        raise ValueError("window must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if len(tokens) == 0:
        return
    i = 0
    n = len(tokens)
    while i < n:
        yield tokens[i : i + window]
        if i + window >= n:
            break
        i += stride


# ---------- Encode windows in mini-batches ----------
@torch.no_grad()
def encode_windows(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    window_batches: List[List[List[str]]],
    device: torch.device,
    fp16: bool,
    max_length: int,
    pooling: str = "mean",
) -> List[np.ndarray]:
    """
    window_batches: list of batches; each batch is a list of windows; each 
window is list[str] tokens
    returns: list of embeddings for each window (np.float32, dim = 
hidden_size)
    """
    out_vecs: List[np.ndarray] = []
    use_amp = fp16 and (device.type in {"cuda", "mps"})
    autocast_device = "cuda" if device.type == "cuda" else "cpu"  # mps 
not supported by autocast; safe fallback

    for batch in window_batches:
        # Tokenize as pre-split words
        inputs = tokenizer(
            batch,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if use_amp and device.type == "cuda":
            with torch.autocast(device_type=autocast_device, 
dtype=torch.float16):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

        last_hidden = outputs.last_hidden_state  # [B, T, H]
        attn = inputs["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
        summed = (last_hidden * attn).sum(dim=1)               # [B, H]
        denom = attn.sum(dim=1).clamp(min=1.0)                 # [B, 1]
        mean_pooled = summed / denom                           # [B, H]

        # Other pooling options could go here
        if pooling == "cls":
            # Use first token ([CLS]) embedding
            cls_vec = last_hidden[:, 0, :]
            out = cls_vec
        else:
            out = mean_pooled

        out_vecs.extend([v.detach().float().cpu().numpy() for v in out])

        # free memory between batches
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return out_vecs


# ---------- Per-document embedding (aggregate window vectors) ----------
def embed_document(
    tokens: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    fp16: bool,
    window: int,
    stride: int,
    batch_size: int,
    max_length: int,
    pooling: str = "mean",
) -> Optional[np.ndarray]:
    windows = list(iter_windows(tokens, window=window, stride=stride))
    if not windows:
        return None

    # Mini-batch windows for efficiency
    batches: List[List[List[str]]] = []
    for i in range(0, len(windows), batch_size):
        batches.append(windows[i : i + batch_size])

    vecs = encode_windows(
        tokenizer=tokenizer,
        model=model,
        window_batches=batches,
        device=device,
        fp16=fp16,
        max_length=max_length,
        pooling=pooling,
    )
    if not vecs:
        return None

    # Aggregate all window vectors → doc vector
    X = np.stack(vecs, axis=0)  # [num_windows, hidden]
    doc_vec = X.mean(axis=0).astype(np.float32)
    return doc_vec


# ---------- Batch runner over *.jsonl.gz ----------
def process_batch_file(
    batch_path: Path,
    out_dir: Path,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    fp16: bool,
    window: int,
    stride: int,
    batch_size: int,
    max_length: int,
    pooling: str,
    max_docs: int,
) -> Tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = batch_path.stem.replace(".jsonl", "")  # e.g., "batch_00"
    out_npz = out_dir / f"{stem}.npz"             # embeddings matrix + 
metadata
    out_meta = out_dir / f"{stem}.meta.jsonl"     # per-doc metadata

    doc_ids: List[str] = []
    years: List[Optional[int]] = []
    rows: List[np.ndarray] = []

    count = 0
    skipped = 0

    for row in tqdm(stream_jsonl_gz(str(batch_path)), desc=f"Embedding 
{batch_path.name}"):
        if max_docs > 0 and count >= max_docs:
            break

        tokens = row.get("full_text", None)
        if not isinstance(tokens, list) or len(tokens) == 0:
            skipped += 1
            continue

        vec = embed_document(
            tokens=tokens,
            tokenizer=tokenizer,
            model=model,
            device=device,
            fp16=fp16,
            window=window,
            stride=stride,
            batch_size=batch_size,
            max_length=max_length,
            pooling=pooling,
        )

        if vec is None:
            skipped += 1
            continue

        rows.append(vec)
        doc_ids.append(row.get("doc_id", f"doc_{count}"))
        years.append(row.get("rights_year", None))
        count += 1

    if rows:
        X = np.vstack(rows).astype(np.float32)
        np.savez_compressed(out_npz, X=X, doc_ids=np.array(doc_ids, 
dtype=object), rights_year=np.array(years, dtype=object))
        with open(out_meta, "w", encoding="utf-8") as f:
            for i, did in enumerate(doc_ids):
                f.write(json.dumps({"doc_id": did, "rights_year": 
years[i]}) + "\n")

    return count, skipped


def main():
    p = argparse.ArgumentParser(description="Windowed BERT embeddings for 
JSONL.GZ batches")
    p.add_argument("--batches-dir", type=str, required=True, 
help="Directory with *.jsonl.gz files")
    p.add_argument("--out-dir", type=str, required=True, help="Where to 
write .npz + .meta.jsonl")
    p.add_argument("--model", type=str, default="bert-base-uncased", 
help="HF model name (e.g., bert-base-uncased, all-MiniLM-L6-v2)")
    p.add_argument("--pooling", type=str, default="mean", choices=["mean", 
"cls"], help="Pooling over tokens per window")
    p.add_argument("--window", type=int, default=256, help="Number of 
input words per window (pre-tokenized)")
    p.add_argument("--stride", type=int, default=200, help="Stride (words) 
between windows")
    p.add_argument("--batch-size", type=int, default=8, help="Windows per 
forward pass")
    p.add_argument("--max-length", type=int, default=256, help="Tokenizer 
truncation length (wordpieces)")
    p.add_argument("--max-docs", type=int, default=-1, help="Limit docs 
per batch file (-1 for all)")
    p.add_argument("--device", type=str, default="auto", help="auto | cuda 
| mps | cpu")
    p.add_argument("--fp16", action="store_true", help="Enable mixed 
precision (cuda preferred)")
    p.add_argument("--start-batch", type=int, default=None, help="Only 
process files with index >= this (e.g., 0)")
    p.add_argument("--end-batch", type=int, default=None, help="Only 
process files with index <= this (e.g., 9)")
    args = p.parse_args()

    device = pick_device(args.device)

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval().to(device)

    # Optional: half-precision weights on CUDA for memory savings (AMP 
covers activations)
    if args.fp16 and device.type == "cuda":
        model.half()

    batches_dir = Path(args.batches_dir)
    out_dir = Path(args.out_dir)
    files = sorted([p for p in batches_dir.glob("*.jsonl.gz")])

    # Optional slice by numeric index in filename (expects 
batch_00.jsonl.gz naming)
    if args.start_batch is not None or args.end_batch is not None:
        filtered = []
        for f in files:
            # try extract trailing number from e.g. batch_00.jsonl.gz
            digits = "".join(ch for ch in f.stem if ch.isdigit())
            idx = int(digits) if digits else -1
            if (args.start_batch is None or idx >= args.start_batch) and 
(args.end_batch is None or idx <= args.end_batch):
                filtered.append(f)
        files = filtered

    total, skipped = 0, 0
    for fp in files:
        done, bad = process_batch_file(
            batch_path=fp,
            out_dir=out_dir,
            tokenizer=tokenizer,
            model=model,
            device=device,
            fp16=args.fp16,
            window=args.window,
            stride=args.stride,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pooling=args.pooling,
            max_docs=args.max_docs,
        )
        total += done
        skipped += bad

    print(f"✅ Completed. Docs embedded: {total}, skipped: {skipped}. 
Output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

