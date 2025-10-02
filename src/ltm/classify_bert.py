import argparse, os, json
from pathlib import Path
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, 
classification_report, confusion_matrix


class TextDataset(Dataset):
    def __init__(self, df, tokenizer, label2id, max_length=256):
        self.texts = df["text"].tolist()
        self.labels = [label2id[l] for l in df["label"].tolist()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred, id2label):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1m = f1_score(labels, preds, average="macro")
    report = classification_report(labels, preds, 
target_names=[id2label[i] for i in range(len(id2label))], 
output_dict=True)
    cm = confusion_matrix(labels, preds).tolist()
    return {"accuracy": acc, "macro_f1": f1m, "report": report, 
"confusion_matrix": cm}


def main():
    ap = argparse.ArgumentParser(description="Fine-tune BERT for time 
period classification")
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--valid-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--patience", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, 
exist_ok=True)

    df_train = pd.read_csv(args.train_csv).dropna(subset=["text","label"])
    df_val   = pd.read_csv(args.valid_csv).dropna(subset=["text","label"])

    labels = sorted(df_train["label"].unique().tolist())
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for l,i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    ds_train = TextDataset(df_train, tokenizer, label2id, args.max_length)
    ds_val   = TextDataset(df_val, tokenizer, label2id, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "hf_ckpt"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # use mixed precision on GPU
        report_to=[]  # disable wandb/tensorboard by default
    )

    def hf_metrics(eval_pred): 
        return {k:v for k,v in compute_metrics(eval_pred, 
id2label).items() if not isinstance(v,(dict,list))}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        compute_metrics=hf_metrics,
        
callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    trainer.train()

    # Evaluate final model
    metrics = trainer.evaluate()
    with open(out_dir / "bert_eval.json", "w") as f:
        json.dump(metrics, f, indent=2)

    trainer.save_model(out_dir / "bert_model")
    tokenizer.save_pretrained(out_dir / "bert_model")

    print("✅ BERT fine-tune complete. Metrics:", metrics)


if __name__ == "__main__":
    main()

