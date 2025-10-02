# Description
Developed a scalable NLP pipeline in Python to process 5GB+/13k+ JSONL legal texts, implementing tokenization, normalization, TF-IDF vectorization, and topic modeling with LDA and BERTopic. Research difference between different text type classifiers.

# Legal Text Processing

Combined repository for two JHU projects on U.S. legal texts:

1. **Topic Modeling & Temporal Trends** (01/2025–05/2025)  
   - Built a scalable NLP pipeline for a 5GB+/13k+ JSONL legal corpus: 
tokenization, normalization, TF-IDF, and topic modeling with LDA and 
BERTopic (UMAP+HDBSCAN).  
   - Engineered memory-efficient batch processing (generators/`tqdm`) with 
optional GPU acceleration; produced reproducible t-SNE/PCA visualizations.  
   - Quantified semantic drift of legal terminology across decades using 
embedding-based similarity metrics.  
   - Developed interactive timelines linking topic shifts to major legal 
reforms.  

2. **Modern Text Classification & Bayesian Lexical Inference** 
(01/2025–07/2025)  
   - Designed streaming NLP pipelines for large-scale legal text 
classification using BERT embeddings, pseudo-labeling, and 
LR/k-NN/MLP/fine-tuned BERT.  
   - Ran GPU-accelerated PyTorch/Scikit-learn workflows with experiment 
tracking; achieved ~97% accuracy with BERT-based models.  
   - Built linguistic utilities with NLTK/regex/finite-state grammars and 
CFG-based syntactic validators.  
   - Implemented a Bayesian model of lexical inference to estimate P(c|w) 
under different priors; produced visual analyses in 
Matplotlib/scikit-image.  

---

## Repo Layout (planned)
- `src/ltm/` — Shared Python library (data IO, preprocessing, embeddings, 
topic models, classifiers)  
- `projects/01_topic_trends/` — LDA & BERTopic pipelines, temporal 
analyses, semantic drift  
- `projects/02_classification_bayesian/` — BERT-based classifiers, 
pseudo-labeling, Bayesian lexical inference  
- `data/` — Local-only data (ignored by git by default)  
- `models/`, `reports/`, `notebooks/`, `scripts/` — Outputs, analysis, and 
helpers  

---

## Quickstart

```bash
# create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Notes
- Large files are ignored; consider Git LFS if you need to version 
corpora.
- GPU is optional but recommended for Transformers.

