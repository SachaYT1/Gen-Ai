# Baseline Implementation Report

**Project:** The Power of Context — Can a Smaller Model with Rich Context Outperform a Stronger Model with Sparse Context?

**Authors:** Ilya Maximov, Aleksandr Gavkovskii, Karim Zakriov — Innopolis University

**Date:** March 2026

---

## 1  Introduction

This report describes the baseline implementation for our study of the trade-off between model capability and context quality in Question Answering (QA).
The central research question is: *Can a smaller language model, supplied with more relevant context, generate more accurate answers than a larger model that sees only minimal context?*

The baseline establishes a controlled experimental pipeline that we will extend with additional model pairs, retrieval strategies, and ablations in later phases.

---

## 2  Experimental Setup

### 2.1  Datasets

| Dataset | Source | Role | Split |
|---------|--------|------|-------|
| **SQuAD v1.1** | `datasets.load_dataset("squad")` | In-context (extractive) QA | validation (10 570 examples) |
| **Natural Questions Open** | `datasets.load_dataset("nq_open")` | Open-domain QA | validation (3 610 examples) |
| **Wikipedia DPR Corpus** | Pyserini pre-built index `wikipedia-dpr` | Knowledge base for NQ retrieval | ~21 M passages |

### 2.2  Models

| Role | Model | Parameters |
|------|-------|------------|
| **Weak** | `google/flan-t5-base` | 250 M |
| **Strong** | `google/flan-t5-large` | 780 M |

Both models are encoder-decoder seq2seq architectures fine-tuned on a mixture of instruction-following tasks.
Inference uses greedy decoding (`do_sample=False`, `max_new_tokens=64`).

### 2.3  Experimental Conditions

| Condition | Model | Context |
|-----------|-------|---------|
| **A  (Weak + Large Context)** | Flan-T5-Base | SQuAD: full gold paragraph; NQ: top-5 BM25 passages concatenated |
| **B  (Strong + Small Context)** | Flan-T5-Large | SQuAD: first sentence of gold paragraph; NQ: top-1 BM25 passage |

We sample **N = 500** questions from each dataset (fixed `seed=42`) and evaluate both conditions on the same question set.

### 2.4  Prompt Template

All experiments use a single deterministic prompt:

```
Context: {context}
Question: {question}
Answer:
```

---

## 3  Implementation Architecture

### 3.1  Project Structure

```
├── configs/
│   └── baseline.yaml            # all hyper-parameters in one place
├── src/
│   ├── data/
│   │   ├── squad.py             # SQuAD v1.1 loader
│   │   ├── nq.py                # NQ Open loader
│   │   └── sampling.py          # random sampling with fixed seed
│   ├── models/
│   │   └── flan_t5.py           # Flan-T5 wrapper (load + batched generation)
│   ├── prompts/
│   │   └── templates.py         # prompt formatting
│   ├── retrieval/
│   │   ├── bm25_pyserini.py     # BM25 retriever (Pyserini + Lucene)
│   │   └── corpus_dpr.py        # Wikipedia DPR index management
│   ├── eval/
│   │   └── metrics.py           # EM, token-F1, BERTScore
│   ├── experiments/
│   │   ├── build_samples.py     # Step 1: prepare samples with contexts
│   │   └── run_baseline.py      # Step 2: inference + evaluation
│   └── utils/
│       ├── seed.py              # reproducibility (random, numpy, torch)
│       ├── text.py              # normalisation, tokenisation, sentence split
│       └── io.py                # YAML / JSON / JSONL I/O helpers
├── outputs/baseline/            # generated at runtime
├── reports/
│   └── baseline_report.md       # this document
├── configs/baseline.yaml
├── requirements.txt
└── README.md
```

### 3.2  Pipeline Overview

The experiment runs in two sequential steps:

1. **`build_samples`** — loads the dataset, samples N questions, constructs large/small contexts, and saves everything as JSONL.
2. **`run_baseline`** — loads the saved samples, runs inference under both conditions, computes metrics, and writes results to JSON.

This two-step design lets us decouple data preparation (which involves retrieval for NQ) from model inference, making each stage independently reproducible and cacheable.

---

## 4  Data Pipeline Details

### 4.1  SQuAD Context Construction

- **Large context:** the full gold paragraph supplied with each question.
- **Small context:** the first sentence of the gold paragraph, extracted via NLTK `sent_tokenize`.

### 4.2  NQ Context Construction (Open-Domain)

- The Wikipedia passage corpus is indexed with BM25 via Pyserini's `LuceneSearcher`.
- For each question we retrieve **top-5** passages.
- **Large context** = concatenation of all 5 passages (separated by double newlines).
- **Small context** = the single top-1 passage.

---

## 5  Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Exact Match (EM)** | 1 if the normalised prediction exactly matches any gold answer, else 0 |
| **Token-level F1** | Harmonic mean of precision and recall over normalised tokens; max over gold answers |
| **BERTScore F1** | Semantic similarity between prediction and gold answer using contextual BERT embeddings |

Normalisation (for EM and F1) follows the standard SQuAD evaluation script: lowercase, strip articles (*a*, *an*, *the*), remove punctuation, and collapse whitespace.

---

## 6  Preliminary Results

> Results will be inserted here after running the baseline experiment.
> Placeholder table:

| Dataset | Condition | EM | F1 | BERTScore F1 |
|---------|-----------|----|----|-------------|
| SQuAD | A (Weak + Large) | — | — | — |
| SQuAD | B (Strong + Small) | — | — | — |
| NQ | A (Weak + Large) | — | — | — |
| NQ | B (Strong + Small) | — | — | — |

---

## 7  Next Steps

1. **Run the SQuAD baseline** (no retrieval required) and populate the results table.
2. **Set up Java + Pyserini** and run the NQ baseline with BM25 retrieval.
3. **Statistical analysis:** paired bootstrap test on F1 differences to assess significance.
4. **Qualitative error analysis:** inspect cases where the weak model wins or loses.
5. **Ablations:** vary context size (top-1, top-3, top-5, top-10) and try additional model pairs (e.g., Flan-T5-XL as the strong model).

---

## 8  Reproducibility

All experiments are fully reproducible via:

```bash
# Step 1: prepare SQuAD samples
python -m src.experiments.build_samples --config configs/baseline.yaml --dataset squad

# Step 2: run inference and evaluation
python -m src.experiments.run_baseline --config configs/baseline.yaml --dataset squad
```

Random seed (`42`) is set globally for Python, NumPy, and PyTorch.
All intermediate data (samples, predictions) is persisted to `outputs/baseline/` in JSON/JSONL format.
