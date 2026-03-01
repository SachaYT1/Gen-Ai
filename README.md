# The Power of Context

> Can a Smaller Model with Rich Context Outperform a Stronger Model with Sparse Context?

Innopolis University — Generative AI course project

---

## Quick Start

### 1. Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | Tested on 3.10 – 3.12 |
| **Java 11+** | Required only for NQ retrieval (Pyserini/Lucene). Not needed for SQuAD. |

**Install Java (only needed for NQ dataset):**

```bash
# macOS
brew install openjdk@21

# Ubuntu / Debian
sudo apt install openjdk-21-jdk
```

### 2. Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the SQuAD baseline (no Java needed)

```bash
# Step 1 — download SQuAD and prepare samples (large/small contexts)
python -m src.experiments.build_samples --config configs/baseline.yaml --dataset squad

# Step 2 — run inference (Flan-T5-Base vs Flan-T5-Large) and evaluate
python -m src.experiments.run_baseline --config configs/baseline.yaml --dataset squad
```

### 4. Run the NQ baseline (requires Java + ~7 GB for BM25 index)

```bash
# Step 1 — download NQ, build BM25 index, retrieve passages
python -m src.experiments.build_samples --config configs/baseline.yaml --dataset nq

# Step 2 — run inference and evaluate
python -m src.experiments.run_baseline --config configs/baseline.yaml --dataset nq
```

---

## Project Structure

```
configs/baseline.yaml          ← experiment hyper-parameters
src/
  data/                        ← dataset loaders (SQuAD, NQ) + sampling
  models/flan_t5.py            ← Flan-T5 wrapper (batched greedy generation)
  prompts/templates.py         ← deterministic prompt template
  retrieval/                   ← BM25 retriever (Pyserini) + DPR corpus
  eval/metrics.py              ← EM, token-F1, BERTScore
  experiments/
    build_samples.py           ← Step 1: prepare data
    run_baseline.py            ← Step 2: inference + evaluation
  utils/                       ← seed, text normalisation, I/O helpers
outputs/baseline/              ← generated results (git-ignored)
reports/baseline_report.md     ← baseline implementation report
```

---

## Configuration

All parameters live in `configs/baseline.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | 42 | Global random seed |
| `n_samples` | 500 | Questions sampled per dataset |
| `models.weak.name` | `google/flan-t5-base` | Weak model (250 M params) |
| `models.strong.name` | `google/flan-t5-large` | Strong model (780 M params) |
| `retrieval.top_k_large` | 5 | Passages for large context (NQ) |
| `retrieval.top_k_small` | 1 | Passages for small context (NQ) |

---

## Datasets

| Dataset | Auto-downloaded? | Size |
|---------|-----------------|------|
| **SQuAD v1.1** | Yes (HuggingFace `datasets`) | ~35 MB |
| **NQ Open** | Yes (HuggingFace `datasets`) | ~4 MB |
| **Wikipedia DPR index** | Yes (Pyserini, first run) | ~7 GB |

All datasets are cached by HuggingFace / Pyserini after the first download.

---

## Evaluation Metrics

- **Exact Match (EM)** — strict normalised string match
- **Token-level F1** — precision/recall over normalised tokens
- **BERTScore F1** — semantic similarity via BERT embeddings

---

## Hardware Notes

- SQuAD baseline runs fine on CPU (≈ 15–30 min for 500 samples).
- GPU (CUDA) or Apple Silicon (MPS) accelerates inference significantly.
- Flan-T5-Large fits in ~4 GB VRAM; Flan-T5-Base needs ~1.5 GB.
- If using MPS (Mac), `pip install torch` with the default PyPI build works.

---

## Team

| Member | Responsibilities |
|--------|-----------------|
| **Aleksandr (Sasha)** | Environment setup, model loading, inference pipeline |
| **Karim** | Data loading, preprocessing, BM25 retrieval module, context construction |
| **Ilya** | Prompts, integration, experiments, metrics, analysis, report |
