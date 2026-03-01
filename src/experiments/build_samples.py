"""Build experimental samples with large/small contexts for each dataset.

Usage:
    python -m src.experiments.build_samples --config configs/baseline.yaml --dataset squad
    python -m src.experiments.build_samples --config configs/baseline.yaml --dataset nq
    python -m src.experiments.build_samples --config configs/baseline.yaml --dataset all
"""

import argparse

from tqdm import tqdm

from src.utils.io import load_yaml, save_jsonl, ensure_dir
from src.utils.seed import set_seed
from src.utils.text import get_first_sentence
from src.data.squad import load_squad
from src.data.nq import load_nq
from src.data.sampling import sample_questions


def build_squad_samples(data, n_samples, seed):
    sampled = sample_questions(data, n_samples, seed)
    samples = []
    for item in tqdm(sampled, desc="Building SQuAD samples"):
        samples.append(
            {
                "id": item["id"],
                "dataset": "squad",
                "question": item["question"],
                "gold_answers": item["answers"],
                "context_large": item["context"],
                "context_small": get_first_sentence(item["context"]),
            }
        )
    return samples


def build_nq_samples(
    data, retriever, n_samples, seed, top_k_large=5, top_k_small=1
):
    sampled = sample_questions(data, n_samples, seed)
    queries = [item["question"] for item in sampled]
    all_results = retriever.batch_retrieve(queries, top_k=top_k_large)

    samples = []
    for item, results in zip(sampled, all_results):
        context_large = "\n\n".join(
            r["text"] for r in results[:top_k_large]
        )
        context_small = results[0]["text"] if results else ""

        samples.append(
            {
                "id": item["id"],
                "dataset": "nq",
                "question": item["question"],
                "gold_answers": item["answers"],
                "context_large": context_large,
                "context_small": context_small,
            }
        )
    return samples


def main():
    parser = argparse.ArgumentParser(description="Build experimental samples")
    parser.add_argument(
        "--config", type=str, default="configs/baseline.yaml"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["squad", "nq", "all"],
        default="all",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    ensure_dir(cfg["output_dir"])

    if args.dataset in ("squad", "all"):
        print("Loading SQuAD dataset...")
        squad_data = load_squad(cfg["datasets"]["squad"]["split"])
        print(f"Loaded {len(squad_data)} SQuAD questions")

        squad_samples = build_squad_samples(
            squad_data, cfg["n_samples"], cfg["seed"]
        )
        out = f"{cfg['output_dir']}/squad_samples.jsonl"
        save_jsonl(squad_samples, out)
        print(f"Saved {len(squad_samples)} SQuAD samples -> {out}")

    if args.dataset in ("nq", "all"):
        print("Loading NQ dataset...")
        nq_data = load_nq(cfg["datasets"]["nq"]["split"])
        print(f"Loaded {len(nq_data)} NQ questions")

        print("Initialising BM25 retriever (first run downloads the index)...")
        from src.retrieval.bm25_pyserini import BM25Retriever

        retriever = BM25Retriever(cfg["retrieval"]["index_name"])

        nq_samples = build_nq_samples(
            nq_data,
            retriever,
            cfg["n_samples"],
            cfg["seed"],
            cfg["retrieval"]["top_k_large"],
            cfg["retrieval"]["top_k_small"],
        )
        retriever.close()

        out = f"{cfg['output_dir']}/nq_samples.jsonl"
        save_jsonl(nq_samples, out)
        print(f"Saved {len(nq_samples)} NQ samples -> {out}")


if __name__ == "__main__":
    main()
