"""Run the baseline experiment: generate answers under both conditions and evaluate.

Usage:
    python -m src.experiments.run_baseline --config configs/baseline.yaml --dataset squad
    python -m src.experiments.run_baseline --config configs/baseline.yaml --dataset nq
"""

import argparse
import time

from src.utils.io import load_yaml, load_jsonl, save_json, ensure_dir
from src.utils.seed import set_seed
from src.models.flan_t5 import FlanT5Model
from src.prompts.templates import build_prompt
from src.eval.metrics import evaluate_all


def run_condition(model, samples, context_key, batch_size):
    prompts = [
        build_prompt(s[context_key], s["question"]) for s in samples
    ]
    return model.generate(prompts, batch_size=batch_size)


def main():
    parser = argparse.ArgumentParser(description="Run baseline experiment")
    parser.add_argument(
        "--config", type=str, default="configs/baseline.yaml"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["squad", "nq"], required=True
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    ensure_dir(cfg["output_dir"])

    samples_path = f"{cfg['output_dir']}/{args.dataset}_samples.jsonl"
    print(f"Loading samples from {samples_path} ...")
    samples = load_jsonl(samples_path)
    print(f"Loaded {len(samples)} samples")

    gold_answers_list = [s["gold_answers"] for s in samples]

    # ---- Condition A: Weak model + Large context ----
    sep = "=" * 60
    weak_cfg = cfg["models"]["weak"]
    print(f"\n{sep}")
    print(f"Condition A: {weak_cfg['name']} + Large Context")
    print(sep)

    weak_model = FlanT5Model(
        weak_cfg["name"], max_new_tokens=weak_cfg["max_new_tokens"]
    )
    t0 = time.time()
    preds_a = run_condition(
        weak_model, samples, "context_large", weak_cfg["batch_size"]
    )
    time_a = time.time() - t0
    del weak_model

    metrics_a = evaluate_all(preds_a, gold_answers_list)
    print(
        f"  EM={metrics_a['exact_match']:.4f}  "
        f"F1={metrics_a['f1']:.4f}  "
        f"BERTScore-F1={metrics_a['bertscore_f1']:.4f}  "
        f"({time_a:.1f}s)"
    )

    # ---- Condition B: Strong model + Small context ----
    strong_cfg = cfg["models"]["strong"]
    print(f"\n{sep}")
    print(f"Condition B: {strong_cfg['name']} + Small Context")
    print(sep)

    strong_model = FlanT5Model(
        strong_cfg["name"], max_new_tokens=strong_cfg["max_new_tokens"]
    )
    t0 = time.time()
    preds_b = run_condition(
        strong_model, samples, "context_small", strong_cfg["batch_size"]
    )
    time_b = time.time() - t0
    del strong_model

    metrics_b = evaluate_all(preds_b, gold_answers_list)
    print(
        f"  EM={metrics_b['exact_match']:.4f}  "
        f"F1={metrics_b['f1']:.4f}  "
        f"BERTScore-F1={metrics_b['bertscore_f1']:.4f}  "
        f"({time_b:.1f}s)"
    )

    # ---- Save results ----
    results = {
        "dataset": args.dataset,
        "n_samples": len(samples),
        "condition_a": {
            "name": "Weak-LargeContext",
            "model": weak_cfg["name"],
            "context": "large",
            "metrics": metrics_a,
            "inference_time_seconds": round(time_a, 2),
        },
        "condition_b": {
            "name": "Strong-SmallContext",
            "model": strong_cfg["name"],
            "context": "small",
            "metrics": metrics_b,
            "inference_time_seconds": round(time_b, 2),
        },
    }
    save_json(results, f"{cfg['output_dir']}/{args.dataset}_results.json")

    per_sample = []
    for s, pa, pb in zip(samples, preds_a, preds_b):
        per_sample.append(
            {
                "id": s["id"],
                "question": s["question"],
                "gold_answers": s["gold_answers"],
                "pred_weak_large": pa,
                "pred_strong_small": pb,
            }
        )
    save_json(
        per_sample,
        f"{cfg['output_dir']}/{args.dataset}_predictions.json",
    )

    print(f"\nResults saved to {cfg['output_dir']}/")


if __name__ == "__main__":
    main()
