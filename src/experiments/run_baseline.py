"""
Run the baseline experiment with:
- all 4 model/context conditions
- multiple prompt styles
- extended metrics
- retrieval analysis for NQ
- error analysis
- pairwise deltas between conditions

Usage:
python -m src.experiments.run_baseline --config configs/baseline.yaml --dataset squad
python -m src.experiments.run_baseline --config configs/baseline.yaml --dataset nq
"""

import argparse
import time
from typing import Dict, List, Tuple

from src.utils.io import load_yaml, load_jsonl, save_json, ensure_dir
from src.utils.seed import set_seed
from src.models.flan_t5 import FlanT5Model
from src.prompts.templates import build_prompt
from src.eval.metrics import evaluate_all
from src.eval.error_analysis import build_error_report
from src.eval.retrieval_analysis import compute_retrieval_metrics


def build_prompts(samples: List[Dict], context_key: str, prompt_style: str) -> List[str]:
    return [
        build_prompt(
            context=s[context_key],
            question=s["question"],
            style=prompt_style,
        )
        for s in samples
    ]


def run_condition(
    model: FlanT5Model,
    samples: List[Dict],
    context_key: str,
    batch_size: int,
    prompt_style: str,
) -> Tuple[List[str], List[str]]:
    prompts = build_prompts(samples, context_key=context_key, prompt_style=prompt_style)
    predictions = model.generate(prompts, batch_size=batch_size)
    contexts = [s[context_key] for s in samples]
    return predictions, contexts


def evaluate_predictions(
    predictions: List[str],
    gold_answers_list: List[List[str]],
    contexts: List[str],
) -> Dict[str, float]:
    return evaluate_all(predictions, gold_answers_list, contexts=contexts)


def compute_pairwise_deltas(results_for_prompt: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    condition_names = list(results_for_prompt.keys())
    deltas = {}

    for a in condition_names:
        for b in condition_names:
            if a == b:
                continue

            metrics_a = results_for_prompt[a]["metrics"]
            metrics_b = results_for_prompt[b]["metrics"]

            deltas[f"{a}__vs__{b}"] = {
                "delta_exact_match": metrics_a.get("exact_match", 0.0) - metrics_b.get("exact_match", 0.0),
                "delta_f1": metrics_a.get("f1", 0.0) - metrics_b.get("f1", 0.0),
                "delta_token_precision": metrics_a.get("token_precision", 0.0) - metrics_b.get("token_precision", 0.0),
                "delta_token_recall": metrics_a.get("token_recall", 0.0) - metrics_b.get("token_recall", 0.0),
                "delta_bertscore_f1": metrics_a.get("bertscore_f1", 0.0) - metrics_b.get("bertscore_f1", 0.0),
                "delta_answer_in_context_rate": metrics_a.get("answer_in_context_rate", 0.0) - metrics_b.get("answer_in_context_rate", 0.0),
                "delta_prediction_in_context_rate": metrics_a.get("prediction_in_context_rate", 0.0) - metrics_b.get("prediction_in_context_rate", 0.0),
                "delta_hallucination_rate": metrics_a.get("hallucination_rate", 0.0) - metrics_b.get("hallucination_rate", 0.0),
                "delta_context_contains_answer_but_model_failed_rate": (
                    metrics_a.get("context_contains_answer_but_model_failed_rate", 0.0)
                    - metrics_b.get("context_contains_answer_but_model_failed_rate", 0.0)
                ),
                "delta_avg_prediction_length_tokens": (
                    metrics_a.get("avg_prediction_length_tokens", 0.0)
                    - metrics_b.get("avg_prediction_length_tokens", 0.0)
                ),
                "delta_latency_per_sample_sec": (
                    metrics_a.get("latency_per_sample_sec", 0.0)
                    - metrics_b.get("latency_per_sample_sec", 0.0)
                ),
            }

    return deltas


def print_metrics(metrics: Dict[str, float], elapsed: float) -> None:
    print(
        f"EM={metrics.get('exact_match', 0.0):.4f} | "
        f"F1={metrics.get('f1', 0.0):.4f} | "
        f"TokP={metrics.get('token_precision', 0.0):.4f} | "
        f"TokR={metrics.get('token_recall', 0.0):.4f} | "
        f"BERT-F1={metrics.get('bertscore_f1', 0.0):.4f} | "
        f"AnsInCtx={metrics.get('answer_in_context_rate', 0.0):.4f} | "
        f"PredInCtx={metrics.get('prediction_in_context_rate', 0.0):.4f} | "
        f"Halluc={metrics.get('hallucination_rate', 0.0):.4f} | "
        f"CtxHasAnsButFail={metrics.get('context_contains_answer_but_model_failed_rate', 0.0):.4f} | "
        f"NotFound={metrics.get('not_found_rate', 0.0):.4f} | "
        f"AvgLen={metrics.get('avg_prediction_length_tokens', 0.0):.2f} | "
        f"Latency/sample={metrics.get('latency_per_sample_sec', 0.0):.4f}s | "
        f"time={elapsed:.1f}s"
    )


def main():
    parser = argparse.ArgumentParser(description="Run baseline experiment")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--dataset", type=str, choices=["squad", "nq"], required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    ensure_dir(cfg["output_dir"])

    samples_path = f"{cfg['output_dir']}/{args.dataset}_samples.jsonl"
    print(f"Loading samples from {samples_path} ...")
    samples = load_jsonl(samples_path)
    print(f"Loaded {len(samples)} samples")

    gold_answers_list = [s["gold_answers"] for s in samples]
    prompt_styles = cfg["prompts"]["active_styles"]
    conditions = cfg["experiments"]["conditions"]

    model_cache: Dict[str, FlanT5Model] = {}

    all_results = {
        "dataset": args.dataset,
        "n_samples": len(samples),
        "prompt_styles": prompt_styles,
        "conditions": {},
    }

    all_predictions: Dict[Tuple[str, str], List[str]] = {}

    for prompt_style in prompt_styles:
        print("\n" + "=" * 100)
        print(f"PROMPT STYLE: {prompt_style}")
        print("=" * 100)

        all_results["conditions"][prompt_style] = {}

        for cond in conditions:
            cond_name = cond["name"]
            model_size = cond["model_size"]
            context_size = cond["context_size"]
            context_key = f"context_{context_size}"

            model_cfg = cfg["models"][model_size]
            model_name = model_cfg["name"]

            if model_name not in model_cache:
                print(f"Loading model: {model_name}")
                model_cache[model_name] = FlanT5Model(
                    model_name,
                    max_new_tokens=model_cfg["max_new_tokens"],
                )

            model = model_cache[model_name]

            print("-" * 100)
            print(
                f"Condition: {cond_name} | "
                f"model={model_name} | "
                f"context={context_size} | "
                f"prompt={prompt_style}"
            )

            t0 = time.time()

            predictions, contexts = run_condition(
                model=model,
                samples=samples,
                context_key=context_key,
                batch_size=model_cfg["batch_size"],
                prompt_style=prompt_style,
            )

            elapsed = time.time() - t0

            metrics = evaluate_predictions(
                predictions=predictions,
                gold_answers_list=gold_answers_list,
                contexts=contexts,
            )
            metrics["latency_per_sample_sec"] = elapsed / len(samples) if samples else 0.0

            print_metrics(metrics, elapsed)

            all_results["conditions"][prompt_style][cond_name] = {
                "model": model_name,
                "model_size": model_size,
                "context_size": context_size,
                "prompt_style": prompt_style,
                "metrics": metrics,
                "inference_time_seconds": round(elapsed, 2),
            }

            all_predictions[(prompt_style, cond_name)] = predictions

    all_results["pairwise_deltas"] = {}
    for prompt_style in prompt_styles:
        all_results["pairwise_deltas"][prompt_style] = compute_pairwise_deltas(
            all_results["conditions"][prompt_style]
        )

    if args.dataset == "nq":
        retrieval_metrics = compute_retrieval_metrics(
            samples=samples,
            eval_top_ks=cfg["retrieval"]["eval_top_ks"],
        )
        all_results["retrieval_analysis"] = retrieval_metrics

        print("\n" + "=" * 100)
        print("RETRIEVAL ANALYSIS")
        print("=" * 100)
        for key, value in retrieval_metrics.items():
            print(f"{key}={value:.4f}")

    if cfg.get("error_analysis", {}).get("enabled", False):
        error_reports = {}

        for prompt_style in prompt_styles:
            error_reports[prompt_style] = {}

            for cond in conditions:
                cond_name = cond["name"]
                context_key = f"context_{cond['context_size']}"
                predictions = all_predictions[(prompt_style, cond_name)]

                error_reports[prompt_style][cond_name] = build_error_report(
                    samples=samples,
                    predictions=predictions,
                    context_key=context_key,
                )

        save_json(
            error_reports,
            f"{cfg['output_dir']}/{args.dataset}_error_analysis.json",
        )

    per_sample_predictions = []

    for i, sample in enumerate(samples):
        row = {
            "id": sample["id"],
            "dataset": sample.get("dataset"),
            "question": sample["question"],
            "gold_answers": sample["gold_answers"],
            "context_small": sample.get("context_small"),
            "context_large": sample.get("context_large"),
        }

        if args.dataset == "nq":
            row["retrieved_passages"] = sample.get("retrieved_passages", [])
            row["retrieval_contexts_by_k"] = sample.get("retrieval_contexts_by_k", {})

        for prompt_style in prompt_styles:
            for cond in conditions:
                cond_name = cond["name"]
                row[f"pred__{prompt_style}__{cond_name}"] = all_predictions[(prompt_style, cond_name)][i]

        per_sample_predictions.append(row)

    save_json(all_results, f"{cfg['output_dir']}/{args.dataset}_results.json")
    save_json(per_sample_predictions, f"{cfg['output_dir']}/{args.dataset}_predictions.json")

    print("\n" + "=" * 100)
    print(f"Results saved to {cfg['output_dir']}/")
    print("=" * 100)


if __name__ == "__main__":
    main()