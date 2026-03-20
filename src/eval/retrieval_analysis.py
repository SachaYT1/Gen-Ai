from typing import Dict, List

from src.utils.text import normalize_answer


def _answer_in_text(gold_answers: List[str], text: str) -> bool:
    norm_text = normalize_answer(text)
    for ans in gold_answers:
        norm_ans = normalize_answer(ans)
        if norm_ans and norm_ans in norm_text:
            return True
    return False


def compute_retrieval_metrics(samples: List[Dict], eval_top_ks: List[int]) -> Dict[str, float]:
    metrics = {}
    n = len(samples)

    for k in eval_top_ks:
        hit_count = 0
        for sample in samples:
            ctx_map = sample.get("retrieval_contexts_by_k", {})
            context = ctx_map.get(str(k), "")
            if _answer_in_text(sample["gold_answers"], context):
                hit_count += 1

        metrics[f"answer_in_context_rate@{k}"] = hit_count / n if n else 0.0
        metrics[f"recall@{k}"] = hit_count / n if n else 0.0

    return metrics