"""QA evaluation metrics: Exact Match, token-level F1, and BERTScore."""

from collections import Counter
from typing import List, Dict

from src.utils.text import normalize_answer, get_tokens


def exact_match(prediction: str, gold_answers: List[str]) -> float:
    norm_pred = normalize_answer(prediction)
    return float(
        any(normalize_answer(g) == norm_pred for g in gold_answers)
    )


def f1_score(prediction: str, gold_answers: List[str]) -> float:
    """Max token-level F1 over all acceptable gold answers."""
    pred_tokens = get_tokens(prediction)

    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = get_tokens(gold)

        if not pred_tokens and not gold_tokens:
            best_f1 = max(best_f1, 1.0)
            continue
        if not pred_tokens or not gold_tokens:
            continue

        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            continue

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


def _sanitize_for_bertscore(texts: List[str]) -> List[str]:
    """Replace empty / whitespace-only strings with a dot.

    The ``bert_score`` library calls a deprecated tokenizer method
    (``build_inputs_with_special_tokens``) for empty inputs, which
    crashes on newer ``transformers`` versions.
    """
    return [t if t.strip() else "." for t in texts]


def compute_bertscore(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    predictions = _sanitize_for_bertscore(predictions)
    references = _sanitize_for_bertscore(references)

    try:
        from bert_score import score as bert_score_fn

        P, R, F1 = bert_score_fn(
            predictions, references, lang="en", verbose=False
        )
        return {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item(),
        }
    except (AttributeError, TypeError):
        import evaluate

        metric = evaluate.load("bertscore")
        results = metric.compute(
            predictions=predictions, references=references, lang="en"
        )
        return {
            "precision": sum(results["precision"]) / len(results["precision"]),
            "recall": sum(results["recall"]) / len(results["recall"]),
            "f1": sum(results["f1"]) / len(results["f1"]),
        }


def evaluate_all(
    predictions: List[str], gold_answers_list: List[List[str]]
) -> Dict[str, float]:
    """Compute EM, F1, and BERTScore for a full list of (pred, golds) pairs."""
    em_scores = [
        exact_match(p, g) for p, g in zip(predictions, gold_answers_list)
    ]
    f1_scores = [
        f1_score(p, g) for p, g in zip(predictions, gold_answers_list)
    ]

    flat_refs = [golds[0] for golds in gold_answers_list]
    bertscore = compute_bertscore(predictions, flat_refs)

    return {
        "exact_match": sum(em_scores) / len(em_scores),
        "f1": sum(f1_scores) / len(f1_scores),
        "bertscore_precision": bertscore["precision"],
        "bertscore_recall": bertscore["recall"],
        "bertscore_f1": bertscore["f1"],
    }
