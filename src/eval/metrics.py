from collections import Counter
from typing import List, Dict

from src.utils.text import normalize_answer, get_tokens


def exact_match(prediction: str, gold_answers: List[str]) -> float:
    norm_pred = normalize_answer(prediction)
    return float(any(normalize_answer(g) == norm_pred for g in gold_answers))


def precision_recall_f1_single(prediction: str, gold: str) -> Dict[str, float]:
    pred_tokens = get_tokens(prediction)
    gold_tokens = get_tokens(gold)

    if not pred_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def precision_recall_f1_max(prediction: str, gold_answers: List[str]) -> Dict[str, float]:
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    for gold in gold_answers:
        cur = precision_recall_f1_single(prediction, gold)
        if cur["f1"] > best["f1"]:
            best = cur

    return best


def f1_score(prediction: str, gold_answers: List[str]) -> float:
    return precision_recall_f1_max(prediction, gold_answers)["f1"]


def _sanitize_for_bertscore(texts: List[str]) -> List[str]:
    return [t if t.strip() else "." for t in texts]


def compute_bertscore(predictions: List[str], references: List[str]) -> Dict[str, float]:
    predictions = _sanitize_for_bertscore(predictions)
    references = _sanitize_for_bertscore(references)

    try:
        from bert_score import score as bert_score_fn

        p_vals, r_vals, f1_vals = bert_score_fn(
            predictions,
            references,
            lang="en",
            verbose=False,
        )
        return {
            "precision": p_vals.mean().item(),
            "recall": r_vals.mean().item(),
            "f1": f1_vals.mean().item(),
        }
    except (AttributeError, TypeError):
        import evaluate

        metric = evaluate.load("bertscore")
        results = metric.compute(
            predictions=predictions,
            references=references,
            lang="en",
        )
        return {
            "precision": sum(results["precision"]) / len(results["precision"]),
            "recall": sum(results["recall"]) / len(results["recall"]),
            "f1": sum(results["f1"]) / len(results["f1"]),
        }


def answer_in_text(gold_answers: List[str], text: str) -> bool:
    norm_text = normalize_answer(text)
    for ans in gold_answers:
        norm_ans = normalize_answer(ans)
        if norm_ans and norm_ans in norm_text:
            return True
    return False


def prediction_in_text(prediction: str, text: str) -> bool:
    norm_pred = normalize_answer(prediction)
    norm_text = normalize_answer(text)
    return bool(norm_pred) and norm_pred in norm_text


def evaluate_all(
    predictions: List[str],
    gold_answers_list: List[List[str]],
    contexts: List[str] | None = None,
) -> Dict[str, float]:
    em_scores = []
    f1_scores = []
    token_precisions = []
    token_recalls = []

    empty_pred_count = 0
    not_found_count = 0
    pred_lengths = []

    answer_in_context_count = 0
    prediction_in_context_count = 0
    hallucination_count = 0
    context_contains_answer_but_model_failed_count = 0

    for i, (pred, golds) in enumerate(zip(predictions, gold_answers_list)):
        em = exact_match(pred, golds)
        prf = precision_recall_f1_max(pred, golds)

        em_scores.append(em)
        f1_scores.append(prf["f1"])
        token_precisions.append(prf["precision"])
        token_recalls.append(prf["recall"])

        pred_tokens = get_tokens(pred)
        pred_lengths.append(len(pred_tokens))

        if not pred.strip():
            empty_pred_count += 1

        if normalize_answer(pred) == "not found":
            not_found_count += 1

        if contexts is not None:
            ctx = contexts[i]
            ans_in_ctx = answer_in_text(golds, ctx)
            pred_in_ctx = prediction_in_text(pred, ctx)

            answer_in_context_count += float(ans_in_ctx)
            prediction_in_context_count += float(pred_in_ctx)

            if (not pred_in_ctx) and em == 0.0 and pred.strip():
                hallucination_count += 1

            if ans_in_ctx and em == 0.0:
                context_contains_answer_but_model_failed_count += 1

    flat_refs = [golds[0] for golds in gold_answers_list]
    bertscore = compute_bertscore(predictions, flat_refs)

    n = len(predictions)

    metrics = {
        "exact_match": sum(em_scores) / n,
        "f1": sum(f1_scores) / n,
        "token_precision": sum(token_precisions) / n,
        "token_recall": sum(token_recalls) / n,
        "bertscore_precision": bertscore["precision"],
        "bertscore_recall": bertscore["recall"],
        "bertscore_f1": bertscore["f1"],
        "empty_prediction_rate": empty_pred_count / n,
        "not_found_rate": not_found_count / n,
        "avg_prediction_length_tokens": sum(pred_lengths) / n,
    }

    if contexts is not None:
        metrics.update(
            {
                "answer_in_context_rate": answer_in_context_count / n,
                "prediction_in_context_rate": prediction_in_context_count / n,
                "hallucination_rate": hallucination_count / n,
                "context_contains_answer_but_model_failed_rate": (
                    context_contains_answer_but_model_failed_count / n
                ),
            }
        )

    return metrics