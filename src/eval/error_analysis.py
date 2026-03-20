from typing import Dict, List

from src.eval.metrics import exact_match, f1_score
from src.utils.text import normalize_answer


def _answer_in_text(gold_answers: List[str], text: str) -> bool:
    norm_text = normalize_answer(text)
    for ans in gold_answers:
        norm_ans = normalize_answer(ans)
        if norm_ans and norm_ans in norm_text:
            return True
    return False


def classify_error(prediction: str, gold_answers: List[str], context: str) -> str:
    pred_norm = normalize_answer(prediction)

    if exact_match(prediction, gold_answers) == 1.0:
        return "correct_exact"

    if f1_score(prediction, gold_answers) > 0.0:
        return "partially_correct"

    answer_present = _answer_in_text(gold_answers, context)

    if pred_norm == "not found":
        if answer_present:
            return "missed_answer_present_in_context"
        return "correctly_abstained"

    if not answer_present:
        return "answer_absent_in_context"

    if prediction.strip() and not answer_present:
        return "hallucination"

    if answer_present and prediction.strip():
        return "wrong_answer_despite_answer_in_context"

    return "other"


def build_error_report(samples: List[Dict], predictions: List[str], context_key: str) -> Dict:
    rows = []
    counts = {}

    for sample, pred in zip(samples, predictions):
        label = classify_error(pred, sample["gold_answers"], sample[context_key])
        counts[label] = counts.get(label, 0) + 1

        rows.append(
            {
                "id": sample["id"],
                "question": sample["question"],
                "gold_answers": sample["gold_answers"],
                "context_key": context_key,
                "prediction": pred,
                "error_type": label,
            }
        )

    total = len(rows)
    distribution = {
        key: {
            "count": value,
            "share": value / total if total else 0.0,
        }
        for key, value in sorted(counts.items(), key=lambda x: x[0])
    }

    return {
        "summary": distribution,
        "rows": rows,
    }