from datasets import load_dataset
from typing import List, Dict


def load_nq(split: str = "validation") -> List[Dict]:
    """Load Natural Questions Open dataset.

    Each item contains: id, question, answers (list of strings).
    Context must be retrieved externally (BM25 / DPR).
    """
    dataset = load_dataset("nq_open", split=split)

    samples = []
    for i, item in enumerate(dataset):
        samples.append(
            {
                "id": f"nq_{i}",
                "question": item["question"],
                "answers": item["answer"],
            }
        )
    return samples
