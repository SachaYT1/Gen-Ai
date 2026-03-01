from datasets import load_dataset
from typing import List, Dict


def load_squad(split: str = "validation") -> List[Dict]:
    """Load SQuAD v1.1 and return a flat list of QA examples.

    Each item contains: id, question, context (gold paragraph), answers (list of strings).
    """
    dataset = load_dataset("squad", split=split)

    samples = []
    for item in dataset:
        samples.append(
            {
                "id": item["id"],
                "question": item["question"],
                "context": item["context"],
                "answers": item["answers"]["text"],
            }
        )
    return samples
