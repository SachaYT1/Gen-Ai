import random
from typing import List, Dict


def sample_questions(data: List[Dict], n: int, seed: int = 42) -> List[Dict]:
    """Randomly sample *n* questions from *data* with a fixed seed."""
    rng = random.Random(seed)
    if n >= len(data):
        return data
    return rng.sample(data, n)
