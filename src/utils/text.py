import re
import string
import nltk
from typing import List


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.

    Follows the standard SQuAD evaluation normalization.
    """
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def get_tokens(s: str) -> List[str]:
    return normalize_answer(s).split()


def get_first_sentence(text: str) -> str:
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        sentences = nltk.sent_tokenize(text)
    return sentences[0] if sentences else text
