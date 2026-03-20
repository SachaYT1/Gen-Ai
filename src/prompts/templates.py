from typing import Literal

PromptStyle = Literal[
    "basic",
    "use_only_context",
    "not_found_if_missing",
    "few_shot_brief",
]

PROMPT_TEMPLATES = {
    "basic": (
        "Context: {context}\n"
        "Question: {question}\n"
        "Answer:"
    ),
    "use_only_context": (
        "Answer the question using only the provided context.\n"
        "Give a short factual answer.\n\n"
        "Context: {context}\n"
        "Question: {question}\n"
        "Answer:"
    ),
    "not_found_if_missing": (
        "Answer the question using only the provided context.\n"
        "If the answer is not present in the context, answer exactly: not found\n\n"
        "Context: {context}\n"
        "Question: {question}\n"
        "Answer:"
    ),
    "few_shot_brief": (
        "Answer the question using only the provided context.\n"
        "Give a short factual answer.\n\n"
        "Example 1\n"
        "Context: Paris is the capital of France.\n"
        "Question: What is the capital of France?\n"
        "Answer: Paris\n\n"
        "Example 2\n"
        "Context: The Pacific Ocean is the largest ocean on Earth.\n"
        "Question: What is the largest ocean on Earth?\n"
        "Answer: Pacific Ocean\n\n"
        "Now answer the next question.\n"
        "Context: {context}\n"
        "Question: {question}\n"
        "Answer:"
    ),
}


def build_prompt(context: str, question: str, style: str = "basic") -> str:
    if style not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt style: {style}")
    return PROMPT_TEMPLATES[style].format(context=context, question=question)