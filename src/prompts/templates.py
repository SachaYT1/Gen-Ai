PROMPT_TEMPLATE = "Context: {context}\nQuestion: {question}\nAnswer:"


def build_prompt(context: str, question: str) -> str:
    """Format context + question into the deterministic QA prompt."""
    return PROMPT_TEMPLATE.format(context=context, question=question)
