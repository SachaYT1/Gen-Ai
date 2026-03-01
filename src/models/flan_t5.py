import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List
from tqdm import tqdm


class FlanT5Model:
    """Thin wrapper around a HuggingFace Flan-T5 checkpoint for greedy QA generation."""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 64,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()

    def generate(self, prompts: List[str], batch_size: int = 8) -> List[str]:
        all_answers: List[str] = []

        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch = prompts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            decoded = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            all_answers.extend(decoded)

        return all_answers

    def __repr__(self) -> str:
        return f"FlanT5Model({self.model_name}, device={self.device})"
