import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from src.constants import USR_END_TKN


class Model(torch.nn.Module):
    def __init__(self, model=None, tokenizer=None) -> None:
        super().__init__()
        self.model = model or T5ForConditionalGeneration.from_pretrained("t5-small")
        self.tokenizer = tokenizer or T5Tokenizer.from_pretrained(
            "t5-small", additional_special_tokens=[USR_END_TKN], extra_ids=0
        )

    @classmethod
    def from_file(cls, filepath: str, tokenizer=None):
        gen = cls(tokenizer=tokenizer)
        gen.load_state_dict(torch.load(filepath))
        return gen

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

