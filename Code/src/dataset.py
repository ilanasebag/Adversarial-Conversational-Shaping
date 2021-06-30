import random
import re

import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.constants import USR_END_TKN


class DailyDialogueDataset(Dataset):
    def __init__(
        self, filepath: str, tokenizer: PreTrainedTokenizer, debug: bool = False
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self._dialogues = []

        with open(filepath, "r", encoding='utf8') as fp:
            dialogues = fp.readlines()

        if debug:
            dialogues = dialogues[:10]

        for lines in dialogues:
            lines = lines.rstrip("\n").rstrip(USR_END_TKN)
            lines = re.sub(r'\s([?.!,"](?:\s|$))', r"\1", lines)
            lines = lines.split(USR_END_TKN)
            splits = []
            for split_point in range(1, len(lines)):
                context_str, reply_str = (
                    USR_END_TKN.join(lines[:split_point]),
                    lines[split_point],
                )
                context = self.tokenizer(
                    context_str, return_tensors="pt", max_length=512
                ).input_ids
                reply = tokenizer(
                    reply_str, return_tensors="pt", max_length=512
                ).input_ids
                splits.append((context, reply))
            self._dialogues.append(splits)

    def __len__(self) -> int:
        return len(self._dialogues)

    def __getitem__(self, index):
        return self._dialogues[index][-1]

    def sample(self):
        splits = self._dialogues[np.random.choice(len(self))]
        return random.choice(splits)

    def sample_dialouge(self, ind):
        return random.choice(self._dialogues[ind])

