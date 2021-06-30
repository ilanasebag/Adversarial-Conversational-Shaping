import torch

from src.model import Model
from src.constants import GEENRATOR_MAX_LENGTH
from src.utils import concat_dialogues


class Generator(Model):
    def __init__(self, model=None, tokenizer=None) -> None:
        super().__init__(model=model, tokenizer=tokenizer)
        self._dialogue_context = None

    def get_loss(self, *args, **kwargs):
        return self.model(*args, **kwargs).loss

    def generate(self, context, do_sample=False):
        return self.model.generate(
            context, do_sample=do_sample, max_length=GEENRATOR_MAX_LENGTH
        )[:, 1:]

    def get_logprob(self, context, reply):
        return torch.log_softmax(
            self.model(input_ids=context, labels=reply).logits, dim=2
        )[:, range(reply.size(1)), reply[0]]

    def get_prob(self, context, reply):
        return torch.softmax(self.model(input_ids=context, labels=reply).logits, dim=2)[
            :, range(reply.size(1)), reply[0]
        ]

    @property
    def device(self):
        return next(self.parameters()).device

    def reset(self):
        self._dialogue_context = None

    def reply(self, utt, do_sample=False):
        tokens = self.tokenizer(utt, return_tensors="pt").input_ids
        self._dialogue_context = (
            tokens
            if self._dialogue_context is None
            else concat_dialogues(self._dialogue_context, tokens, self.tokenizer)
        )
        reply = self.generate(self._dialogue_context)
        self._dialogue_context = concat_dialogues(
            self._dialogue_context, reply, self.tokenizer
        )
        return self.tokenizer.decode(reply.squeeze())

