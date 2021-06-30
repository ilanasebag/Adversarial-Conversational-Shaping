import torch

from src.model import Model
from src.utils import concat_dialogues


class Discriminator(Model):
    def __init__(self, model=None, tokenizer=None) -> None:
        super().__init__(model, tokenizer)
        self._label_real = self.tokenizer("real", return_tensors="pt").input_ids
        self._label_fake = self.tokenizer("fake", return_tensors="pt").input_ids

        self.register_buffer("label_real", self._label_real)
        self.register_buffer("label_fake", self._label_fake)

    def get_loss(self, context, real_reply, fake_reply):

        output_real = self.model(
            input_ids=concat_dialogues(context, real_reply, self.tokenizer),
            labels=self.label_real,
        )

        output_fake = self.model(
            input_ids=concat_dialogues(context, fake_reply, self.tokenizer),
            labels=self.label_fake,
        )

        reward_real = torch.softmax(output_real.logits[0, 0, [490, 9901]], dim=0)[
            0
        ].item()

        reward_fake = torch.softmax(output_fake.logits[0, 0, [490, 9901]], dim=0)[
            0
        ].item()

        loss = output_real.loss + output_fake.loss
        return loss, reward_real, reward_fake

    def get_reward(self, context, reply):

        with torch.no_grad():
            output = self.model(
                input_ids=concat_dialogues(context, reply, self.tokenizer),
                labels=self.label_real,
            )
        reward = torch.softmax(output.logits[0, 0, [490, 9901]], dim=0)[0].item()

        return reward

