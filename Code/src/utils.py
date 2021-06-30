import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu

from src.constants import USR_END_TKN


def concat_dialogues(part_a, part_b, tokenizer):

    join = (
        tokenizer(USR_END_TKN, return_tensors="pt").input_ids[:, :1].to(part_a.device)
    )
    return torch.cat([part_a, join, part_b], dim=1)


def print_dialogue(context, real_reply, tokenizer, fake_reply=None):
    context, real_reply = tokenizer.decode(context[0]), tokenizer.decode(real_reply[0])

    for i, line in enumerate(context.split(USR_END_TKN)):
        prefix = "PERSON A" if i % 2 == 0 else "PERSON B"
        print(f"{prefix}: {line}")
    print(f"REAL REPLY: {real_reply}")

    if fake_reply is not None:
        fake_reply = tokenizer.decode(fake_reply[0])
        print(f"FAKE REPLY: {fake_reply}")


def dist1(sentences):
    """ Returns distinct-1 scores given sentences

    The parameter sentences should a list of all the sentences generared by the model.
    The score is the number of distinct unigrams/number of tokens
    This is only a relative score comparing different models generating responses to the same inputs.
    """
    unique_tokens = set()
    num_tokens = 0
    for s in sentences:
        for token in s.split(" "):
            unique_tokens.add(token)
            num_tokens = num_tokens + 1
    return len(unique_tokens) / num_tokens


def dist2(sentences):
    """ Returns distinct-2 scores given sentences

    The parameter sentences should be all the sentences generared by the model.
    The score is the number of distinct bigrams/number of tokens
    This is only a relative score comparing different models generating responses to the same inputs.
    """
    unique_bigrams = set()
    total_words = 0
    for s in sentences:
        tokens = s.split(" ")
        total_words = total_words + len(tokens)
        for i in range(len(tokens) - 1):
            bigram = tokens[i] + " " + tokens[i + 1]
            unique_bigrams.add(bigram)
    return len(unique_bigrams) / total_words


def dist2unbiased(sentences):
    """ Like distinct 2 but with a modification to account for sentence length bias.

    We don't count the total number of tokens but instead the total number of possible bigrams so
    that unique responses are not penalised just for being part of shorter sentences and so that a
    score of 1 is theoretically attainable.
    """
    unique_bigrams = set()
    possible_bigrams = 0
    for s in sentences:
        tokens = s.split(" ")
        possible_bigrams = possible_bigrams + len(tokens) - 1
        for i in range(len(tokens) - 1):
            bigram = tokens[i] + " " + tokens[i + 1]
            unique_bigrams.add(bigram)
    return len(unique_bigrams) / possible_bigrams


def bleuscore(modelsentences, targetsentences, order=4):
    assert len(modelsentences) == len(targetsentences)
    l = len(modelsentences)
    score = 0
    for i in range(l):
        score = score + sentence_bleu(
            [modelsentences[i].split()],
            targetsentences[i].split(),
            weights=np.ones(order) / order,
        )
    score = score / l * 1000
    return score
