import argparse
import random
from os.path import join as path_join

import numpy as np
import torch
from transformers import AdamW
from tqdm import tqdm

from src.generator import Generator
from src.discriminator import Discriminator
from src.dataset import DailyDialogueDataset
from src.utils import dist1, dist2unbiased, bleuscore


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-path", type=str, required=True,
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--generator-path", type=str, required=True)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device("cuda")

    generator = Generator.from_file(args.generator_path).to(device)
    generator.eval()
    discriminator = Discriminator(tokenizer=generator.tokenizer).to(device)

    train_dataset = DailyDialogueDataset(
        path_join(args.dataset_path, "train/dialogues_train.txt"),
        tokenizer=generator.tokenizer,
    )
    valid_dataset = DailyDialogueDataset(
        path_join(args.dataset_path, "validation/dialogues_validation.txt"),
        tokenizer=generator.tokenizer,
    )

    print(len(train_dataset), len(valid_dataset))

    optimizer = AdamW(discriminator.parameters(), lr=args.lr)

    for epoch in tqdm(range(args.num_epochs)):
        train_loss, valid_loss = [], []
        rewards_real, rewards_fake, accuracy = [], [], []
        discriminator.train()
        for ind in np.random.permutation(len(train_dataset)):
            optimizer.zero_grad()
            context, real_reply = train_dataset.sample_dialouge(ind)
            context, real_reply = (
                context.to(device),
                real_reply.to(device),
            )
            fake_reply = generator.generate(context, do_sample=True)

            loss, _, _ = discriminator.get_loss(context, real_reply, fake_reply)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        discriminator.eval()
        real_replies, fake_replies = [], []
        for ind in range(len(valid_dataset)):
            context, real_reply = valid_dataset[ind]
            context, real_reply = (
                context.to(device),
                real_reply.to(device),
            )
            fake_reply = generator.generate(context, do_sample=True)

            with torch.no_grad():
                loss, reward_real, reward_fake = discriminator.get_loss(
                    context, real_reply, fake_reply
                )
            valid_loss.append(loss.item())
            rewards_real.append(reward_real)
            rewards_fake.append(reward_fake)
            accuracy.extend([reward_real > 0.5, reward_fake < 0.5])

            real_reply, fake_reply = (
                generator.tokenizer.decode(real_reply[0]),
                generator.tokenizer.decode(fake_reply[0]),
            )
            real_replies.append(real_reply)
            fake_replies.append(fake_reply)

        train_loss, valid_loss = np.mean(train_loss), np.mean(valid_loss)
        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.2f}, Valid Loss: {valid_loss:.2f}, Reward real: {np.mean(rewards_real):.2f}, Reward fake: {np.mean(rewards_fake):.2f}, Accuracy: {np.mean(accuracy):.2f}"
        )
        print(f"Adversarial accuracy, {np.mean(accuracy):.2f}")
        for order in range(1, 5):
            print(f"BLEU-{order}: {bleuscore(real_replies, fake_replies, order=order)}")
        print(f"DIST-1: {dist1(fake_replies)}")
        print(f"DIST-2: {dist2unbiased(fake_replies)}")


if __name__ == "__main__":
    main()
