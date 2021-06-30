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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-path", type=str, required=True,
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--generator-path", type=str, default="generator_pretrained.pt")
    parser.add_argument(
        "--output-path", type=str, default="discriminator_pretrained.pt"
    )
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--freeze", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device("cuda")

    generator = Generator.from_file(args.generator_path).to(device)
    if args.freeze:
        for name, param in generator.named_parameters():
            if ("shared" not in name) and ("decoder.block.5" not in name):
                param.requires_grad = False
    generator.eval()

    discriminator = Discriminator(tokenizer=generator.tokenizer).to(device)
    if args.freeze:
        for name, param in discriminator.named_parameters():
            if ("shared" not in name) and ("decoder.block.5" not in name):
                param.requires_grad = False

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

    best_loss = np.float("inf")

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

            if args.partial:
                split_real = random.randint(1, real_reply.size(1))
                real_reply = real_reply[:, :split_real]
                split_fake = random.randint(1, fake_reply.size(1) - 1)
                fake_reply = fake_reply[:, :split_fake]

            loss, _, _ = discriminator.get_loss(context, real_reply, fake_reply)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        discriminator.eval()
        for ind in range(len(valid_dataset)):
            context, real_reply = valid_dataset[ind]
            context, real_reply = (
                context.to(device),
                real_reply.to(device),
            )
            fake_reply = generator.generate(context, do_sample=True)

            if args.partial:
                split_real = random.randint(1, real_reply.size(1))
                real_reply = real_reply[:, :split_real]
                split_fake = random.randint(1, fake_reply.size(1) - 1)
                fake_reply = fake_reply[:, :split_fake]

            with torch.no_grad():
                loss, reward_real, reward_fake = discriminator.get_loss(
                    context, real_reply, fake_reply
                )
            valid_loss.append(loss.item())
            rewards_real.append(reward_real)
            rewards_fake.append(reward_fake)
            accuracy.extend([reward_real > 0.5, reward_fake < 0.5])

        train_loss, valid_loss = np.mean(train_loss), np.mean(valid_loss)
        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.2f}, Valid Loss: {valid_loss:.2f}, Reward real: {np.mean(rewards_real):.2f}, Reward fake: {np.mean(rewards_fake):.2f}, Accuracy: {np.mean(accuracy):.2f}"
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(discriminator.state_dict(), args.output_path)


if __name__ == "__main__":
    main()
