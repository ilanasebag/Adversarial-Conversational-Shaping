import argparse
from os.path import join as path_join

import numpy as np
import torch
from transformers import AdamW
from tqdm import tqdm

from src.generator import Generator
from src.dataset import DailyDialogueDataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-path", type=str, required=True,
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--output-path", type=str, default="generator_pretrained.pt")
    parser.add_argument("--freeze", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device("cuda")

    generator = Generator().to(device)
    if args.freeze:
        for name, param in generator.named_parameters():
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

    optimizer = AdamW(generator.parameters(), lr=args.lr)

    best_loss = np.float("inf")

    for epoch in tqdm(range(args.num_epochs)):
        train_loss, valid_loss = [], []
        generator.train()
        for ind in np.random.permutation(len(train_dataset)):
            optimizer.zero_grad()
            context, reply = train_dataset.sample_dialouge(ind)
            context, reply = context.to(device), reply.to(device)
            loss = generator.get_loss(input_ids=context, labels=reply)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        generator.eval()
        for ind in range(len(valid_dataset)):
            context, reply = valid_dataset[ind]
            context, reply = context.to(device), reply.to(device)
            with torch.no_grad():
                loss = generator.get_loss(input_ids=context, labels=reply)
            valid_loss.append(loss.item())

        train_loss, valid_loss = np.mean(train_loss), np.mean(valid_loss)
        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.2f}, Valid Loss: {valid_loss:.2f}"
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(generator.state_dict(), args.output_path)


if __name__ == "__main__":
    main()
