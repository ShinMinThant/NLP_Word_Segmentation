import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataset import WordSegmentationDataset, collate_fn
from models.transformer import TransformerWordSegmentation


TRAIN_PATH = "./data/train.jsonl"
VALID_PATH = "./data/valid.jsonl"
VOCAB_PATH = "./data/vocab.json"
MODEL_PATH = "./models/transformer_masc_v2_best.pt"

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for input_ids, labels, attention_mask in dataloader:
            input_ids = input_ids.to(DEVICE)
            labels = labels.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            logits = model(input_ids, attention_mask)

            loss = criterion(
                logits.reshape(-1, 2),
                labels.reshape(-1)
            )
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)
            valid_mask = labels != -100

            all_predictions.extend(
                predictions[valid_mask].cpu().tolist()
            )
            all_labels.extend(
                labels[valid_mask].cpu().tolist()
            )

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="binary",
        zero_division=0
    )

    return total_loss / len(dataloader), precision, recall, f1


def main():
    print("=" * 60)
    print("MASC Transformer Word Segmentation Training")
    print("=" * 60)
    print("Device:", DEVICE)

    train_dataset = WordSegmentationDataset(TRAIN_PATH, VOCAB_PATH)
    valid_dataset = WordSegmentationDataset(VALID_PATH, VOCAB_PATH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab_size = len(vocab)

    model = TransformerWordSegmentation(
        vocab_size=vocab_size,
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        feedforward_dim=256,
        dropout=0.1,
        max_length=256
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = -1.0

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(valid_dataset))
    print("Vocabulary size:", vocab_size)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0

        for batch_index, (input_ids, labels, attention_mask) in enumerate(
            train_loader, start=1
        ):
            input_ids = input_ids.to(DEVICE)
            labels = labels.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits.reshape(-1, 2), labels.reshape(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if batch_index % 100 == 0:
                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Batch {batch_index}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss = total_train_loss / len(train_loader)
        valid_loss, precision, recall, f1 = evaluate(
            model, valid_loader, criterion
        )

        print("\n" + "-" * 60)
        print(f"Epoch {epoch}/{EPOCHS} completed")
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Valid Loss : {valid_loss:.4f}")
        print(f"Precision  : {precision:.4f}")
        print(f"Recall     : {recall:.4f}")
        print(f"F1-score   : {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": vocab_size,
                    "embedding_dim": 128,
                    "num_heads": 4,
                    "num_layers": 2,
                    "feedforward_dim": 256,
                    "dropout": 0.1,
                    "max_length": 256,
                    "best_validation_f1": best_f1
                },
                MODEL_PATH
            )
            print("Best model saved!")
            print(f"Best Validation F1: {best_f1:.4f}")

        print("-" * 60)

    print("\nTraining finished!")
    print(f"Best Validation F1: {best_f1:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
