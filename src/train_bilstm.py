import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataset import WordSegmentationDataset, collate_fn
from models.bilstm import BiLSTMWordSegmentation


TRAIN_PATH = "./data/train.jsonl"
VALID_PATH = "./data/valid.jsonl"
VOCAB_PATH = "./data/vocab.json"
MODEL_PATH = "./models/bilstm_masc_v2_best.pt"

BATCH_SIZE = 16
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

            lengths = attention_mask.sum(dim=1)
            logits = model(input_ids, lengths)

            loss = criterion(logits.reshape(-1, 2), labels.reshape(-1))
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)
            valid_mask = labels != -100

            all_predictions.extend(predictions[valid_mask].cpu().tolist())
            all_labels.extend(labels[valid_mask].cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="binary",
        zero_division=0
    )

    return total_loss / max(len(dataloader), 1), precision, recall, f1


def main():
    print("=" * 60)
    print("MASC BiLSTM Word Segmentation Training")
    print("=" * 60)
    print("Device:", DEVICE)

    train_dataset = WordSegmentationDataset(TRAIN_PATH, VOCAB_PATH)
    valid_dataset = WordSegmentationDataset(VALID_PATH, VOCAB_PATH)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn
    )

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab_size = len(vocab)

    model = BiLSTMWordSegmentation(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = -1.0

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(valid_dataset))
    print("Vocabulary size:", vocab_size)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Checkpoint:", MODEL_PATH)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0

        for batch_index, (input_ids, labels, attention_mask) in enumerate(
            train_loader, start=1
        ):
            input_ids = input_ids.to(DEVICE)
            labels = labels.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            lengths = attention_mask.sum(dim=1)

            optimizer.zero_grad()
            logits = model(input_ids, lengths)
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

        train_loss = total_train_loss / max(len(train_loader), 1)
        valid_loss, precision, recall, f1 = evaluate(
            model, valid_loader, criterion
        )

        print("-" * 60)
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": vocab_size,
                    "embedding_dim": 128,
                    "hidden_dim": 128,
                    "num_layers": 2,
                    "dropout": 0.2,
                    "best_validation_f1": best_f1
                },
                MODEL_PATH
            )
            print("Best MASC BiLSTM checkpoint saved.")

    print("=" * 60)
    print("Training finished.")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
