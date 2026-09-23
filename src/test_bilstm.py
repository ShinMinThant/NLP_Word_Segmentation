import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataset import WordSegmentationDataset, collate_fn
from models.bilstm import BiLSTMWordSegmentation


# =========================
# Configuration
# =========================

TEST_PATH = "./data/test.jsonl"
VOCAB_PATH = "./data/vocab.json"
MODEL_PATH = "./models/bilstm_best.pt"

BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================
# Main
# =========================

def main():

    print("=" * 60)
    print("BiLSTM Test Evaluation")
    print("=" * 60)

    print("Device:", DEVICE)

    # Load vocabulary
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab_size = len(vocab)

    # Load test dataset
    test_dataset = WordSegmentationDataset(
        TEST_PATH,
        VOCAB_PATH
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Load checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Create model
    model = BiLSTMWordSegmentation(
        vocab_size=vocab_size,
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        dropout=0.2
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    all_predictions = []
    all_labels = []

    print("\nRunning evaluation...\n")

    with torch.no_grad():

        for batch_index, (
            input_ids,
            labels,
            attention_mask
        ) in enumerate(test_loader, start=1):

            input_ids = input_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(input_ids)

            predictions = torch.argmax(
                logits,
                dim=-1
            )

            # Ignore padded label positions
            valid_mask = labels != -100

            all_predictions.extend(
                predictions[valid_mask].cpu().tolist()
            )

            all_labels.extend(
                labels[valid_mask].cpu().tolist()
            )

            if batch_index % 50 == 0:
                print(
                    f"Processed batches: "
                    f"{batch_index}/{len(test_loader)}"
                )

    # Calculate metrics
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="binary",
            zero_division=0
        )
    )

    print("\n" + "=" * 60)
    print("BiLSTM TEST RESULTS")
    print("=" * 60)

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("\nPercentage:")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1-score  : {f1 * 100:.2f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()


