import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataset import WordSegmentationDataset, collate_fn
from models.bilstm import BiLSTMWordSegmentation


TEST_PATH = "./data/test.jsonl"
VOCAB_PATH = "./data/vocab.json"
MODEL_PATH = "./models/bilstm_masc_v2_best.pt"
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("=" * 60)
    print("MASC BiLSTM Test Evaluation")
    print("=" * 60)
    print("Device:", DEVICE)

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    test_dataset = WordSegmentationDataset(TEST_PATH, VOCAB_PATH)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    checkpoint_path = Path(MODEL_PATH)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {MODEL_PATH}")
        print("Train the MASC BiLSTM model first.")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model = BiLSTMWordSegmentation(
        vocab_size=checkpoint.get("vocab_size", len(vocab)),
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint.get("dropout", 0.2)
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_predictions = []
    all_labels = []

    print("Test samples:", len(test_dataset))
    print("Running evaluation...")

    with torch.no_grad():
        for batch_index, (input_ids, labels, attention_mask) in enumerate(
            test_loader, start=1
        ):
            input_ids = input_ids.to(DEVICE)
            labels = labels.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            lengths = attention_mask.sum(dim=1)
            logits = model(input_ids, lengths)
            predictions = torch.argmax(logits, dim=-1)

            valid_mask = labels != -100
            all_predictions.extend(predictions[valid_mask].cpu().tolist())
            all_labels.extend(labels[valid_mask].cpu().tolist())

            if batch_index % 50 == 0:
                print(f"Processed batches: {batch_index}/{len(test_loader)}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="binary",
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("MASC BiLSTM TEST RESULTS")
    print("=" * 60)
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall   : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-score : {f1:.4f} ({f1 * 100:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
