import json
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.transformer import TransformerWordSegmentation


VOCAB_PATH = "./data/vocab.json"
MODEL_PATH = "./models/transformer_best.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model = TransformerWordSegmentation(
        vocab_size=len(vocab),
        embedding_dim=checkpoint["embedding_dim"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        feedforward_dim=checkpoint["feedforward_dim"],
        max_length=checkpoint["max_length"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    return model, vocab


def segment_text(text, model, vocab):
    if len(text) < 2:
        return text

    input_ids = [
        vocab.get(char, vocab["<UNK>"])
        for char in text
    ]

    input_tensor = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=DEVICE
    )

    attention_mask = torch.ones(
        input_tensor.shape,
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():
        logits = model(
            input_tensor,
            attention_mask
        )

        predictions = torch.argmax(
            logits,
            dim=-1
        )[0].cpu().tolist()

    words = []
    current_word = text[0]

    for i, boundary in enumerate(predictions):
        next_char = text[i + 1]

        if boundary == 1:
            words.append(current_word)
            current_word = next_char
        else:
            current_word += next_char

    words.append(current_word)

    return " | ".join(words)


def main():
    print("=" * 70)
    print("Transformer Word Segmentation Prediction")
    print("=" * 70)

    print("Device:", DEVICE)

    model, vocab = load_model()

    examples = [
        "Ican'tbelieveIwroteallthatlastyear.",
        "Thequickbrownfoxjumpsoverthelazydog.",
        "Machinelearningisinteresting."
    ]

    print("\nPredicted Segmentation:\n")

    for text in examples:
        result = segment_text(
            text,
            model,
            vocab
        )

        print("Input     :", text)
        print("Prediction:", result)
        print("-" * 70)


if __name__ == "__main__":
    main()