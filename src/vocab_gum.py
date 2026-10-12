import json
from pathlib import Path


TRAIN_PATH = Path("./data/gum/train.jsonl")
VOCAB_PATH = Path("./data/gum/vocab.json")


def build_vocab():
    characters = set()

    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            characters.update(sample["input"])

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for char in sorted(characters):
        if char not in vocab:
            vocab[char] = len(vocab)

    return vocab


def main():
    vocab = build_vocab()

    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print("GUM vocabulary created successfully!")
    print("Vocabulary size:", len(vocab))
    print("Saved to:", VOCAB_PATH)


if __name__ == "__main__":
    main()