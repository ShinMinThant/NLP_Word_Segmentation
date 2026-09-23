import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class WordSegmentationDataset(Dataset):
    def __init__(self, jsonl_path, vocab_path):
        self.jsonl_path = Path(jsonl_path)
        self.vocab_path = Path(vocab_path)

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            self.samples = [
                json.loads(line)
                for line in f
                if line.strip()
            ]

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.pad_id = self.vocab["<PAD>"]
        self.unk_id = self.vocab["<UNK>"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        text = sample["input"]
        labels = sample["labels"]

        input_ids = [
            self.vocab.get(char, self.unk_id)
            for char in text
        ]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def collate_fn(batch):
    input_ids_list, labels_list = zip(*batch)

    max_length = max(len(x) for x in input_ids_list)

    padded_inputs = []
    padded_labels = []
    attention_masks = []

    for input_ids, labels in zip(input_ids_list, labels_list):
        input_padding = max_length - len(input_ids)

        # Input has one ID per character
        padded_input = torch.cat([
            input_ids,
            torch.full(
                (input_padding,),
                0,
                dtype=torch.long
            )
        ])

        # Labels represent gaps, so they have length input_length - 1.
        # We need max_length - 1 labels in the batch.
        target_label_length = max_length - 1
        label_padding = target_label_length - len(labels)

        padded_label = torch.cat([
            labels,
            torch.full(
                (label_padding,),
                -100,
                dtype=torch.long
            )
        ])

        attention_mask = torch.cat([
            torch.ones(len(input_ids), dtype=torch.long),
            torch.zeros(input_padding, dtype=torch.long)
        ])

        padded_inputs.append(padded_input)
        padded_labels.append(padded_label)
        attention_masks.append(attention_mask)

    return (
        torch.stack(padded_inputs),
        torch.stack(padded_labels),
        torch.stack(attention_masks),
    )


if __name__ == "__main__":
    dataset = WordSegmentationDataset(
        "./data/train.jsonl",
        "./data/vocab.json"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    print("Dataset loaded successfully!")
    print("Number of samples:", len(dataset))
    print("Vocabulary size:", len(dataset.vocab))

    input_ids, labels, attention_mask = next(iter(dataloader))

    print("\nFirst batch:")
    print("Input shape:", input_ids.shape)
    print("Labels shape:", labels.shape)
    print("Attention mask shape:", attention_mask.shape)

    print("\nInput IDs:")
    print(input_ids)

    print("\nLabels:")
    print(labels)

    print("\nAttention mask:")
    print(attention_mask)