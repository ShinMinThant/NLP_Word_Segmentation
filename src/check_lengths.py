import json


def check_file(path):
    lengths = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                lengths.append(len(sample["input"]))

    print(f"\nFile: {path}")
    print(f"Samples: {len(lengths)}")
    print(f"Shortest: {min(lengths)}")
    print(f"Longest: {max(lengths)}")

    over_256 = [x for x in lengths if x > 256]
    over_512 = [x for x in lengths if x > 512]

    print(f"> 256 characters: {len(over_256)}")
    print(f"> 512 characters: {len(over_512)}")


print("=" * 50)
print("Dataset Sequence Length Check")
print("=" * 50)

check_file("./data/train.jsonl")
check_file("./data/valid.jsonl")
check_file("./data/test.jsonl")
