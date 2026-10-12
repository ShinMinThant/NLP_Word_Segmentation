from pathlib import Path


GUM_FILE = Path(
    r"./GUM_temp/UD_English-GUM-master/en_gum-ud-train.conllu"
)


def parse_conllu(path):
    sentences = []
    current_text = None
    current_tokens = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("# text = "):
                current_text = line[len("# text = "):]

            elif line == "":
                if current_text and current_tokens:
                    sentences.append({
                        "text": current_text,
                        "tokens": current_tokens,
                    })

                current_text = None
                current_tokens = []

            elif line[0].isdigit():
                parts = line.split("\t")

                if len(parts) >= 10:
                    token_id = parts[0]

                    # Ignore multi-word token ranges such as 2-3
                    if "-" not in token_id and "." not in token_id:
                        token = parts[1]
                        current_tokens.append(token)

    return sentences


if __name__ == "__main__":
    sentences = parse_conllu(GUM_FILE)

    print("Total sentences:", len(sentences))

    for i, sent in enumerate(sentences[:10], start=1):
        print(f"\nSentence {i}")
        print("Text  :", sent["text"])
        print("Tokens:", " | ".join(sent["tokens"]))