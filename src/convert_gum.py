from pathlib import Path
import json


GUM_DIR = Path(r"./GUM_temp/UD_English-GUM-master")
OUTPUT_DIR = Path(r"./data/gum")

MAX_LENGTH = 256


def parse_conllu(path):
    sentences = []

    current_text = None
    current_tokens = []

    with path.open("r", encoding="utf-8") as f:

        for line in f:
            line = line.rstrip("\n")

            if line.startswith("# text = "):

                current_text = line[len("# text = "):]

            elif line and line[0].isdigit():

                parts = line.split("\t")

                if len(parts) < 10:
                    continue

                token_id = parts[0]
                form = parts[1]

                # Ignore MWT range rows such as 1-2.
                if "-" in token_id:
                    continue

                # Ignore empty nodes such as 3.1.
                if "." in token_id:
                    continue

                current_tokens.append(form)

            elif line == "":

                if current_text and current_tokens:

                    sentences.append({
                        "text": current_text,
                        "tokens": current_tokens,
                    })

                current_text = None
                current_tokens = []

    return sentences


def create_example(tokens):

    if not tokens:
        return None

    text = "".join(tokens)

    if len(text) < 2:
        return None

    labels = [0] * (len(text) - 1)

    position = 0

    for token in tokens[:-1]:

        position += len(token)

        labels[position - 1] = 1

    return {
        "input": text,
        "labels": labels,
        "tokens": tokens,
    }


def chunk_example(example):

    text = example["input"]
    labels = example["labels"]
    tokens = example["tokens"]

    if len(text) <= MAX_LENGTH:
        return [example]

    chunks = []

    current_text = ""
    current_labels = []
    current_tokens = []

    position = 0

    for token_index, token in enumerate(tokens):

        token_length = len(token)

        # If adding this token would exceed MAX_LENGTH,
        # save the current chunk first.
        if (
            current_text
            and len(current_text) + token_length > MAX_LENGTH
        ):

            chunks.append({
                "input": current_text,
                "labels": current_labels,
                "tokens": current_tokens,
            })

            current_text = ""
            current_labels = []
            current_tokens = []

        # Skip an individual token longer than MAX_LENGTH.
        if token_length > MAX_LENGTH:
            continue

        start_position = position

        current_text += token
        current_tokens.append(token)

        position += token_length

        # Add labels belonging to this token.
        #
        # Labels describe gaps between characters.
        # For the first token there is no boundary before it.
        #
        # For every following token, add the boundary
        # immediately before that token.
        if token_index > 0:

            boundary_position = len(current_text) - token_length

            local_boundary_index = boundary_position - 1

            # Current labels correspond to gaps in current_text.
            while len(current_labels) < len(current_text) - 1:
                current_labels.append(0)

            if 0 <= local_boundary_index < len(current_labels):
                current_labels[local_boundary_index] = 1

        # Make sure label length is exactly len(text) - 1.
        while len(current_labels) < len(current_text) - 1:
            current_labels.append(0)

    if current_text:

        chunks.append({
            "input": current_text,
            "labels": current_labels,
            "tokens": current_tokens,
        })

    return chunks


def convert_file(input_file, output_file):

    sentences = parse_conllu(input_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    count = 0
    chunked_count = 0

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as out:

        for sentence in sentences:

            example = create_example(
                sentence["tokens"]
            )

            if example is None:
                continue

            chunks = chunk_example(example)

            if len(chunks) > 1:
                chunked_count += 1

            for chunk in chunks:

                # Safety check.
                if len(chunk["input"]) > MAX_LENGTH:
                    raise ValueError(
                        "Chunk exceeds MAX_LENGTH: "
                        f"{len(chunk['input'])}"
                    )

                if len(chunk["labels"]) != (
                    len(chunk["input"]) - 1
                ):
                    raise ValueError(
                        "Label length mismatch: "
                        f"text={len(chunk['input'])}, "
                        f"labels={len(chunk['labels'])}"
                    )

                out.write(
                    json.dumps(
                        chunk,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                count += 1

    print(
        f"{input_file.name}: "
        f"{count} examples "
        f"({chunked_count} sentences chunked)"
    )


if __name__ == "__main__":

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    convert_file(
        GUM_DIR / "en_gum-ud-train.conllu",
        OUTPUT_DIR / "train.jsonl"
    )

    convert_file(
        GUM_DIR / "en_gum-ud-dev.conllu",
        OUTPUT_DIR / "valid.jsonl"
    )

    convert_file(
        GUM_DIR / "en_gum-ud-test.conllu",
        OUTPUT_DIR / "test.jsonl"
    )

    print("GUM conversion complete.")