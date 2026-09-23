import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path


# =========================
# Configuration
# =========================

DATA_DIR = Path(
    r"C:\Users\HP\OneDrive\Desktop\NLP_Word_Segmentation\MASC\MASC-3.0.0\data\written"
)

OUTPUT_DIR = Path("./data")

RANDOM_SEED = 42

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

# Maximum number of characters in one example.
MAX_LENGTH = 256


# =========================
# Read Token Annotations
# =========================

def read_tokens(penn_path, seg_path):

    seg_tree = ET.parse(seg_path)
    seg_root = seg_tree.getroot()

    penn_tree = ET.parse(penn_path)
    penn_root = penn_tree.getroot()

    # MASC XML namespace
    ns = {
        "graf": "http://www.xces.org/ns/GrAF/1.0/"
    }

    # region_id -> (start, end)
    regions = {}

    for region in seg_root.findall(
        ".//graf:region",
        ns
    ):
        region_id = region.get(
            "{http://www.w3.org/XML/1998/namespace}id"
        )

        anchors = region.get("anchors")

        if region_id and anchors:
            start, end = map(
                int,
                anchors.split()
            )

            regions[region_id] = (
                start,
                end
            )

    # node_id -> region_id
    node_regions = {}

    for node in penn_root.findall(
        ".//graf:node",
        ns
    ):
        node_id = node.get(
            "{http://www.w3.org/XML/1998/namespace}id"
        )

        link = node.find(
            "graf:link",
            ns
        )

        if node_id and link is not None:
            target = link.get("targets")

            if target:
                node_regions[node_id] = target

    tokens = []

    for annotation in penn_root.findall(
        ".//graf:a",
        ns
    ):

        if annotation.get("label") != "tok":
            continue

        node_id = annotation.get("ref")

        token_string = None

        for feature in annotation.findall(
            ".//graf:f",
            ns
        ):
            if feature.get("name") == "string":
                token_string = feature.get("value")
                break

        if (
            node_id
            and token_string
            and node_id in node_regions
        ):
            region_id = node_regions[node_id]

            if region_id in regions:
                start, end = regions[region_id]

                tokens.append(
                    (
                        token_string,
                        start,
                        end
                    )
                )

    tokens.sort(
        key=lambda x: x[1]
    )

    return tokens

    # -------------------------
    # Read segment spans
    # -------------------------

    seg_tree = ET.parse(seg_path)
    seg_root = seg_tree.getroot()

    regions = {}

    for region in seg_root.iter("region"):

        region_id = region.attrib.get("{http://www.w3.org/XML/1998/namespace}id")

        if region_id is None:
            region_id = region.attrib.get("xml:id")

        anchors = region.attrib.get("anchors")

        if not region_id or not anchors:
            continue

        start, end = map(int, anchors.split())

        regions[region_id] = (start, end)

    # -------------------------
    # Read Penn annotations
    # -------------------------

    penn_tree = ET.parse(penn_path)
    penn_root = penn_tree.getroot()

    node_to_region = {}

    for node in penn_root.iter("node"):

        node_id = node.attrib.get(
            "{http://www.w3.org/XML/1998/namespace}id"
        )

        if node_id is None:
            node_id = node.attrib.get("xml:id")

        if not node_id:
            continue

        for feature in node.findall("f"):

            name = feature.attrib.get("name")
            value = feature.attrib.get("value")

            if name == "region":
                node_to_region[node_id] = value

    # -------------------------
    # Read token annotations
    # -------------------------

    tokens = []

    for annotation in penn_root.iter("a"):

        if annotation.attrib.get("label") != "tok":
            continue

        ref = annotation.attrib.get("ref")

        if ref not in node_to_region:
            continue

        region_id = node_to_region[ref]

        if region_id not in regions:
            continue

        token_text = ""

        for feature in annotation.findall("f"):

            if feature.attrib.get("name") == "string":
                token_text = feature.attrib.get("value", "")
                break

        if not token_text:
            continue

        start, end = regions[region_id]

        tokens.append(
            (
                token_text,
                start,
                end
            )
        )

    # Sort tokens by character position
    tokens.sort(key=lambda x: x[1])

    return tokens


# =========================
# Read Sentence Spans
# =========================

def read_sentences(sentence_path):

    tree = ET.parse(sentence_path)
    root = tree.getroot()

    ns = {
        "graf": "http://www.xces.org/ns/GrAF/1.0/"
    }

    regions = {}

    for region in root.findall(
        ".//graf:region",
        ns
    ):
        region_id = region.get(
            "{http://www.w3.org/XML/1998/namespace}id"
        )

        anchors = region.get("anchors")

        if region_id and anchors:
            start, end = map(
                int,
                anchors.split()
            )

            regions[region_id] = (
                start,
                end
            )

    sentences = []

    for node in root.findall(
        ".//graf:node",
        ns
    ):
        link = node.find(
            "graf:link",
            ns
        )

        if link is None:
            continue

        target = link.get("targets")

        if target in regions:
            sentences.append(
                regions[target]
            )

    sentences.sort(
        key=lambda x: x[0]
    )

    return sentences

    tree = ET.parse(sentence_path)
    root = tree.getroot()

    sentences = []

    for region in root.iter("region"):

        region_id = region.attrib.get(
            "{http://www.w3.org/XML/1998/namespace}id"
        )

        if region_id is None:
            region_id = region.attrib.get("xml:id")

        if not region_id:
            continue

        # Sentence regions normally start with "s-"
        if not region_id.startswith("s-"):
            continue

        anchors = region.attrib.get("anchors")

        if not anchors:
            continue

        start, end = map(int, anchors.split())

        sentences.append(
            (
                start,
                end
            )
        )

    sentences.sort(key=lambda x: x[0])

    return sentences


# =========================
# Create One Example
# =========================

def create_example(
    text,
    sentence_start,
    sentence_end,
    tokens
):

    sentence_tokens = []

    for token_text, start, end in tokens:

        if (
            start >= sentence_start
            and end <= sentence_end
        ):
            sentence_tokens.append(
                (
                    token_text,
                    start,
                    end
                )
            )

    if not sentence_tokens:
        return None

    # Build whitespace-free input
    input_text = "".join(
        token[0]
        for token in sentence_tokens
    )

    if len(input_text) < 2:
        return None

    # -------------------------
    # Create gap labels
    # -------------------------

    labels = [0] * (len(input_text) - 1)

    current_position = 0

    for token_index, (token_text, _, _) in enumerate(
        sentence_tokens
    ):

        current_position += len(token_text)

        # Boundary between this token and next token
        if (
            token_index < len(sentence_tokens) - 1
            and current_position > 0
            and current_position < len(input_text)
        ):
            labels[current_position - 1] = 1

    return {
        "input": input_text,
        "labels": labels,
        "tokens": [
            token[0]
            for token in sentence_tokens
        ],
        "source_span": [
            sentence_start,
            sentence_end
        ]
    }


# =========================
# Safe Chunking
# =========================

def chunk_example(example):

    text = example["input"]
    labels = example["labels"]

    if len(text) <= MAX_LENGTH:
        return [example]

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        # Last chunk
        if start + MAX_LENGTH >= text_length:

            chunk_end = text_length

        else:

            # Initial maximum end
            candidate_end = start + MAX_LENGTH

            # Find the nearest valid word boundary
            # before the maximum length.
            valid_boundaries = [
                i
                for i in range(
                    start + 1,
                    candidate_end + 1
                )
                if i < len(labels)
                and labels[i - 1] == 1
            ]

            if valid_boundaries:

                chunk_end = max(valid_boundaries)

            else:

                # If there is no word boundary before
                # MAX_LENGTH, use the maximum length.
                #
                # This is extremely rare. The original
                # boundary label at the chunk edge is kept
                # rather than creating a new boundary.
                chunk_end = candidate_end

        chunk_text = text[start:chunk_end]

        # Gaps inside the chunk
        chunk_labels = labels[
            start:chunk_end - 1
        ]

        if len(chunk_text) >= 2:

            chunks.append(
                {
                    "input": chunk_text,
                    "labels": chunk_labels,
                    "tokens": [],
                    "source_span": example["source_span"]
                }
            )

        # Move to the next chunk
        start = chunk_end

    return chunks


# =========================
# Collect Documents
# =========================

def collect_documents():

    txt_files = sorted(
        DATA_DIR.rglob("*.txt")
    )

    documents = []

    for txt_path in txt_files:

        base = txt_path.stem

        penn_path = txt_path.parent / f"{base}-penn.xml"
        seg_path = txt_path.parent / f"{base}-seg.xml"
        sentence_path = txt_path.parent / f"{base}-s.xml"

        if not (
            penn_path.exists()
            and seg_path.exists()
            and sentence_path.exists()
        ):
            continue

        documents.append(
            {
                "txt": txt_path,
                "penn": penn_path,
                "seg": seg_path,
                "sentence": sentence_path
            }
        )

    return documents


# =========================
# Process Documents
# =========================

def process_document(document):

    with open(
        document["txt"],
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    tokens = read_tokens(
        document["penn"],
        document["seg"]
    )

    sentences = read_sentences(
        document["sentence"]
    )

    examples = []

    for sentence_start, sentence_end in sentences:

        example = create_example(
            text,
            sentence_start,
            sentence_end,
            tokens
        )

        if example is None:
            continue

        examples.extend(
            chunk_example(example)
        )

    return examples


# =========================
# Write JSONL
# =========================

def write_jsonl(path, examples):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for example in examples:

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                )
                + "\n"
            )


# =========================
# Main
# =========================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    documents = collect_documents()

    print("Found TXT files:", len(
        list(DATA_DIR.glob("*.txt"))
    ))

    print(
        "Documents with annotations:",
        len(documents)
    )

    # -------------------------
    # Document-level split
    # -------------------------

    random.seed(RANDOM_SEED)

    random.shuffle(documents)

    total_documents = len(documents)

    train_count = int(
        total_documents * TRAIN_RATIO
    )

    valid_count = int(
        total_documents * VALID_RATIO
    )

    train_documents = documents[
        :train_count
    ]

    valid_documents = documents[
        train_count:
        train_count + valid_count
    ]

    test_documents = documents[
        train_count + valid_count:
    ]

    print("\nDOCUMENT SPLIT")
    print(
        "Train:",
        len(train_documents)
    )

    print(
        "Validation:",
        len(valid_documents)
    )

    print(
        "Test:",
        len(test_documents)
    )

    # -------------------------
    # Create examples
    # -------------------------

    def process_split(split_documents):

        all_examples = []

        for document in split_documents:

            examples = process_document(
                document
            )

            all_examples.extend(
                examples
            )

        return all_examples

    train_examples = process_split(
        train_documents
    )

    valid_examples = process_split(
        valid_documents
    )

    test_examples = process_split(
        test_documents
    )

    # -------------------------
    # Save
    # -------------------------

    write_jsonl(
        OUTPUT_DIR / "train.jsonl",
        train_examples
    )

    write_jsonl(
        OUTPUT_DIR / "valid.jsonl",
        valid_examples
    )

    write_jsonl(
        OUTPUT_DIR / "test.jsonl",
        test_examples
    )

    print("\nDATASET CREATED")
    print("=" * 50)

    print(
        "train:",
        len(train_examples),
        "examples"
    )

    print(
        "valid:",
        len(valid_examples),
        "examples"
    )

    print(
        "test:",
        len(test_examples),
        "examples"
    )

    print(
        "\nMaximum allowed length:",
        MAX_LENGTH
    )

    # Verify maximum lengths
    print(
        "Train max length:",
        max(
            len(x["input"])
            for x in train_examples
        )
    )

    print(
        "Valid max length:",
        max(
            len(x["input"])
            for x in valid_examples
        )
    )

    print(
        "Test max length:",
        max(
            len(x["input"])
            for x in test_examples
        )
    )

    print(
        "\nOutput directory:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
