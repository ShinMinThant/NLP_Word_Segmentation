import xml.etree.ElementTree as ET
from pathlib import Path

base = Path(
    r".\MASC\MASC-3.0.0\data\written\blog\Acephalous-Cant-believe"
)

text = base.with_suffix(".txt").read_text(encoding="utf-8")

# -----------------------------
# 1. Read sentence annotation
# -----------------------------
sentence_root = ET.parse(str(base) + "-s.xml").getroot()

sentence_regions = [
    region
    for region in sentence_root
    if region.tag.endswith("region")
]

sentence_start, sentence_end = map(
    int,
    sentence_regions[0].get("anchors").split()
)

sentence_text = text[sentence_start:sentence_end]

# -----------------------------
# 2. Read token annotations
# -----------------------------
seg_root = ET.parse(str(base) + "-seg.xml").getroot()
penn_root = ET.parse(str(base) + "-penn.xml").getroot()

# Region ID -> character span
spans = {}

for region in seg_root:
    if region.tag.endswith("region"):
        region_id = region.get(
            "{http://www.w3.org/XML/1998/namespace}id"
        )

        start, end = map(
            int,
            region.get("anchors").split()
        )

        spans[region_id] = (start, end)

# Penn node ID -> region ID
node_to_region = {}

for node in penn_root:
    if node.tag.endswith("node"):
        node_id = node.get(
            "{http://www.w3.org/XML/1998/namespace}id"
        )

        link = node.find(".//{*}link")

        if link is not None:
            node_to_region[node_id] = link.get("targets")

# Get tokens
tokens = []

for annotation in penn_root.iter():
    if annotation.tag.endswith("a") and annotation.get("label") == "tok":

        token_element = annotation.find(
            './/{*}f[@name="string"]'
        )

        if token_element is None:
            continue

        token_text = token_element.get("value")
        node_id = annotation.get("ref")
        region_id = node_to_region.get(node_id)

        if region_id in spans:
            start, end = spans[region_id]
            tokens.append((token_text, start, end))

tokens.sort(key=lambda x: x[1])

# -----------------------------
# 3. Get tokens in Sentence 1
# -----------------------------
sentence_tokens = [
    token
    for token in tokens
    if token[1] >= sentence_start
    and token[2] <= sentence_end
]

# -----------------------------
# 4. Build character sequence
# -----------------------------
characters = list(sentence_text)

# Boundary after each character
# 1 = token boundary
# 0 = no token boundary
boundaries = []

for i in range(len(characters) - 1):

    absolute_position = sentence_start + i + 1

    boundary = 0

    for token_index in range(len(sentence_tokens) - 1):
        current_end = sentence_tokens[token_index][2]

        if absolute_position == current_end:
            boundary = 1
            break

    boundaries.append(boundary)

# -----------------------------
# 5. Display result
# -----------------------------
print("SENTENCE:")
print(sentence_text)

print()
print("INPUT WITHOUT SPACES:")
print(sentence_text.replace(" ", ""))

print()
print("CHARACTER + BOUNDARY:")
print()

for i, char in enumerate(characters[:-1]):
    print(f"{char!r:5} -> {boundaries[i]}")

print()
print("CHARACTERS:", len(characters))
print("GAPS:", len(boundaries))
print("BOUNDARIES:", sum(boundaries))