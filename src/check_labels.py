from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from convert_gum import parse_conllu, build_boundaries, GUM_DIR


sentences = parse_conllu(
    GUM_DIR / "en_gum-ud-train.conllu"
)

for sentence in sentences:
    if "can't remember" in sentence["text"]:
        text = sentence["text"]
        labels = build_boundaries(sentence)

        print("TEXT:")
        print(text)

        print("\nGOLD TOKENS:")
        print(" | ".join(token["form"] for token in sentence["tokens"]))

        print("\nBOUNDARIES:")

        for i, label in enumerate(labels):
            if label == 1:
                print(
                    f"gap {i}: "
                    f"'{text[i]}' | '{text[i + 1]}'"
                )

        break