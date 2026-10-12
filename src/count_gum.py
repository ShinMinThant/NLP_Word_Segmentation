from pathlib import Path


GUM_DIR = Path(r"./GUM_temp/UD_English-GUM-master")


for file in sorted(GUM_DIR.glob("en_gum-ud-*.conllu")):
    count = 0

    with file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("# sent_id = "):
                count += 1

    print(f"{file.name}: {count} sentences")