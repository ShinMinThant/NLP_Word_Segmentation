from parse_gum import parse_conllu, GUM_FILE


sentences = parse_conllu(GUM_FILE)

for sent in sentences:
    if "can't remember" in sent["text"]:
        print("Text  :", sent["text"])
        print("Tokens:", " | ".join(sent["tokens"]))
        break