import pandas as pd

RESULTS_PATH = "./results/corpus_model_comparison.csv"


def main():
    df = pd.read_csv(RESULTS_PATH)

    print("=" * 72)
    print("NLP WORD SEGMENTATION: MODEL AND CORPUS COMPARISON")
    print("=" * 72)

    print("\n1. TEST RESULTS")
    print("-" * 72)

    print(
        df.to_string(
            index=False,
            formatters={
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1-score": "{:.4f}".format,
            },
        )
    )

    print("\n2. BiLSTM vs Transformer WITHIN EACH CORPUS")
    print("-" * 72)

    for corpus in df["Corpus"].unique():
        subset = df[df["Corpus"] == corpus]
        bilstm = subset[subset["Model"] == "BiLSTM"].iloc[0]
        transformer = subset[subset["Model"] == "Transformer"].iloc[0]

        difference = (bilstm["F1-score"] - transformer["F1-score"]) * 100

        print(f"\n{corpus}:")
        print(f"  BiLSTM F1:      {bilstm['F1-score'] * 100:.2f}%")
        print(f"  Transformer F1: {transformer['F1-score'] * 100:.2f}%")

        if difference > 0:
            print(f"  Higher F1: BiLSTM (+{difference:.2f} percentage points)")
        elif difference < 0:
            print(f"  Higher F1: Transformer (+{-difference:.2f} percentage points)")
        else:
            print("  F1 result: Tie")

    print("\n3. BEST MODEL IN EACH CORPUS")
    print("-" * 72)

    for corpus in df["Corpus"].unique():
        subset = df[df["Corpus"] == corpus]
        best = subset.loc[subset["F1-score"].idxmax()]
        print(
            f"{corpus}: {best['Model']} "
            f"(F1 = {best['F1-score'] * 100:.2f}%)"
        )

    print("\n4. INTERPRETATION")
    print("-" * 72)
    print(
        "BiLSTM achieved a higher test F1-score than Transformer "
        "in both reported corpus experiments."
    )
    print(
        "These are within-corpus results. They do not, by themselves, "
        "measure cross-corpus generalization."
    )
    print(
        "Confirm consistent preprocessing, label construction, and "
        "evaluation before drawing final conclusions."
    )

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
