import pandas as pd

RESULTS_PATH = "./results/model_comparison.csv"


def main():

    print("=" * 70)
    print("BiLSTM vs Transformer Comparison")
    print("=" * 70)

    df = pd.read_csv(RESULTS_PATH)

    # Display test results
    print("\nTest Results:")
    print("-" * 70)

    print(
        df.to_string(
            index=False,
            formatters={
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1-score": "{:.4f}".format
            }
        )
    )

    # Get model results
    bilstm = df[df["Model"] == "BiLSTM"].iloc[0]
    transformer = df[df["Model"] == "Transformer"].iloc[0]

    # Compare metrics
    print("\n" + "=" * 70)
    print("Metric Comparison")
    print("=" * 70)

    for metric in ["Precision", "Recall", "F1-score"]:

        bilstm_score = bilstm[metric]
        transformer_score = transformer[metric]

        difference = bilstm_score - transformer_score

        if bilstm_score > transformer_score:
            higher_model = "BiLSTM"
        elif transformer_score > bilstm_score:
            higher_model = "Transformer"
        else:
            higher_model = "Tie"

        print(
            f"{metric:<10}: {higher_model} performed higher "
            f"({difference * 100:.2f} percentage points difference)"
        )

    # Overall comparison based on F1-score
    print("\n" + "=" * 70)
    print("Overall Comparison")
    print("=" * 70)

    if bilstm["F1-score"] > transformer["F1-score"]:
        overall_model = "BiLSTM"
    elif transformer["F1-score"] > bilstm["F1-score"]:
        overall_model = "Transformer"
    else:
        overall_model = "Tie"

    print(f"BiLSTM F1-score     : {bilstm['F1-score'] * 100:.2f}%")
    print(f"Transformer F1-score: {transformer['F1-score'] * 100:.2f}%")

    print(
        f"\nBased on the test-set F1-score, "
        f"{overall_model} performed higher in this experiment."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
