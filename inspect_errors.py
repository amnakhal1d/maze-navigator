from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FILE = BASE_DIR / "reports" / "test_predictions.csv"


def main():
    if not FILE.exists():
        raise FileNotFoundError(
            "Run python train_model.py first."
        )

    data = pd.read_csv(FILE)

    data["absolute_error"] = (
        data["actual_price"] - data["predicted_price"]
    ).abs()

    columns = [
        "name",
        "year",
        "kms_driven",
        "actual_price",
        "predicted_price",
        "absolute_error",
    ]

    print("10 largest test-set errors:\n")

    print(
        data.sort_values("absolute_error", ascending=False)
        .head(10)[columns]
        .to_string(index=False)
    )

    suspicious = data[
        data["actual_price"] == 8500003
    ]

    print("\nINR 8,500,003 listing in test set:",
          not suspicious.empty)

    if not suspicious.empty:
        print(suspicious[columns].to_string(index=False))

    print(
        "\nNon-positive predictions:",
        int((data["predicted_price"] <= 0).sum()),
    )


if __name__ == "__main__":
    main()