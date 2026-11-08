from datetime import datetime
from pathlib import Path

import pandas as pd

from data_utils import clean_data, find_dataset

BASE_DIR = Path(__file__).resolve().parent


def main():
    source = find_dataset(BASE_DIR)
    raw = pd.read_csv(source)

    cleaned, report = clean_data(
        raw,
        max_year=datetime.now().year,
    )

    output_dir = BASE_DIR / "reports"
    output_dir.mkdir(exist_ok=True)

    print("Dataset:", source.name)
    print("\nCleaning report:")

    for key, value in report.items():
        print(f"{key}: {value}")

    print("\nFuel categories:")
    print(cleaned["fuel_type"].value_counts().to_string())

    print("\n15 highest recorded prices:")
    print(
        cleaned.sort_values("Price", ascending=False)
        .head(15)
        .to_string(index=False)
    )

    cleaned.sort_values(
        "Price", ascending=False
    ).head(25).to_csv(
        output_dir / "highest_prices_for_review.csv",
        index=False,
    )

    # This is a review flag, not a rule for deletion.
    low_mileage = cleaned[
        cleaned["kms_driven"] < 1000
    ]

    low_mileage.to_csv(
        output_dir / "low_mileage_for_review.csv",
        index=False,
    )

    allowed_fuels = {"Petrol", "Diesel", "LPG", "CNG", "Electric"}

    unusual_fuels = cleaned[
        ~cleaned["fuel_type"].isin(allowed_fuels)
    ]

    unusual_fuels.to_csv(
        output_dir / "fuel_labels_for_review.csv",
        index=False,
    )

    print("\nAudit files saved in reports/.")
    print("No rows were changed or deleted by this audit.")


if __name__ == "__main__":
    main()