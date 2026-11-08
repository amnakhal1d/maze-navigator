"""Validation and cleaning for Quikr-style listings."""
from pathlib import Path
import pandas as pd

FEATURES = ["name", "company", "year", "kms_driven", "fuel_type"]
REQUIRED = FEATURES + ["Price"]


def find_dataset(base_dir):
    for folder in (base_dir / "data", base_dir):
        for name in ("quik_car.csv", "quikr_car.csv", "quikr_car_data.csv"):
            candidate = folder / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        "Put quik_car.csv or quikr_car.csv in the data folder, "
        "or pass --data PATH to train_model.py."
    )


def clean_data(raw, max_year):
    df = raw.copy()
    df.columns = df.columns.str.strip()
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df[REQUIRED].copy()
    original_count = len(df)
    for column in ("name", "company", "fuel_type"):
        df[column] = (df[column].astype("string").str.strip()
                      .str.replace(r"\s+", " ", regex=True))
        df[column] = df[column].replace("", pd.NA)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for column in ("Price", "kms_driven"):
        text = df[column].astype("string").str.replace(",", "", regex=False).str.strip()
        if column == "kms_driven":
            text = text.str.replace(r"(?i)\s*kms?\s*$", "", regex=True).str.strip()
        df[column] = pd.to_numeric(text, errors="coerce")
    df = df.dropna(subset=REQUIRED)
    valid = ((df["Price"] > 0) & (df["kms_driven"] >= 0)
             & df["year"].between(1900, max_year)
             & (df["year"] % 1 == 0))
    df = df.loc[valid].copy()
    before_duplicates = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df["year"] = df["year"].astype(int)
    report = {"raw_rows": original_count, "clean_rows": len(df),
              "invalid_rows_removed": original_count - before_duplicates,
              "duplicates_removed": before_duplicates - len(df)}
    return df, report
