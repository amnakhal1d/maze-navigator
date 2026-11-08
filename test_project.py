"""Small automated checks using synthetic data, not the real dataset."""
import tempfile
import unittest
from pathlib import Path
import joblib
import pandas as pd
from data_utils import FEATURES, clean_data
from train_model import build_pipeline, train


def sample_data():
    return pd.DataFrame({"name": ["Model A"] * 15 + ["Model B"] * 15,
        "company": ["Brand"] * 30, "fuel_type": ["Petrol"] * 30,
        "year": [2000 + i % 20 for i in range(30)],
        "kms_driven": [10000 + 1000 * i for i in range(30)],
        "Price": [100000 + 2000 * i for i in range(30)]})


class ProjectTests(unittest.TestCase):
    def test_cleaning(self):
        raw = sample_data().iloc[:2].copy()
        raw["Price"] = ["1,20,000", "Ask For Price"]
        raw["kms_driven"] = ["35,000 kms", "bad"]
        cleaned, report = clean_data(raw, 2026)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["Price"], 120000)
        self.assertEqual(cleaned.iloc[0]["kms_driven"], 35000)

    def test_duplicates(self):
        raw = sample_data()
        cleaned, report = clean_data(pd.concat([raw, raw.iloc[:1]]), 2026)
        self.assertEqual(report["duplicates_removed"], 1)
        self.assertEqual(len(cleaned), 30)

    def test_missing_column(self):
        with self.assertRaises(ValueError):
            clean_data(sample_data().drop(columns="Price"), 2026)

    def test_invalid_year(self):
        raw = sample_data().iloc[:1].copy()
        raw["year"] = 2099
        cleaned, _ = clean_data(raw, 2026)
        self.assertTrue(cleaned.empty)

    def test_unknown_category(self):
        df = sample_data()
        model = build_pipeline().fit(df[FEATURES], df["Price"])
        row = df[FEATURES].iloc[:1].copy()
        row["name"] = "Unseen model"
        self.assertEqual(len(model.predict(row)), 1)

    def test_training_and_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source = base / "quik_car.csv"
            sample_data().to_csv(source, index=False)
            report = train(source, base_dir=base, max_year=2026)
            saved = joblib.load(base / "models" / "car_price_pipeline.pkl")
            result = saved["pipeline"].predict(sample_data()[FEATURES].iloc[:1])
            self.assertEqual(len(result), 1)
            self.assertEqual(report["training_rows"], 24)
            self.assertTrue((base / "reports" / "evaluation.png").exists())


if __name__ == "__main__":
    unittest.main()
