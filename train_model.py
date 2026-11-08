"""Train, evaluate, and save model pipelines with log-target transformation and non-linear benchmarks."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_utils import FEATURES, clean_data, find_dataset

BASE_DIR = Path(__file__).resolve().parent


def build_preprocessor():
    return ColumnTransformer([
        ("numeric", StandardScaler(), ["year", "kms_driven"]),
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         ["name", "company", "fuel_type"]),
    ])


def build_pipeline():
    """Helper for test compatibility."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", LinearRegression())
    ])


def metrics(y_true, predictions):
    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": float(mean_squared_error(y_true, predictions) ** 0.5),
        "R2": float(r2_score(y_true, predictions))
    }


def train(data_path=None, base_dir=BASE_DIR, max_year=None):
    max_year = max_year or datetime.now().year
    source = Path(data_path) if data_path else find_dataset(base_dir)
    df, cleaning = clean_data(pd.read_csv(source), max_year)
    
    if len(df) < 20:
        raise ValueError("At least 20 unique, valid listings are needed for this demo.")
    if df["Price"].nunique() < 2:
        raise ValueError("Prices must contain more than one distinct value.")

    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES], df["Price"], test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()

    # Define Candidate Models with Log Transformation on Target (Price)
    candidate_models = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "linear_regression": TransformedTargetRegressor(
            regressor=Pipeline([("prep", preprocessor), ("reg", LinearRegression())]),
            func=np.log1p, inverse_func=np.expm1
        ),
        "ridge_regression": TransformedTargetRegressor(
            regressor=Pipeline([("prep", preprocessor), ("reg", Ridge(alpha=10.0))]),
            func=np.log1p, inverse_func=np.expm1
        ),
        "random_forest": TransformedTargetRegressor(
            regressor=Pipeline([
                ("prep", preprocessor),
                ("reg", RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=2))
            ]),
            func=np.log1p, inverse_func=np.expm1
        )
    }

    scores = {}
    fitted_pipelines = {}

    for name, model in candidate_models.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        scores[name] = metrics(y_test, preds)
        fitted_pipelines[name] = model

    # Select the model with the highest R2 score on test set
    best_model_name = max(
        [m for m in scores if m != "mean_baseline"],
        key=lambda m: scores[m]["R2"]
    )
    best_pipeline = fitted_pipelines[best_model_name]
    best_predictions = best_pipeline.predict(x_test)

    # Directories setup
    model_dir = base_dir / "models"
    report_dir = base_dir / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # UI Options Map
    options = {}
    for company, group in x_train.groupby("company"):
        options[company] = {}
        for car_name, model_rows in group.groupby("name"):
            options[company][car_name] = sorted(model_rows["fuel_type"].unique().tolist())

    artifacts = {
        "pipeline": best_pipeline,
        "options": options,
        "year_min": int(x_train["year"].min()),
        "year_max": int(x_train["year"].max()),
        "kms_max": float(x_train["kms_driven"].max()),
        "metrics": scores,
        "best_model": best_model_name,
        "features": FEATURES,
        "sklearn_version": sklearn.__version__
    }

    joblib.dump(artifacts, model_dir / "car_price_pipeline.pkl")

    report = {
        "source_file": source.name,
        "cleaning": cleaning,
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "split_seed": 42,
        "best_model_selected": best_model_name,
        "metrics": scores,
        "sklearn_version": sklearn.__version__
    }

    (report_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    output = x_test.copy()
    output["actual_price"] = y_test
    output["predicted_price"] = best_predictions
    output["residual"] = y_test - best_predictions
    output.to_csv(report_dir / "test_predictions.csv", index=False)

    # Plot evaluation
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_test, best_predictions, alpha=0.6)
    low = min(float(y_test.min()), float(best_predictions.min()))
    high = max(float(y_test.max()), float(best_predictions.max()))
    axes[0].plot([low, high], [low, high], "k--")
    axes[0].set(xlabel="Actual price (INR)", ylabel=f"Predicted ({best_model_name})")

    axes[1].scatter(best_predictions, y_test - best_predictions, alpha=0.6)
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set(xlabel="Predicted price (INR)", ylabel="Residual (INR)")
    
    fig.tight_layout()
    fig.savefig(report_dir / "evaluation.png", dpi=150)
    plt.close(fig)

    print("\n================ Model Benchmark Results ================")
    print(pd.DataFrame(scores).T.to_string())
    print("=========================================================\n")
    print(f"Selected Best Model: {best_model_name}")
    print("Saved updated pipeline and evaluation metrics in reports/.")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--max-year", type=int, default=datetime.now().year)
    args = parser.parse_args()
    try:
        train(args.data, max_year=args.max_year)
    except (OSError, ValueError) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        raise SystemExit(1)