# Used-Car Price Predictor

A Streamlit app that estimates used-car listing prices using linear regression and historical Quikr data. Prices are in Indian rupees (INR).

## Features

- Data cleaning and duplicate removal
- Categorical encoding and numeric scaling
- Linear regression with a mean-price baseline
- MAE, RMSE and R² evaluation
- Interactive prediction interface
- Automated tests and diagnostic reports

## Dataset

Place `quik_car.csv` in the `data/` folder.

Required columns:
`name`, `company`, `year`, `Price`, `kms_driven`, `fuel_type`.

The local dataset contains 892 records, with 724 remaining after cleaning and exact deduplication.

Dataset source: ADD YOUR ACTUAL DOWNLOAD LINK HERE.

## Setup and Run

Use Python 3.10 or 3.11. Run these commands from the project folder in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

## Results

80/20 train-test split with `random_state=42`.

| Model | MAE (INR) | RMSE (INR) | R² |
|---|---:|---:|---:|
| Mean baseline | 271,807.62 | 448,619.76 | -0.0142 |
| Linear regression | 188,219.42 | 461,798.03 | -0.0747 |

Linear regression improved MAE but performed worse on RMSE and R². The current model needs further investigation before practical use.

## Limitations

- Historical listing prices are not current market valuations or final sale prices.
- Some car models have limited training examples.

