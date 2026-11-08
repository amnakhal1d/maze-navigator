"""Streamlit interface for historical used-car listing estimates."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "car_price_pipeline.pkl"
st.set_page_config(page_title="Used-Car Listing Price Predictor", layout="centered")


@st.cache_resource
def load_artifacts(path, modified_at):
    return joblib.load(path)


st.title("Car Price Predictor")
st.write("Predicts used-car listing prices using linear regression.")

if not MODEL_PATH.exists():
    st.error("Model missing. Run: python train_model.py")
    st.stop()
try:
    artifacts = load_artifacts(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
    pipeline = artifacts["pipeline"]
    options = artifacts["options"]
except Exception as error:
    st.error(f"Could not load model: {error}. Retrain with python train_model.py.")
    st.stop()

left, right = st.columns(2)
with left:
    company = st.selectbox("Company", sorted(options))
    name = st.selectbox("Car model / listing name", sorted(options[company]))
    fuel_type = st.selectbox("Fuel type", options[company][name])
with right:
    low, high = artifacts["year_min"], artifacts["year_max"]
    year = st.number_input("Manufacturing year", min_value=low, max_value=high,
                           value=high, step=1)
    kms = st.number_input("Kilometres driven", min_value=0,
                          value=35000, step=1000)
if kms > artifacts["kms_max"]:
    st.warning("Mileage exceeds the largest training value; this estimate is extrapolation.")
st.caption("Year choices are restricted to the overall training range. "
           "Some model/year/mileage combinations may still be poorly represented.")
if st.button("Estimate listing price", use_container_width=True):
    row = pd.DataFrame([{"name": name, "company": company, "year": int(year),
                         "fuel_type": fuel_type, "kms_driven": int(kms)}])
    try:
        prediction = float(pipeline.predict(row)[0])
        if not np.isfinite(prediction) or prediction <= 0:
            st.warning("The model returned an invalid or non-positive price. "
                       "Do not use this prediction as a valuation.")
        else:
            st.success(f"Estimated historical listing price: INR {prediction:,.0f}")
            st.caption("No artificial minimum price or clipping has been applied.")
    except Exception as error:
        st.error(f"Prediction failed: {error}")
with st.expander("Held-out evaluation"):
    st.dataframe(pd.DataFrame(artifacts["metrics"]).T)
    st.caption("MAE and RMSE are in INR. R2 is not percentage accuracy. "
               "Average errors are not individual confidence intervals.")
