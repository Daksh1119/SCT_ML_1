import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")
st.title("🏠 House Price Prediction App")

@st.cache_resource
def load_model():
    path = Path("linear_log.pkl")
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

st.markdown("Provide inputs to estimate **SalePrice**:")
grlivarea = st.number_input("GrLivArea (Above ground living area, sqft)", min_value=300, max_value=6000, step=50, value=1500)
bedrooms  = st.slider("BedroomAbvGr (Bedrooms above grade)", 0, 10, 3)
bathrooms = st.slider("FullBath (Full bathrooms above grade)", 0, 5, 2)

if st.button("Predict"):
    if model is None:
        st.warning("Model file `linear_log.pkl` not found. Please add it to the repo root.")
    else:
        X = pd.DataFrame([[grlivarea, bedrooms, bathrooms]],
                         columns=["GrLivArea", "BedroomAbvGr", "FullBath"])
        log_pred = model.predict(X)[0]
        price = float(np.expm1(log_pred))
        st.success(f"💰 Predicted SalePrice: ${price:,.0f}")