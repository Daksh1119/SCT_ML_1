import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# Page Config

st.set_page_config(
    page_title="House Price Prediction",
    page_icon=":bar_chart:",
    layout="wide"
)


# Custom CSS

st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', sans-serif;
    }
    .card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .subtitle {
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    .prediction {
        background: #e8f5e9;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #2ecc71;
        font-size: 1.2rem;
        font-weight: bold;
        color: #27ae60;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


# Sidebar

st.sidebar.title("📊 About this App")
st.sidebar.markdown(
    """
    This **House Price Prediction App** estimates the market value of a house 
    based on square footage, number of bedrooms, and bathrooms.  

    **Features:**
    - Linear Regression with log-transform  
    - Real-time prediction  
    - Batch prediction via CSV upload  

    **Dataset:** Ames Housing Dataset (Kaggle)  
    **Repo:** [GitHub](https://github.com/Daksh1119/SCT_ML_1)
    """
)

st.sidebar.markdown("---")
st.sidebar.subheader("📘 How to Use")
st.sidebar.markdown(
    """
    - Use **Single Prediction** for one property  
    - Use **Batch Prediction** to upload a CSV and get multiple predictions  
    - Download results as a CSV file  
    """
)

st.sidebar.markdown("---")
st.sidebar.info("Developed by **Daksh1119** | ML Portfolio Project")


# Logo / Header

logo_path = Path("logo.png")
if logo_path.exists():
    st.image(str(logo_path), width=120)
st.markdown("<h1>House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Estimate housing market value with a trained regression model.</p>", unsafe_allow_html=True)
st.markdown("---")


# Load Model

@st.cache_resource
def load_model():
    path = Path("linear_log.pkl")
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

model = load_model()


# Tabs: Single vs Batch Prediction

tab1, tab2 = st.tabs(["🔹 Single Prediction", "📂 Batch Prediction"])


# Tab 1: Single Prediction

with tab1:
    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📋 Input Property Details")

        grlivarea = st.number_input("GrLivArea (sqft)", min_value=300, max_value=6000, step=50, value=1500)
        bedrooms  = st.slider("Bedrooms above grade", 0, 10, 3)
        bathrooms = st.slider("Full bathrooms above grade", 0, 5, 2)

        predict_btn = st.button("🔍 Predict Price", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("💡 Predicted Price")

        if predict_btn:
            if model is None:
                st.warning("⚠️ Model file `linear_log.pkl` not found. Please add it to the repo root.")
            else:
                X = pd.DataFrame([[grlivarea, bedrooms, bathrooms]],
                                 columns=["GrLivArea", "BedroomAbvGr", "FullBath"])
                log_pred = model.predict(X)[0]
                price = float(np.expm1(log_pred))
                st.markdown(f"<div class='prediction'>💰 Estimated Price: ${price:,.0f}</div>", unsafe_allow_html=True)
        else:
            st.info("Enter details and click **Predict Price** to see results.")
        st.markdown("</div>", unsafe_allow_html=True)


# Tab 2: Batch Prediction

with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📂 Batch Prediction via CSV Upload")
    st.caption("Upload a CSV with columns: `GrLivArea`, `BedroomAbvGr`, `FullBath`")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if all(col in df.columns for col in ["GrLivArea", "BedroomAbvGr", "FullBath"]):
                log_preds = model.predict(df[["GrLivArea", "BedroomAbvGr", "FullBath"]])
                df["PredictedSalePrice"] = np.expm1(log_preds).round(0).astype(int)

                st.success("✅ Predictions generated successfully!")
                st.dataframe(df)

                # Download option
                csv_out = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Predictions", 
                    data=csv_out, 
                    file_name="batch_predictions.csv", 
                    mime="text/csv"
                )
            else:
                st.error("CSV must include columns: GrLivArea, BedroomAbvGr, FullBath")
        except Exception as e:
            st.error(f"Failed to process file: {e}")
    st.markdown("</div>", unsafe_allow_html=True)


# Footer

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6c757d;'>Built with ❤️ using Streamlit | Project: SCT_ML_1</p>",
    unsafe_allow_html=True
)
