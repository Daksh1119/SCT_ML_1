# RUN THIS IN A SINGLE COLAB CELL

import os, shutil, subprocess, textwrap, getpass, json, sys
from pathlib import Path

GH_USER = "Daksh1119"
REPO    = "SCT_ML_1"

# === 1) Ask for your GitHub PAT securely ===
print("Enter your GitHub Personal Access Token (PAT) — input hidden:")
PAT = getpass.getpass()

# === 2) Clean local workspace and clone fresh empty repo ===
workdir = Path("/content")
repo_dir = workdir / REPO

if repo_dir.exists():
    shutil.rmtree(repo_dir)

clone_url = f"https://{GH_USER}:{PAT}@github.com/{GH_USER}/{REPO}.git"
subprocess.run(["git", "clone", clone_url, str(repo_dir)], check=True)

os.chdir(repo_dir)

# === 3) Create folders ===
(repo_dir / "notebooks").mkdir(parents=True, exist_ok=True)
(repo_dir / "docs").mkdir(parents=True, exist_ok=True)

# === 4) Copy trained model if found ===
# If you trained & saved: joblib.dump(model, "/content/linear_log.pkl")
model_src = workdir / "linear_log.pkl"
if model_src.exists():
    shutil.copy2(model_src, repo_dir / "linear_log.pkl")
else:
    print("⚠️  No /content/linear_log.pkl found. App will show a friendly message until you add it.")

# === 5) Copy your notebook if it’s at /content; else create a placeholder ===
nb_src = workdir / "house_price_analysis.ipynb"
nb_dst = repo_dir / "notebooks" / "house_price_analysis.ipynb"
if nb_src.exists():
    shutil.copy2(nb_src, nb_dst)
else:
    # Minimal valid notebook placeholder
    nb_json = {
        "cells": [{
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# House Price Analysis\n",
                "\n",
                "This is a placeholder. Upload your full notebook from Colab when ready."
            ]
        }],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 2
    }
    nb_dst.write_text(json.dumps(nb_json, indent=2))

# === 6) Create app.py (Streamlit) ===
app_py = textwrap.dedent("""
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
""").strip()
(repo_dir / "app.py").write_text(app_py, encoding="utf-8")

# === 7) requirements.txt ===
requirements = textwrap.dedent("""
    streamlit
    pandas
    numpy
    scikit-learn
    plotly
    joblib
    matplotlib
    seaborn
""").strip() + "\n"
(repo_dir / "requirements.txt").write_text(requirements, encoding="utf-8")

# === 8) README.md (clean & professional) ===
readme = textwrap.dedent("""
    # 🏠 House Price Prediction (SCT_ML_1)

    A machine learning project that predicts house prices using **Linear Regression** on features like square footage, bedrooms, and bathrooms.  
    The project compares **raw vs log-transformed targets**, selects the best model, and deploys it with an interactive **Streamlit app**.

    ---

    ## 📊 Features
    - Data preprocessing & feature engineering  
    - Model comparison (raw vs log-transformed)  
    - Cross-validation with metrics (MAE, RMSE, R^2)  
    - Outlier handling & standardization  
    - Interactive **Streamlit dashboard** for real-time predictions  
    - Visualizations with **Plotly**

    ---

    ## 🚀 Live Demo
    👉 **Streamlit App**: *(configure after first push)*  
    - Go to **streamlit.io** → Deploy new app → Repo: `Daksh1119/SCT_ML_1`, Branch: `main`, File: `app.py`.

    ---

    ## ⚙️ Installation
    ```bash
    git clone https://github.com/Daksh1119/SCT_ML_1.git
    cd SCT_ML_1
    pip install -r requirements.txt
    ```

    Run the app locally:
    ```bash
    streamlit run app.py
    ```

    ---

    ## 📊 Dashboard (from notebook)
    If you exported an interactive dashboard (`docs/dashboard.html`) and a snapshot (`newplot.png`), they’ll show here:

    ![Dashboard](newplot.png)

    👉 If you enable **GitHub Pages** (Settings → Pages → Branch: `main`, Folder: `/docs`), the interactive dashboard will be at:  
    `https://daksh1119.github.io/SCT_ML_1/dashboard.html`

    ---

    ## 🔑 Key Insights (from analysis)
    - **Log transformation** of SalePrice improves error distribution and often reduces RMSE.  
    - **Outliers** can distort linear coefficients; trimming or robust modeling helps.  
    - Key predictors include `GrLivArea`; adding quality variables (e.g., `OverallQual`) typically improves fit.

    ---

    ## 📂 Project Structure
    ```
    SCT_ML_1/
    ├─ app.py
    ├─ requirements.txt
    ├─ README.md
    ├─ linear_log.pkl               # trained model (add after training)
    ├─ notebooks/
    │  └─ house_price_analysis.ipynb
    ├─ docs/
    │  └─ dashboard.html            # optional, for GitHub Pages
    └─ newplot.png                  # optional snapshot of dashboard
    ```

    ---

    ## 📌 Author
    **Daksh1119** · Repository: [SCT_ML_1](https://github.com/Daksh1119/SCT_ML_1)
""").strip() + "\n"
(repo_dir / "README.md").write_text(readme, encoding="utf-8")

# === 9) Optional: add placeholder dashboard.html if not present ===
docs_html = repo_dir / "docs" / "dashboard.html"
if not docs_html.exists():
    docs_html.write_text("<!doctype html><title>Dashboard</title><h2>Dashboard placeholder</h2>", encoding="utf-8")

# === 10) Git config, commit, push ===
subprocess.run(["git", "config", "user.name", GH_USER], check=True)
subprocess.run(["git", "config", "user.email", f"{GH_USER}@users.noreply.github.com"], check=True)
subprocess.run(["git", "add", "-A"], check=True)
# If the repo is truly empty, initial commit message:
subprocess.run(["git", "commit", "-m", "Initial clean project setup (app, reqs, README, notebooks, docs, model if present)"], check=True)
subprocess.run(["git", "branch", "-M", "main"], check=True)
# Use tokenized remote for push to avoid credential prompts
subprocess.run(["git", "remote", "set-url", "origin", clone_url], check=True)
subprocess.run(["git", "push", "-u", "origin", "main"], check=True)

print("\n✅ Done. Repo pushed to GitHub.")
if not model_src.exists():
    print("ℹ️ Tip: Re-run this cell after saving /content/linear_log.pkl to include the model.")
