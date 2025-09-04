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
