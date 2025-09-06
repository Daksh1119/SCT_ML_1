# app.py — Professional UI + Explainability + OS-aware Dark Mode
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import streamlit.components.v1 as components
import plotly.graph_objects as go
import requests

# -------------------
# Page config
# -------------------
st.set_page_config(page_title="House Price Prediction", page_icon=":bar_chart:", layout="wide")

# -------------------
# Sidebar: Dark mode toggle (persist)
# -------------------
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

# The checkbox allows explicit override. Default unchecked — UI will still follow OS preference via CSS media query.
dark_mode_override = st.sidebar.checkbox("Enable dark mode (override OS)", value=st.session_state["dark_mode"])
st.session_state["dark_mode"] = dark_mode_override

# -------------------
# CSS: base + prefers-color-scheme media query (auto-detect) + explicit dark override
# -------------------
# Base (light) variables
base_css = r"""
:root{
  --accent: #5b7cff;
  --accent-2: #33d3ff;
  --muted: #6c757d;
  --bg: linear-gradient(180deg,#f6f8fb 0%,#ffffff 100%);
  --card-bg: #ffffff;
  --text: #0f2b3d;
  --muted-2: #97a0aa;
  --shadow: 0 8px 30px rgba(35,40,50,0.04);
}
.stApp { background: var(--bg); color: var(--text); font-family: Inter, Roboto, sans-serif; }
.hero { padding:18px 22px; border-radius:12px; margin-bottom:14px; background: linear-gradient(90deg, rgba(91,124,255,0.09), rgba(91,197,255,0.04)); box-shadow:var(--shadow); position:relative; overflow:hidden; }
.title { display:flex; align-items:center; gap:14px; }
.title h1 { margin:0; font-size:22px; color:var(--text); }
.subtitle { color:var(--muted); margin-top:4px; font-size:13px; }
.card { background:var(--card-bg); padding:18px; border-radius:12px; box-shadow:var(--shadow); margin-bottom:18px; }
.result-box { padding:16px; border-radius:10px; text-align:center; border:1px solid rgba(91,124,255,0.06); background: linear-gradient(180deg,#ffffff,#fbfdff); }
.result-amount { font-size:28px; font-weight:700; color:#063047; }
.muted { color:var(--muted); font-size:13px; }
.stButton>button { background: linear-gradient(90deg,var(--accent), var(--accent-2)); color:white; border:none; padding:10px 14px; border-radius:10px; font-weight:600; box-shadow:0 8px 20px rgba(91,124,255,0.14); }
.stButton>button:hover { transform: translateY(-3px); transition: .12s; }
"""

# Auto-apply dark theme when OS prefers dark (no JS needed)
prefers_dark_css = r"""
@media (prefers-color-scheme: dark) {
  :root{
    --accent: #7aa1ff;
    --accent-2: #1fb6ff;
    --muted: #c7d2da;
    --bg: linear-gradient(180deg,#061018 0%, #081623 100%);
    --card-bg: rgba(10,13,20,0.65);
    --text: #e8f0f6;
    --muted-2: #9fb0bf;
    --shadow: 0 12px 36px rgba(2,6,23,0.6);
  }
  .card { background: var(--card-bg); border: 1px solid rgba(255,255,255,0.02); }
  .hero { background: linear-gradient(90deg, rgba(122,161,255,0.06), rgba(31,182,255,0.03)); }
  .result-box { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border:1px solid rgba(122,161,255,0.06); }
}
"""

# Explicit override CSS (when user checks the sidebar box)
override_dark_css = r"""
:root{
  --accent: #7aa1ff;
  --accent-2: #1fb6ff;
  --muted: #c7d2da;
  --bg: linear-gradient(180deg,#061018 0%, #081623 100%);
  --card-bg: rgba(10,13,20,0.65);
  --text: #e8f0f6;
  --muted-2: #9fb0bf;
  --shadow: 0 12px 36px rgba(2,6,23,0.6);
}
.card { background: var(--card-bg); border: 1px solid rgba(255,255,255,0.02); }
.hero { background: linear-gradient(90deg, rgba(122,161,255,0.06), rgba(31,182,255,0.03)); }
.result-box { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border:1px solid rgba(122,161,255,0.06); }
"""

# Inject CSS: base + prefers-color-scheme
st.markdown(f"<style>{base_css}</style>", unsafe_allow_html=True)
st.markdown(f"<style>{prefers_dark_css}</style>", unsafe_allow_html=True)
# If override checked, inject override CSS (this takes precedence)
if st.session_state["dark_mode"]:
    st.markdown(f"<style>{override_dark_css}</style>", unsafe_allow_html=True)

# -------------------
# Sidebar details
# -------------------
st.sidebar.title("About")
st.sidebar.markdown("""
**House Price Prediction** — a polished demo for portfolio review.

- Linear Regression (log target)
- Single & batch predictions
- Model explainability: standardized coefficients
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** Ames Housing (Kaggle)")
st.sidebar.markdown("[Repository](https://github.com/Daksh1119/SCT_ML_1)")

# -------------------
# Robust logo loading (local first, then GitHub raw URL fallback)
# -------------------
LOGO_LOCAL = Path("logo.png")
RAW_URL = "https://raw.githubusercontent.com/Daksh1119/SCT_ML_1/main/logo.png"

def show_logo(width=84):
    try:
        if LOGO_LOCAL.exists():
            st.image(str(LOGO_LOCAL), width=width)
            return
    except Exception:
        pass
    try:
        resp = requests.head(RAW_URL, timeout=5)
        if resp.status_code == 200:
            st.image(RAW_URL, width=width)
            return
    except Exception:
        pass
    fallback_svg = """
    <svg width="48" height="48" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
      <rect rx="4" width="24" height="24" fill="#5b7cff"/>
      <path d="M6 14h3v4H6zM11 10h3v8h-3zM16 6h3v12h-3z" fill="white"/>
    </svg>
    """
    st.markdown(fallback_svg, unsafe_allow_html=True)

show_logo(width=84)
st.markdown("<div class='hero'><div class='title'><div><h1>House Price Prediction</h1><div class='subtitle'>Estimate property value quickly and professionally</div></div></div></div>", unsafe_allow_html=True)

# -------------------
# Load model function
# -------------------
@st.cache_resource
def load_model():
    p = Path("linear_log.pkl")
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None

model = load_model()

# -------------------
# Explainability helper
# -------------------
FEATURES = ["GrLivArea", "BedroomAbvGr", "FullBath"]

def explain_model(m):
    if m is None:
        return None
    lr = None
    if hasattr(m, "named_steps"):
        for step in m.named_steps.values():
            if hasattr(step, "coef_") and hasattr(step, "intercept_"):
                lr = step
                break
    else:
        lr = m
    if lr is None:
        return None
    coefs = np.ravel(lr.coef_)
    coefs = coefs[:len(FEATURES)]
    pct = (np.exp(coefs) - 1) * 100.0
    return pd.DataFrame({"feature": FEATURES, "coef": coefs, "pct_effect": pct})

expl_df = explain_model(model)

# -------------------
# Tabs: Single & Batch
# -------------------
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Input features")
        st.write("Provide property attributes:")
        grlivarea = st.number_input("Above-ground living area (sqft)", min_value=300, max_value=9000, value=1500, step=50)
        bedrooms = st.slider("Bedrooms above grade", 0, 8, 3)
        bathrooms = st.slider("Full bathrooms above grade", 0, 5, 2)
        st.write("")
        predict = st.button("Predict Price")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Predicted value")
        if predict:
            if model is None:
                st.error("Model file `linear_log.pkl` not found in repo root.")
            else:
                X = pd.DataFrame([[grlivarea, bedrooms, bathrooms]], columns=FEATURES)
                logp = model.predict(X)[0]
                price = np.expm1(logp)
                animate = f"""
                <div class='result-box'>
                  <div style='font-size:12px;color:var(--muted);margin-bottom:8px'>Predicted market value</div>
                  <div id='price' style='font-size:28px;font-weight:700;color:var(--text)'>$0</div>
                  <div style='color:var(--muted);font-size:12px;margin-top:8px'>Model: Linear Regression (log target)</div>
                </div>
                <script>
                  (function(){{
                    const el = document.getElementById('price');
                    const target = {int(round(price))};
                    let current = 0;
                    const step = Math.max(1, Math.round(target/120));
                    function tick() {{
                      current += step;
                      if (current >= target) {{
                        current = target;
                        el.innerText = '$' + current.toLocaleString();
                      }} else {{
                        el.innerText = '$' + current.toLocaleString();
                        requestAnimationFrame(tick);
                      }}
                    }}
                    tick();
                  }})();
                </script>
                """
                components.html(animate, height=160)
        else:
            st.info("Enter inputs and click **Predict Price**.")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🔎 Model explainability — standardized coefficients (1 SD effect)"):
        if expl_df is None:
            st.write("Explainability not available (model not loaded or not in expected format).")
        else:
            st.markdown("Coefficients are shown in model units. For the log-target model, the **percent effect** column approximates the % change in price for a 1-SD increase in the feature: `exp(coef) - 1`.")
            st.dataframe(expl_df.style.format({"coef":"{:.4f}", "pct_effect":"{:.1f}%"}), height=160)

            plot_df = expl_df.copy().sort_values("pct_effect", ascending=False)
            fig = go.Figure(go.Bar(
                x=plot_df["pct_effect"],
                y=plot_df["feature"],
                orientation="h",
                marker=dict(color=plot_df["pct_effect"], colorscale="Tealrose")
            ))
            fig.update_layout(title="Approx % change in price for +1 SD increase (by feature)", xaxis_title="% change in price", margin=dict(l=120))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Batch predictions (CSV upload)")
    st.write("Upload a CSV with columns: `GrLivArea`, `BedroomAbvGr`, `FullBath`")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        try:
            df = pd.read_csv(up)
            required = FEATURES
            if not all(c in df.columns for c in required):
                st.error(f"CSV must include columns: {required}")
            else:
                if model is None:
                    st.error("Model not loaded — add `linear_log.pkl` to repo root.")
                else:
                    preds_log = model.predict(df[required])
                    df["PredictedSalePrice"] = np.expm1(preds_log).round(0).astype(int)
                    st.success("Predictions generated")
                    st.dataframe(df)
                    csv_out = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions (CSV)", data=csv_out, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Failed to process file: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.write("---")
st.markdown("<div style='text-align:center;color:var(--muted-2);font-size:13px'>Built with care • SCT_ML_1 • Daksh1119</div>", unsafe_allow_html=True)
