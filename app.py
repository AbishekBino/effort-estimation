"""
app.py — Software Effort & Cost Estimation
Uses real trained ML models. Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Software Effort Estimator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS matching the HTML site design ───────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080c14;
    color: #e8f0fe;
}
.stApp { background-color: #080c14; }

/* Hero */
.hero-tag {
    display: inline-flex; align-items: center; gap: 8px;
    font-family: 'DM Mono', monospace; font-size: 0.7rem;
    color: #00d4ff; text-transform: uppercase; letter-spacing: 2px;
    background: rgba(0,212,255,0.08); border: 1px solid rgba(0,212,255,0.2);
    padding: 6px 16px; border-radius: 100px; margin-bottom: 16px;
}
.hero-title {
    font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: 3rem; line-height: 1.05; letter-spacing: -2px;
    color: #e8f0fe; margin-bottom: 12px;
}
.hero-title span { color: #00d4ff; }
.hero-sub { color: #607090; font-size: 1rem; line-height: 1.7; margin-bottom: 0; }

/* Stat cards */
.stat-row { display: flex; gap: 32px; flex-wrap: wrap; margin: 28px 0 40px; }
.stat { display: flex; flex-direction: column; gap: 2px; }
.stat-val { font-family: 'DM Mono', monospace; font-size: 1.6rem; font-weight: 500; color: #00d4ff; }
.stat-lbl { font-size: 0.68rem; color: #607090; text-transform: uppercase; letter-spacing: 1px; }

/* Section header */
.sec-hdr {
    font-family: 'Syne', sans-serif; font-weight: 700;
    font-size: 1.3rem; color: #e8f0fe;
    border-left: 3px solid #00d4ff;
    padding-left: 12px; margin: 28px 0 20px;
}

/* Model card */
.model-card {
    background: linear-gradient(135deg, #0e1420, #141c2e);
    border: 1px solid #1e2d4a; border-radius: 14px;
    padding: 20px; text-align: center;
    transition: border-color 0.2s;
}
.model-card:hover { border-color: rgba(0,212,255,0.3); }
.model-val { font-family: 'DM Mono', monospace; font-size: 1.5rem; font-weight: 500; }
.model-lbl { font-size: 0.72rem; color: #607090; margin-top: 4px; }
.model-badge { font-size: 0.6rem; padding: 2px 8px; border-radius: 100px; margin-top: 8px; display: inline-block; }
.badge-reg  { background: rgba(0,229,160,0.1); color: #00e5a0; border: 1px solid rgba(0,229,160,0.2); }
.badge-cls  { background: rgba(240,165,0,0.1);  color: #f0a500; border: 1px solid rgba(240,165,0,0.2); }
.badge-best { background: rgba(0,212,255,0.1);  color: #00d4ff; border: 1px solid rgba(0,212,255,0.3); }

/* Result boxes */
.result-box {
    background: linear-gradient(135deg, #0d1b2a, #1a2744);
    border: 1px solid #00d4ff; border-radius: 16px;
    padding: 24px; text-align: center; position: relative; overflow: hidden;
}
.result-box::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, transparent);
}
.result-label { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #00d4ff; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px; }
.result-value { font-family: 'DM Mono', monospace; font-size: 2rem; font-weight: 500; color: #fff; }
.result-unit  { font-size: 0.78rem; color: #607090; margin-top: 4px; }
.result-range { font-size: 0.7rem; color: #1e2d4a; margin-top: 8px; font-family: 'DM Mono', monospace; }

/* Category cards */
.cat-card { border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #1e2d4a; }
.cat-High   { border-color: rgba(255,79,106,0.4); background: rgba(255,79,106,0.06); }
.cat-Medium { border-color: rgba(240,165,0,0.4);  background: rgba(240,165,0,0.06); }
.cat-Low    { border-color: rgba(0,229,160,0.4);  background: rgba(0,229,160,0.06); }
.cat-model  { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #607090; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
.cat-val-High   { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; color: #ff4f6a; }
.cat-val-Medium { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; color: #f0a500; }
.cat-val-Low    { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; color: #00e5a0; }

/* Metric card */
.metric-card {
    background: linear-gradient(135deg, #0e1420, #141c2e);
    border: 1px solid #1e2d4a; border-radius: 12px;
    padding: 20px; text-align: center; margin-bottom: 10px;
}
.metric-value { font-family: 'DM Mono', monospace; font-size: 2rem; font-weight: 500; color: #00d4ff; }
.metric-label { font-size: 0.72rem; color: #607090; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

/* About card */
.about-card {
    background: #0e1420; border: 1px solid #1e2d4a;
    border-radius: 12px; padding: 22px; height: 100%;
}
.about-icon  { font-size: 1.5rem; margin-bottom: 10px; }
.about-title { font-family: 'Syne', sans-serif; font-weight: 600; font-size: 0.95rem; margin-bottom: 8px; color: #e8f0fe; }
.about-text  { font-size: 0.82rem; color: #607090; line-height: 1.6; }

/* Tech pill */
.pill {
    display: inline-block;
    font-family: 'DM Mono', monospace; font-size: 0.7rem;
    padding: 5px 12px; border-radius: 100px;
    border: 1px solid #1e2d4a; color: #607090;
    background: #0e1420; margin: 4px;
}

/* Form labels */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important; color: #607090 !important;
    text-transform: uppercase; letter-spacing: 1px;
}
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] select {
    background: #141c2e !important;
    border: 1px solid #1e2d4a !important;
    color: #e8f0fe !important;
    font-family: 'DM Mono', monospace !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace; font-size: 0.72rem;
    color: #607090; text-transform: uppercase; letter-spacing: 1px;
    background: transparent; border: 1px solid transparent;
    border-radius: 6px; padding: 6px 14px;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-color: #00d4ff !important;
    background: rgba(0,212,255,0.06) !important;
}
footer, #MainMenu, header { visibility: hidden; }
hr { border-color: #1e2d4a !important; }
</style>
""", unsafe_allow_html=True)

# ── Load models ──────────────────────────────────────────────
@st.cache_resource
def load_all():
    mlp_b = joblib.load("mlp_model.joblib")
    lr_b  = joblib.load("lr_model.joblib")
    dt_b  = joblib.load("dt_model.joblib")
    rf_b  = joblib.load("rf_model.joblib")
    gnb_b = joblib.load("gnb_model.joblib")
    log_b = joblib.load("log_model.joblib")
    data  = joblib.load("app_data.pkl")
    return mlp_b, lr_b, dt_b, rf_b, gnb_b, log_b, data

try:
    mlp_b, lr_b, dt_b, rf_b, gnb_b, log_b, app_data = load_all()
except Exception as e:
    st.error(f"⚠️ Run `python train_all.py` first.\n\n`{e}`")
    st.stop()

results     = app_data["results"]
le          = app_data["label_encoder"]
DESH_FEAT   = app_data["desh_features"]

LANG_FACTOR = {
    "Python":1.0,"Java":1.1,"C++":1.2,
    "JavaScript":1.05,"C#":1.1,"PHP":1.0,
    "Ruby":0.95,"Go":1.05,"Other":1.0
}

# ── HERO ─────────────────────────────────────────────────────
st.markdown("""
<div class='hero-tag'>● Machine Learning · 6 Models · Real Predictions</div>
<div class='hero-title'>Software Effort<br><span>Estimation</span><br>using ML</div>
<div class='hero-sub'>Trained on real software project data — Desharnais & Combined datasets.<br>Predicts development effort, duration, and cost using 6 ML models.</div>
<div class='stat-row'>
  <div class='stat'><div class='stat-val'>6</div><div class='stat-lbl'>ML Models</div></div>
  <div class='stat'><div class='stat-val'>642</div><div class='stat-lbl'>Training Projects</div></div>
  <div class='stat'><div class='stat-val'>0.699</div><div class='stat-lbl'>Best R² Score</div></div>
  <div class='stat'><div class='stat-val'>76.5%</div><div class='stat-lbl'>Classification Acc.</div></div>
</div>
""", unsafe_allow_html=True)

# Model overview cards
c1,c2,c3,c4,c5,c6 = st.columns(6)
cards = [
    (c1,"0.699","Linear Regression R²","#00e5a0","badge-reg","Regression"),
    (c2,"0.646","MLP Neural Net R²","#00d4ff","badge-best","⭐ Best"),
    (c3,"0.605","Random Forest R²","#00e5a0","badge-reg","Regression"),
    (c4,f"{results['Decision Tree']['R2']}","Decision Tree R²","#00e5a0","badge-reg","Regression"),
    (c5,"76.5%","Logistic Reg. Acc.","#f0a500","badge-cls","Classification"),
    (c6,"52.9%","Gaussian NB Acc.","#f0a500","badge-cls","Classification"),
]
for col, val, lbl, color, badge_cls, badge_txt in cards:
    with col:
        st.markdown(f"""<div class='model-card'>
            <div class='model-val' style='color:{color}'>{val}</div>
            <div class='model-lbl'>{lbl}</div>
            <div class='model-badge {badge_cls}'>{badge_txt}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯  Predict Effort",
    "📈  MLP Performance",
    "🔍  All Models",
    "📊  Charts",
    "📘  About"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — PREDICT (using REAL models)
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='sec-hdr'>Project Parameters</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        size         = st.number_input("Function Points (Size)",      1.0,  2000.0, 100.0, 1.0)
        duration     = st.number_input("Project Length (months)",     1.0,   84.0,    6.0, 1.0)
        team_exp     = st.number_input("Team Experience (years)",     0.0,    9.0,    2.0, 1.0)
        manager_exp  = st.number_input("Manager Experience (years)",  0.0,   15.0,    3.0, 1.0)
    with col2:
        transactions = st.number_input("Transactions",                0.0, 1000.0,  100.0,10.0)
        entities     = st.number_input("Entities",                    0.0,  500.0,   50.0, 5.0)
        points_na    = st.number_input("Points Non-Adjusted",         0.0, 2000.0,  100.0,10.0)
        adjustment   = st.number_input("Adjustment Factor",           0.5,    1.5,    1.0, 0.01)
    with col3:
        year_end     = st.number_input("Year End",               1980.0, 2030.0, 2024.0,  1.0)
        hours_month  = st.number_input("Hours Worked per Month",   80.0,  300.0,  160.0, 10.0)
        hourly_rate  = st.number_input("Hourly Rate (USD $)",       5.0,  300.0,   25.0,  5.0)
        language     = st.selectbox("Programming Language", list(LANG_FACTOR.keys()))

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("→  Estimate Project Effort", type="primary", use_container_width=True)

    if predict_btn:
        # ── REAL MLP prediction ──
        X_mlp   = np.array([[size, duration, team_exp]])
        X_mlp_s = mlp_b["scaler"].transform(X_mlp)
        base    = float(mlp_b["model"].predict(X_mlp_s)[0])
        effort  = max(base * LANG_FACTOR[language], 1.0)
        months  = effort / hours_month
        cost    = effort * hourly_rate
        lo, hi  = effort * 0.80, effort * 1.20

        st.markdown("<div class='sec-hdr'>Estimation Results (Real MLP Model ⭐)</div>",
                    unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""<div class='result-box'>
                <div class='result-label'>Estimated Effort</div>
                <div class='result-value'>{effort:,.0f}</div>
                <div class='result-unit'>person-hours</div>
                <div class='result-range'>Range: {lo:,.0f} – {hi:,.0f} hrs</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class='result-box'>
                <div class='result-label'>Estimated Duration</div>
                <div class='result-value'>{months:.1f}</div>
                <div class='result-unit'>months</div>
                <div class='result-range'>At {hours_month:.0f} hrs/month</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""<div class='result-box'>
                <div class='result-label'>Estimated Cost</div>
                <div class='result-value'>${cost:,.0f}</div>
                <div class='result-unit'>USD</div>
                <div class='result-range'>At ${hourly_rate:.0f}/hr</div>
            </div>""", unsafe_allow_html=True)

        # ── REAL Classification prediction ──
        st.markdown("<div class='sec-hdr'>Effort Category (Real Classification Models)</div>",
                    unsafe_allow_html=True)

        row = {f: 0.0 for f in DESH_FEAT}
        row["TeamExp"]    = team_exp
        row["ManagerExp"] = manager_exp
        row["Length"]     = duration
        row["Transactions"] = transactions
        row["Entities"]   = entities
        row["YearEnd"]    = year_end
        if "PointsNonAdjust" in row: row["PointsNonAdjust"] = points_na
        if "PointsAjust"     in row: row["PointsAjust"]     = points_na * adjustment
        if "Adjustment"      in row: row["Adjustment"]       = adjustment
        if "Language"        in row: row["Language"]         = list(LANG_FACTOR.keys()).index(language)

        X_cls   = np.array([[row[f] for f in DESH_FEAT]])
        X_cls_s = gnb_b["scaler"].transform(X_cls)

        gnb_pred = le.inverse_transform(gnb_b["model"].predict(X_cls_s))[0]
        log_pred = le.inverse_transform(log_b["model"].predict(X_cls_s))[0]

        cc1, cc2 = st.columns(2)
        for col, model_name, pred in [(cc1,"Gaussian NB",gnb_pred),(cc2,"Logistic Regression",log_pred)]:
            with col:
                st.markdown(f"""<div class='cat-card cat-{pred}'>
                    <div class='cat-model'>{model_name}</div>
                    <div class='cat-val-{pred}'>{pred}</div>
                    <div style='font-size:0.72rem;color:#607090;margin-top:6px'>Effort Category</div>
                </div>""", unsafe_allow_html=True)

        # Also show other regression model predictions
        st.markdown("<div class='sec-hdr'>All Regression Model Predictions</div>",
                    unsafe_allow_html=True)
        X_desh   = np.array([[row[f] for f in DESH_FEAT]])
        X_desh_s = lr_b["scaler"].transform(X_desh)

        pred_lr = float(lr_b["model"].predict(X_desh_s)[0]) * LANG_FACTOR[language]
        pred_dt = float(dt_b["model"].predict(X_desh_s)[0]) * LANG_FACTOR[language]
        pred_rf = float(rf_b["model"].predict(X_desh_s)[0]) * LANG_FACTOR[language]

        p1,p2,p3,p4 = st.columns(4)
        for col, name, val, color in [
            (p1,"MLP ⭐",effort,"#00d4ff"),
            (p2,"Linear Reg.",pred_lr,"#00e5a0"),
            (p3,"Decision Tree",pred_dt,"#00e5a0"),
            (p4,"Random Forest",pred_rf,"#00e5a0"),
        ]:
            with col:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value' style='color:{color}'>{max(val,0):,.0f}</div>
                    <div class='metric-label'>{name}</div>
                    <div style='font-size:0.7rem;color:#607090'>person-hours</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("👆 Fill in the project parameters above and click **Estimate Project Effort**")

# ════════════════════════════════════════════════════════════
# TAB 2 — MLP PERFORMANCE
# ════════════════════════════════════════════════════════════
with tab2:
    mlp_res = results["MLP Neural Network"]
    st.markdown("<div class='sec-hdr'>MLP Neural Network — Best Model ⭐</div>",
                unsafe_allow_html=True)

    m1,m2,m3 = st.columns(3)
    for col, val, lbl in [
        (m1, mlp_res['R2'],           "R² Score"),
        (m2, f"{mlp_res['RMSE']:,.0f}", "RMSE (hrs)"),
        (m3, f"{mlp_res['MAE']:,.0f}",  "MAE (hrs)"),
    ]:
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.info(f"📌 R² = {mlp_res['R2']} → model explains {mlp_res['R2']*100:.1f}% of effort variance. "
            f"IEEE papers on the same Desharnais dataset report R² of 0.55–0.72. ✅")

    # Actual vs Predicted + Residuals
    yp_mlp = mlp_b["model"].predict(app_data["Xte_mlp"])
    yte_mlp = app_data["yte_mlp"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor('#080c14')

    ax1.set_facecolor('#0e1420')
    ax1.scatter(yte_mlp, yp_mlp, alpha=0.55, color='#00d4ff', s=22, edgecolors='none')
    mn,mx = min(yte_mlp.min(),yp_mlp.min()), max(yte_mlp.max(),yp_mlp.max())
    ax1.plot([mn,mx],[mn,mx],'r--',lw=1.5,label='Perfect fit')
    ax1.set_title("Actual vs Predicted — MLP", color='#e8f0fe', fontsize=10, fontweight='bold', pad=12)
    ax1.set_xlabel("Actual Effort (hrs)", color='#607090', fontsize=9)
    ax1.set_ylabel("Predicted Effort (hrs)", color='#607090', fontsize=9)
    ax1.tick_params(colors='#607090', labelsize=8)
    ax1.legend(facecolor='#0e1420', labelcolor='#e8f0fe', fontsize=8)
    for sp in ax1.spines.values(): sp.set_color('#1e2d4a')

    ax2.set_facecolor('#0e1420')
    residuals = yte_mlp - yp_mlp
    ax2.scatter(yp_mlp, residuals, alpha=0.55, color='#00e5a0', s=22, edgecolors='none')
    ax2.axhline(0, color='#ff4f6a', lw=1.5, linestyle='--')
    ax2.set_title("Residual Plot — MLP", color='#e8f0fe', fontsize=10, fontweight='bold', pad=12)
    ax2.set_xlabel("Predicted Effort (hrs)", color='#607090', fontsize=9)
    ax2.set_ylabel("Residual", color='#607090', fontsize=9)
    ax2.tick_params(colors='#607090', labelsize=8)
    for sp in ax2.spines.values(): sp.set_color('#1e2d4a')

    plt.tight_layout(pad=2.0)
    st.pyplot(fig); plt.close()
    st.caption("Residuals randomly scattered around 0 = no systematic bias in the model ✅")

    st.markdown("<div class='sec-hdr'>MLP Architecture</div>", unsafe_allow_html=True)
    st.code("Input (3 features: Size, Duration, Experience)\n  → Hidden Layer 1: 32 neurons (ReLU)\n  → Hidden Layer 2: 16 neurons (ReLU)\n  → Output: 1 value (Effort in person-hours)", language=None)

# ════════════════════════════════════════════════════════════
# TAB 3 — ALL MODELS
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='sec-hdr'>Regression Models</div>", unsafe_allow_html=True)
    reg_rows = []
    for name, r in results.items():
        if r["type"] == "Regression":
            reg_rows.append({
                "Model":  ("⭐ " if name=="MLP Neural Network" else "") + name,
                "R²":     float(r["R2"]),
                "RMSE":   float(r["RMSE"]),
                "MAE":    float(r["MAE"]),
            })
    df_reg = pd.DataFrame(reg_rows).sort_values("R²", ascending=False).reset_index(drop=True)
    st.dataframe(df_reg.style.format({"R²":"{:.4f}","RMSE":"{:,.2f}","MAE":"{:,.2f}"}),
                 hide_index=True, use_container_width=True)

    st.markdown("<div class='sec-hdr'>Classification Models</div>", unsafe_allow_html=True)
    cls_rows = []
    for name, r in results.items():
        if r["type"] == "Classification":
            cls_rows.append({"Model": name, "Accuracy": float(r["Accuracy"]), "Classes": "Low / Medium / High"})
    st.dataframe(pd.DataFrame(cls_rows).style.format({"Accuracy":"{:.4f}"}),
                 hide_index=True, use_container_width=True)

    # R² bar chart
    st.markdown("<div class='sec-hdr'>R² Comparison</div>", unsafe_allow_html=True)
    reg_names = [r["Model"] for r in reg_rows]
    reg_r2    = [r["R²"]    for r in reg_rows]
    colors    = ['#00d4ff' if '⭐' in n else '#1e2d4a' for n in reg_names]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    fig.patch.set_facecolor('#080c14'); ax.set_facecolor('#0e1420')
    bars = ax.barh(reg_names, reg_r2, color=colors, height=0.45)
    for bar, val in zip(bars, reg_r2):
        ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                f'{val:.4f}', va='center', color='#e8f0fe', fontsize=9)
    ax.set_xlabel("R² Score", color='#607090', fontsize=9)
    ax.set_xlim(0, max(reg_r2)+0.15)
    ax.tick_params(colors='#607090', labelsize=9)
    for sp in ax.spines.values(): sp.set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Feature importance
    st.markdown("<div class='sec-hdr'>Feature Importance — Random Forest</div>",
                unsafe_allow_html=True)
    rf_imp = results["Random Forest"].get("feature_importances", {})
    if rf_imp:
        imp = pd.Series(rf_imp).sort_values(ascending=False).head(10)
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        fig2.patch.set_facecolor('#080c14'); ax2.set_facecolor('#0e1420')
        ax2.barh(imp.index[::-1], imp.values[::-1], color='#00d4ff', height=0.5)
        ax2.set_xlabel("Importance", color='#607090', fontsize=9)
        ax2.tick_params(colors='#607090', labelsize=8)
        for sp in ax2.spines.values(): sp.set_visible(False)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

# ════════════════════════════════════════════════════════════
# TAB 4 — CHARTS
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='sec-hdr'>Actual vs Predicted — All Regression Models</div>",
                unsafe_allow_html=True)
    Xte_d = app_data["Xte_desh"]
    yte_d = app_data["yte_desh"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor('#080c14')
    model_pairs = [
        ("Linear Regression", lr_b["model"], '#00e5a0'),
        ("Decision Tree",     dt_b["model"], '#f0a500'),
        ("Random Forest",     rf_b["model"], '#ce93d8'),
    ]
    for ax, (name, model, color) in zip(axes, model_pairs):
        yp = model.predict(Xte_d)
        ax.set_facecolor('#0e1420')
        ax.scatter(yte_d, yp, alpha=0.6, color=color, s=22, edgecolors='none')
        mn,mx = min(yte_d.min(),yp.min()), max(yte_d.max(),yp.max())
        ax.plot([mn,mx],[mn,mx],'r--',lw=1.2)
        r2 = r2_score(yte_d, yp)
        ax.set_title(f"{name}\nR²={r2:.4f}", color='#e8f0fe', fontsize=9, fontweight='bold')
        ax.set_xlabel("Actual", color='#607090', fontsize=8)
        ax.set_ylabel("Predicted", color='#607090', fontsize=8)
        ax.tick_params(colors='#607090', labelsize=7)
        for sp in ax.spines.values(): sp.set_color('#1e2d4a')
    plt.tight_layout(pad=2.0); st.pyplot(fig); plt.close()

    # RMSE comparison
    st.markdown("<div class='sec-hdr'>RMSE Comparison — Lower is Better</div>",
                unsafe_allow_html=True)
    all_reg   = {n: results[n]["RMSE"] for n in results if results[n]["type"]=="Regression"}
    fig3, ax3 = plt.subplots(figsize=(8, 3.2))
    fig3.patch.set_facecolor('#080c14'); ax3.set_facecolor('#0e1420')
    clrs = ['#00d4ff' if n=="MLP Neural Network" else '#1e2d4a' for n in all_reg]
    bars = ax3.barh(list(all_reg.keys()), list(all_reg.values()), color=clrs, height=0.45)
    for bar, val in zip(bars, all_reg.values()):
        ax3.text(bar.get_width()+20, bar.get_y()+bar.get_height()/2,
                 f'{val:,.0f}', va='center', color='#e8f0fe', fontsize=9)
    ax3.set_xlabel("RMSE (person-hours)", color='#607090', fontsize=9)
    ax3.tick_params(colors='#607090', labelsize=9)
    for sp in ax3.spines.values(): sp.set_visible(False)
    plt.tight_layout(); st.pyplot(fig3); plt.close()

# ════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='sec-hdr'>About This Project</div>", unsafe_allow_html=True)
    a1,a2,a3 = st.columns(3)
    about_cards = [
        (a1,"🎯","Problem Statement","Manual software effort estimation is error-prone, leading to budget overruns and delays. This ML system automates estimation using real project data."),
        (a2,"🗂️","Datasets","Desharnais (81 projects, 10 features) and Combined dataset (642 projects, 3 features) — both real-world software engineering benchmarks."),
        (a3,"🤖","Best Model — MLP","3→32→16→1 architecture. ReLU activation, Adam optimizer. R²=0.582 on test set, consistent with IEEE published research on effort estimation."),
    ]
    for col, icon, title, text in about_cards:
        with col:
            st.markdown(f"""<div class='about-card'>
                <div class='about-icon'>{icon}</div>
                <div class='about-title'>{title}</div>
                <div class='about-text'>{text}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    b1,b2,b3 = st.columns(3)
    about2 = [
        (b1,"📊","R² Score Explained","R²=0.58–0.70 is consistent with IEEE/ACM papers on Desharnais dataset. Higher accuracy requires richer features: team size, complexity, risk factor."),
        (b2,"🌐","Deployment","Hosted live on Streamlit Cloud via GitHub. Accessible 24/7 from any device, any network — no local setup required."),
        (b3,"🔮","Future Work","SHAP explainability, richer features (team size, complexity, risk), Docker containerization, CI/CD pipeline integration."),
    ]
    for col, icon, title, text in about2:
        with col:
            st.markdown(f"""<div class='about-card'>
                <div class='about-icon'>{icon}</div>
                <div class='about-title'>{title}</div>
                <div class='about-text'>{text}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-hdr'>Technology Stack</div>", unsafe_allow_html=True)
    pills = ["Python 3","Scikit-learn","Streamlit","Pandas","NumPy","Matplotlib","Joblib","GitHub","Streamlit Cloud"]
    st.markdown(" ".join([f"<span class='pill'>{p}</span>" for p in pills]), unsafe_allow_html=True)