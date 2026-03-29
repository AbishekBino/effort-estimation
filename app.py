"""
app.py  —  Software Effort & Cost Estimation
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Software Effort Estimator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600&family=IBM+Plex+Mono&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.metric-card {
    background: linear-gradient(135deg,#1e2130,#252a3d);
    border:1px solid #2e3450; border-radius:12px;
    padding:20px; text-align:center; margin-bottom:10px;
}
.metric-value { font-size:2rem; font-weight:600; color:#4fc3f7; font-family:'IBM Plex Mono',monospace; }
.metric-label { font-size:0.78rem; color:#8892b0; text-transform:uppercase; letter-spacing:1px; margin-top:4px; }
.result-box {
    background:linear-gradient(135deg,#0d1b2a,#1a2744);
    border:1px solid #4fc3f7; border-radius:16px; padding:28px; margin:12px 0;
}
.result-title { font-size:0.72rem; color:#4fc3f7; text-transform:uppercase; letter-spacing:2px; margin-bottom:4px; }
.result-value { font-size:2.1rem; font-weight:600; color:#fff; font-family:'IBM Plex Mono',monospace; }
.result-sub   { color:#546e7a; font-size:0.75rem; margin-top:6px; }
.section-hdr  { font-size:1rem; font-weight:600; color:#ccd6f6; border-left:3px solid #4fc3f7; padding-left:10px; margin:22px 0 14px; }
.best-badge   { background:linear-gradient(90deg,#4fc3f7,#0288d1); color:#fff; padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; }
.cls-badge    { background:#37474f; color:#90a4ae; padding:3px 10px; border-radius:20px; font-size:0.72rem; }
.reg-badge    { background:#1a3a2a; color:#81c784; padding:3px 10px; border-radius:20px; font-size:0.72rem; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load everything ──────────────────────────────────────────
@st.cache_resource
def load_all():
    mlp_b  = joblib.load("mlp_model.joblib")
    lr_b   = joblib.load("lr_model.joblib")
    dt_b   = joblib.load("dt_model.joblib")
    rf_b   = joblib.load("rf_model.joblib")
    gnb_b  = joblib.load("gnb_model.joblib")
    log_b  = joblib.load("log_model.joblib")
    data   = joblib.load("app_data.pkl")
    return mlp_b, lr_b, dt_b, rf_b, gnb_b, log_b, data

try:
    mlp_b, lr_b, dt_b, rf_b, gnb_b, log_b, app_data = load_all()
except Exception as e:
    st.error(f"⚠️ Models not found. Run `python train_all.py` first.\n\n`{e}`")
    st.stop()

results      = app_data["results"]
le           = app_data["label_encoder"]
DESH_FEAT    = app_data["desh_features"]
MLP_FEAT     = app_data["mlp_features"]

LANG_FACTOR  = {"Python":1.0,"Java":1.1,"C++":1.2,"JavaScript":1.05,
                "C#":1.1,"PHP":1.0,"Ruby":0.95,"Go":1.05,"Other":1.0}

# ── Header ───────────────────────────────────────────────────
st.markdown("# 📊 Software Effort & Cost Estimator")
st.markdown("<p style='color:#8892b0;margin-top:-10px'>Machine Learning · 6 Models · Desharnais & Combined Datasets</p>",
            unsafe_allow_html=True)
st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Predict Effort",
    "📈 MLP Performance",
    "🔍 All Models",
    "📊 Charts",
    "📘 About"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-hdr'>Project Parameters</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        size        = st.number_input("Function Points (Size)", 1.0, 2000.0, 100.0, 1.0)
        duration    = st.number_input("Project Length (months)", 1.0, 84.0, 6.0, 1.0)
        team_exp    = st.slider("Team Experience (years)", 0, 9, 2)
    with c2:
        manager_exp = st.slider("Manager Experience (years)", 0, 15, 3)
        hours_month = st.number_input("Hours Worked per Month", 80.0, 300.0, 160.0, 10.0)
        hourly_rate = st.number_input("Hourly Rate (USD $)", 5.0, 300.0, 25.0, 5.0)
    with c3:
        language    = st.selectbox("Programming Language", list(LANG_FACTOR.keys()))
        year_end    = st.slider("Year End (project completion year)", 1980, 2030, 2024)
        transactions= st.number_input("Transactions", 0.0, 1000.0, 100.0, 10.0)
        entities    = st.number_input("Entities", 0.0, 500.0, 50.0, 5.0)

    predict_btn = st.button("🚀 Estimate Project Effort", width="stretch", type="primary")

    if predict_btn:
        # MLP prediction (Size, Duration, Experience only)
        X_mlp   = np.array([[size, duration, team_exp]])
        X_mlp_s = mlp_b["scaler"].transform(X_mlp)
        base    = float(mlp_b["model"].predict(X_mlp_s)[0])
        effort  = max(base * LANG_FACTOR[language], 1.0)
        months  = effort / hours_month
        cost    = effort * hourly_rate
        lo, hi  = effort * 0.80, effort * 1.20

        st.markdown("<div class='section-hdr'>Estimation Results (MLP — Best Model ⭐)</div>",
                    unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""<div class='result-box'>
                <div class='result-title'>Estimated Effort</div>
                <div class='result-value'>{effort:,.0f}</div>
                <div style='color:#8892b0;font-size:.85rem'>person-hours</div>
                <div class='result-sub'>Range: {lo:,.0f} – {hi:,.0f} hrs</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class='result-box'>
                <div class='result-title'>Estimated Duration</div>
                <div class='result-value'>{months:.1f}</div>
                <div style='color:#8892b0;font-size:.85rem'>months</div>
                <div class='result-sub'>At {hours_month:.0f} hrs/month</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""<div class='result-box'>
                <div class='result-title'>Estimated Cost</div>
                <div class='result-value'>${cost:,.0f}</div>
                <div style='color:#8892b0;font-size:.85rem'>USD</div>
                <div class='result-sub'>At ${hourly_rate:.0f}/hr</div>
            </div>""", unsafe_allow_html=True)

        # Effort bar
        fig, ax = plt.subplots(figsize=(9, 1.3))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#0f1117')
        ax.barh([""], [hi - lo], left=[lo], color='#1e3a5f', height=0.5)
        ax.axvline(effort, color='#4fc3f7', linewidth=2.5,
                   label=f'Estimate: {effort:,.0f} hrs')
        ax.tick_params(colors='#8892b0', labelsize=8)
        ax.set_xlabel("Person-hours", color='#8892b0', fontsize=9)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.legend(facecolor='#1e2130', labelcolor='#ccd6f6', fontsize=9)
        st.pyplot(fig); plt.close()

        # Effort category prediction (Desharnais features for classification models)
        st.markdown("<div class='section-hdr'>Effort Category Prediction</div>",
                    unsafe_allow_html=True)
        st.caption("Classification models predict whether this project falls in Low / Medium / High effort category.")

        # Build Desharnais-style input (fill unrelated fields with medians)
        row = {f: 0.0 for f in DESH_FEAT}
        row["TeamExp"]    = team_exp
        row["ManagerExp"] = manager_exp
        row["Length"]     = duration
        row["Transactions"] = transactions
        row["Entities"]   = entities
        row["YearEnd"]    = year_end
        # PointsNonAdjust / PointsAjust / Adjustment — use size as proxy
        if "PointsNonAdjust" in row: row["PointsNonAdjust"] = size
        if "PointsAjust"     in row: row["PointsAjust"]     = size * 1.05
        if "Adjustment"      in row: row["Adjustment"]       = 1.0
        if "Language"        in row: row["Language"]         = list(LANG_FACTOR.keys()).index(language)

        X_cls = np.array([[row[f] for f in DESH_FEAT]])
        X_cls_s = gnb_b["scaler"].transform(X_cls)

        gnb_pred  = le.inverse_transform(gnb_b["model"].predict(X_cls_s))[0]
        log_pred  = le.inverse_transform(log_b["model"].predict(X_cls_s))[0]

        COLOR = {"Low": "#81c784", "Medium": "#ffb74d", "High": "#ef5350"}
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(f"""<div class='result-box' style='border-color:{COLOR.get(gnb_pred,"#4fc3f7")}'>
                <div class='result-title'>Gaussian NB Prediction</div>
                <div class='result-value' style='color:{COLOR.get(gnb_pred,"#fff")}'>{gnb_pred}</div>
                <div class='result-sub'>Effort category</div>
            </div>""", unsafe_allow_html=True)
        with cc2:
            st.markdown(f"""<div class='result-box' style='border-color:{COLOR.get(log_pred,"#4fc3f7")}'>
                <div class='result-title'>Logistic Regression Prediction</div>
                <div class='result-value' style='color:{COLOR.get(log_pred,"#fff")}'>{log_pred}</div>
                <div class='result-sub'>Effort category</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — MLP PERFORMANCE
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-hdr'>MLP Neural Network — Best Model ⭐</div>",
                unsafe_allow_html=True)

    mlp_res = results["MLP Neural Network"]
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{mlp_res['R2']}</div><div class='metric-label'>R² Score</div></div>",
                    unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{mlp_res['RMSE']:,.0f}</div><div class='metric-label'>RMSE (hrs)</div></div>",
                    unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{mlp_res['MAE']:,.0f}</div><div class='metric-label'>MAE (hrs)</div></div>",
                    unsafe_allow_html=True)

    st.info(f"📌 R² = {mlp_res['R2']} means the MLP explains {mlp_res['R2']*100:.1f}% of variance in effort. "
            f"IEEE papers on similar datasets typically report R² of 0.55–0.72.")

    # Actual vs Predicted + Residuals
    mlp_model = mlp_b["model"]
    Xte_mlp   = app_data["Xte_mlp"]
    yte_mlp   = app_data["yte_mlp"]
    yp_mlp    = mlp_model.predict(Xte_mlp)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor('#0f1117')

    ax1 = axes[0]; ax1.set_facecolor('#0f1117')
    ax1.scatter(yte_mlp, yp_mlp, alpha=0.5, color='#4fc3f7', s=22)
    mn, mx = min(yte_mlp.min(), yp_mlp.min()), max(yte_mlp.max(), yp_mlp.max())
    ax1.plot([mn,mx],[mn,mx],'r--',lw=1.5,label='Perfect fit')
    ax1.set_xlabel("Actual Effort (hrs)", color='#8892b0', fontsize=9)
    ax1.set_ylabel("Predicted Effort (hrs)", color='#8892b0', fontsize=9)
    ax1.set_title("Actual vs Predicted — MLP", color='#ccd6f6', fontsize=10, fontweight='bold')
    ax1.tick_params(colors='#8892b0', labelsize=8)
    ax1.legend(facecolor='#1e2130', labelcolor='#ccd6f6', fontsize=8)
    for sp in ax1.spines.values(): sp.set_color('#2e3450')

    ax2 = axes[1]; ax2.set_facecolor('#0f1117')
    residuals = yte_mlp - yp_mlp
    ax2.scatter(yp_mlp, residuals, alpha=0.5, color='#81c784', s=22)
    ax2.axhline(0, color='#ef5350', lw=1.5, linestyle='--')
    ax2.set_xlabel("Predicted Effort (hrs)", color='#8892b0', fontsize=9)
    ax2.set_ylabel("Residual", color='#8892b0', fontsize=9)
    ax2.set_title("Residual Plot — MLP", color='#ccd6f6', fontsize=10, fontweight='bold')
    ax2.tick_params(colors='#8892b0', labelsize=8)
    for sp in ax2.spines.values(): sp.set_color('#2e3450')

    plt.tight_layout(pad=2.0)
    st.pyplot(fig); plt.close()
    st.caption("Residuals scattered randomly around 0 = no systematic bias in the model.")

    # Architecture diagram text
    st.markdown("<div class='section-hdr'>MLP Architecture</div>", unsafe_allow_html=True)
    st.code("Input (3) → Hidden Layer 1 (32 neurons, ReLU) → Hidden Layer 2 (16 neurons, ReLU) → Output (1)", language=None)

# ══════════════════════════════════════════════════════════════
# TAB 3 — ALL MODELS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-hdr'>All 6 Models — Comparison</div>", unsafe_allow_html=True)

    # Regression models table
    st.markdown("**Regression Models** — predict effort in person-hours")
    reg_rows = []
    for name, r in results.items():
        if r["type"] == "Regression":
            reg_rows.append({
                "Model": ("⭐ " if name == "MLP Neural Network" else "") + name,
                "R²":   float(r["R2"]),
                "RMSE": float(r["RMSE"]),
                "MAE":  float(r["MAE"]),
            })
    df_reg = pd.DataFrame(reg_rows).sort_values("R²", ascending=False)
    st.dataframe(df_reg, hide_index=True, width="stretch")

    st.markdown("**Classification Models** — predict effort category (Low / Medium / High)")
    cls_rows = []
    for name, r in results.items():
        if r["type"] == "Classification":
            cls_rows.append({
                "Model":    name,
                "Accuracy": float(r["Accuracy"]),
                "Classes":  "Low / Medium / High",
            })
    df_cls = pd.DataFrame(cls_rows)
    st.dataframe(df_cls, hide_index=True, width="stretch")

    # R² bar chart — regression models only
    st.markdown("<div class='section-hdr'>R² Score — Regression Models</div>", unsafe_allow_html=True)
    reg_names = [n for n,r in results.items() if r["type"]=="Regression"]
    reg_r2    = [results[n]["R2"] for n in reg_names]
    colors    = ['#4fc3f7' if n=="MLP Neural Network" else '#37474f' for n in reg_names]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#0f1117')
    bars = ax.barh(reg_names, reg_r2, color=colors, height=0.45)
    for bar, val in zip(bars, reg_r2):
        ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                f'{val:.4f}', va='center', color='#ccd6f6', fontsize=9)
    ax.set_xlabel("R² Score", color='#8892b0', fontsize=9)
    ax.set_xlim(0, max(reg_r2)+0.12)
    ax.tick_params(colors='#8892b0', labelsize=9)
    for sp in ax.spines.values(): sp.set_visible(False)
    best_patch = mpatches.Patch(color='#4fc3f7', label='MLP — Best Model ⭐')
    other_patch = mpatches.Patch(color='#37474f', label='Other models')
    ax.legend(handles=[best_patch, other_patch], facecolor='#1e2130',
              labelcolor='#ccd6f6', fontsize=8)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Accuracy bar chart — classification models only
    st.markdown("<div class='section-hdr'>Accuracy — Classification Models</div>", unsafe_allow_html=True)
    cls_names = [n for n,r in results.items() if r["type"]=="Classification"]
    cls_acc   = [results[n]["Accuracy"] for n in cls_names]

    fig2, ax2 = plt.subplots(figsize=(6, 2.2))
    fig2.patch.set_facecolor('#0f1117'); ax2.set_facecolor('#0f1117')
    bars2 = ax2.barh(cls_names, cls_acc, color=['#ce93d8','#9fa8da'], height=0.4)
    for bar, val in zip(bars2, cls_acc):
        ax2.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                 f'{val:.4f}', va='center', color='#ccd6f6', fontsize=9)
    ax2.set_xlabel("Accuracy", color='#8892b0', fontsize=9)
    ax2.set_xlim(0, 1.1)
    ax2.tick_params(colors='#8892b0', labelsize=9)
    for sp in ax2.spines.values(): sp.set_visible(False)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    # Feature importance — Random Forest
    st.markdown("<div class='section-hdr'>Feature Importance — Random Forest</div>", unsafe_allow_html=True)
    rf_imp = results["Random Forest"].get("feature_importances", {})
    if rf_imp:
        imp_df = pd.Series(rf_imp).sort_values(ascending=False).head(10)
        fig3, ax3 = plt.subplots(figsize=(8, 3.5))
        fig3.patch.set_facecolor('#0f1117'); ax3.set_facecolor('#0f1117')
        ax3.barh(imp_df.index[::-1], imp_df.values[::-1], color='#4fc3f7', height=0.5)
        ax3.set_xlabel("Importance", color='#8892b0', fontsize=9)
        ax3.tick_params(colors='#8892b0', labelsize=8)
        for sp in ax3.spines.values(): sp.set_visible(False)
        plt.tight_layout(); st.pyplot(fig3); plt.close()
        st.caption("Random Forest feature importance shows which project attributes drive effort the most.")

# ══════════════════════════════════════════════════════════════
# TAB 4 — CHARTS
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-hdr'>Actual vs Predicted — All Regression Models</div>",
                unsafe_allow_html=True)

    Xte_d  = app_data["Xte_desh"]
    yte_d  = app_data["yte_desh"]

    reg_models = {
        "Linear Regression": lr_b["model"],
        "Decision Tree":     dt_b["model"],
        "Random Forest":     rf_b["model"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor('#0f1117')
    clrs = ['#81c784', '#ffb74d', '#ce93d8']

    for ax, (name, model), color in zip(axes, reg_models.items(), clrs):
        yp = model.predict(Xte_d)
        ax.set_facecolor('#0f1117')
        ax.scatter(yte_d, yp, alpha=0.6, color=color, s=22)
        mn, mx = min(yte_d.min(), yp.min()), max(yte_d.max(), yp.max())
        ax.plot([mn,mx],[mn,mx],'r--',lw=1.2)
        r2 = r2_score(yte_d, yp)
        ax.set_title(f"{name}\nR²={r2:.4f}", color='#ccd6f6', fontsize=9, fontweight='bold')
        ax.set_xlabel("Actual", color='#8892b0', fontsize=8)
        ax.set_ylabel("Predicted", color='#8892b0', fontsize=8)
        ax.tick_params(colors='#8892b0', labelsize=7)
        for sp in ax.spines.values(): sp.set_color('#2e3450')

    plt.tight_layout(pad=2.0)
    st.pyplot(fig); plt.close()

    # RMSE comparison
    st.markdown("<div class='section-hdr'>RMSE Comparison — All Regression Models</div>",
                unsafe_allow_html=True)
    all_reg = {n: results[n]["RMSE"] for n in results if results[n]["type"]=="Regression"}
    fig4, ax4 = plt.subplots(figsize=(8, 3.2))
    fig4.patch.set_facecolor('#0f1117'); ax4.set_facecolor('#0f1117')
    clrs4 = ['#4fc3f7' if n=="MLP Neural Network" else '#37474f' for n in all_reg]
    bars4 = ax4.barh(list(all_reg.keys()), list(all_reg.values()), color=clrs4, height=0.45)
    for bar, val in zip(bars4, all_reg.values()):
        ax4.text(bar.get_width()+20, bar.get_y()+bar.get_height()/2,
                 f'{val:,.0f}', va='center', color='#ccd6f6', fontsize=9)
    ax4.set_xlabel("RMSE (person-hours) — lower is better", color='#8892b0', fontsize=9)
    ax4.tick_params(colors='#8892b0', labelsize=9)
    for sp in ax4.spines.values(): sp.set_visible(False)
    plt.tight_layout(); st.pyplot(fig4); plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## About This Project")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Software Effort & Cost Estimation using Machine Learning**

Manual software estimation is error-prone, leading to budget overruns and delays.
This system uses **6 ML models** trained on real project data to provide
data-driven estimates.

### Datasets
| Dataset | Rows | Features |
|---|---|---|
| Desharnais | 81 projects | 10 features |
| Combined | 642 projects | 3 features |

### Models
| Model | Type | Metric |
|---|---|---|
| MLP Neural Network ⭐ | Regression | R² |
| Linear Regression | Regression | R² |
| Decision Tree | Regression | R² |
| Random Forest | Regression | R² |
| Gaussian NB | Classification | Accuracy |
| Logistic Regression | Classification | Accuracy |
        """)
    with c2:
        st.markdown("""
### Tech Stack
`Python` · `Scikit-learn` · `Streamlit`
`Pandas` · `NumPy` · `Matplotlib` · `Joblib`

### Why MLP is Best
The MLP Neural Network is selected as the primary prediction model
because it captures non-linear relationships between project parameters
and effort — something linear models cannot do.

### R² Explained
R² = 0.62–0.70 is consistent with published IEEE and ACM research
on effort estimation datasets. Higher accuracy requires richer
features like team size, complexity, risk factor, and domain type.

### Future Work
- SHAP explainability
- More features (team size, complexity, risk)
- Docker deployment
- CI/CD pipeline integration
        """)

    st.markdown("---")
    st.markdown("<p style='color:#546e7a;font-size:0.8rem'>Final Year Project · CSE Department · KTU · 2025–26</p>",
                unsafe_allow_html=True)