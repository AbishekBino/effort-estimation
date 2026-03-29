"""
train_all.py  —  Trains all 6 models and saves everything for the Streamlit app.
Run this ONCE before launching the app.

Datasets needed in the same folder:
  - desharnais.csv       (for LR, RF, DT, GNB, Logistic)
  - combined_dataset.csv (for MLP)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, accuracy_score, classification_report)
import joblib
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  Software Effort Estimation — Training All Models")
print("=" * 60)

# ═══════════════════════════════════════════════════════════
# DATASET 1 — Desharnais  (LR, RF, DT, GNB, Logistic)
# ═══════════════════════════════════════════════════════════
print("\n📂 Loading desharnais.csv...")
df = pd.read_csv("desharnais.csv")
df = df.dropna(subset=["Effort"]).copy()

# Remove id/project columns
drop_ids = [c for c in df.columns
            if any(t in c.lower() for t in ("id", "project")) and c != "Effort"]
X_raw = df.drop(columns=["Effort"] + drop_ids, errors="ignore")
y_reg = df["Effort"].astype(float)

# Encode categoricals
X_raw = pd.get_dummies(X_raw, drop_first=True)
X_raw = X_raw.select_dtypes(include=[np.number]).fillna(X_raw.median())
DESH_FEATURES = X_raw.columns.tolist()

print(f"   Rows: {len(df)}  |  Features: {DESH_FEATURES}")

# Classification target (Low / Medium / High)
def effort_class(x):
    if x < 1000:   return "Low"
    elif x < 3000: return "Medium"
    else:          return "High"

y_cls = y_reg.apply(effort_class)
le = LabelEncoder()
y_cls_enc = le.fit_transform(y_cls)

# Train/test split
Xtr, Xte, ytr_r, yte_r = train_test_split(X_raw, y_reg,     test_size=0.2, random_state=42)
_,   _,  ytr_c, yte_c  = train_test_split(X_raw, y_cls_enc, test_size=0.2, random_state=42)

scaler_d = StandardScaler()
Xtr_s = scaler_d.fit_transform(Xtr)
Xte_s = scaler_d.transform(Xte)

results = {}   # store metrics for app

# ── Linear Regression ────────────────────────────────────────
print("\n🔵 Linear Regression...")
lr = LinearRegression()
lr.fit(Xtr_s, ytr_r)
yp = lr.predict(Xte_s)
results["Linear Regression"] = {
    "type":  "Regression",
    "R2":    round(r2_score(yte_r, yp), 4),
    "RMSE":  round(np.sqrt(mean_squared_error(yte_r, yp)), 2),
    "MAE":   round(mean_absolute_error(yte_r, yp), 2),
}
print(f"   R²={results['Linear Regression']['R2']}  RMSE={results['Linear Regression']['RMSE']}")
joblib.dump({"model": lr, "scaler": scaler_d, "features": DESH_FEATURES}, "lr_model.joblib")

# ── Decision Tree ────────────────────────────────────────────
print("\n🔵 Decision Tree...")
dt = DecisionTreeRegressor(max_depth=6, min_samples_leaf=3, random_state=42)
dt.fit(Xtr_s, ytr_r)
yp = dt.predict(Xte_s)
results["Decision Tree"] = {
    "type":  "Regression",
    "R2":    round(r2_score(yte_r, yp), 4),
    "RMSE":  round(np.sqrt(mean_squared_error(yte_r, yp)), 2),
    "MAE":   round(mean_absolute_error(yte_r, yp), 2),
    "feature_importances": dict(zip(DESH_FEATURES, dt.feature_importances_)),
}
print(f"   R²={results['Decision Tree']['R2']}  RMSE={results['Decision Tree']['RMSE']}")
joblib.dump({"model": dt, "scaler": scaler_d, "features": DESH_FEATURES}, "dt_model.joblib")

# ── Random Forest ────────────────────────────────────────────
print("\n🔵 Random Forest...")
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(Xtr_s, ytr_r)
yp = rf.predict(Xte_s)
results["Random Forest"] = {
    "type":  "Regression",
    "R2":    round(r2_score(yte_r, yp), 4),
    "RMSE":  round(np.sqrt(mean_squared_error(yte_r, yp)), 2),
    "MAE":   round(mean_absolute_error(yte_r, yp), 2),
    "feature_importances": dict(zip(DESH_FEATURES, rf.feature_importances_)),
}
print(f"   R²={results['Random Forest']['R2']}  RMSE={results['Random Forest']['RMSE']}")
joblib.dump({"model": rf, "scaler": scaler_d, "features": DESH_FEATURES}, "rf_model.joblib")

# ── Gaussian NB (Classification) ─────────────────────────────
print("\n🔵 Gaussian NB (Classification)...")
gnb = GaussianNB()
gnb.fit(Xtr_s, ytr_c)
yp_c = gnb.predict(Xte_s)
results["Gaussian NB"] = {
    "type":     "Classification",
    "Accuracy": round(accuracy_score(yte_c, yp_c), 4),
    "classes":  le.classes_.tolist(),
    "report":   classification_report(yte_c, yp_c,
                    target_names=le.classes_, output_dict=True),
}
print(f"   Accuracy={results['Gaussian NB']['Accuracy']}")
joblib.dump({"model": gnb, "scaler": scaler_d,
             "features": DESH_FEATURES, "label_encoder": le}, "gnb_model.joblib")

# ── Logistic Regression (Classification) ─────────────────────
print("\n🔵 Logistic Regression (Classification)...")
log = LogisticRegression(max_iter=2000, random_state=42)
log.fit(Xtr_s, ytr_c)
yp_c = log.predict(Xte_s)
results["Logistic Regression"] = {
    "type":     "Classification",
    "Accuracy": round(accuracy_score(yte_c, yp_c), 4),
    "classes":  le.classes_.tolist(),
    "report":   classification_report(yte_c, yp_c,
                    target_names=le.classes_, output_dict=True),
}
print(f"   Accuracy={results['Logistic Regression']['Accuracy']}")
joblib.dump({"model": log, "scaler": scaler_d,
             "features": DESH_FEATURES, "label_encoder": le}, "log_model.joblib")

# ═══════════════════════════════════════════════════════════
# DATASET 2 — Combined  (MLP — Best Model)
# ═══════════════════════════════════════════════════════════
print("\n📂 Loading combined_dataset.csv...")
df2 = pd.read_csv("combined_dataset.csv")
df2 = df2[df2["Experience"] >= 0].dropna()

X2 = df2[["Size", "Duration", "Experience"]]
y2 = df2["Effort"]
MLP_FEATURES = ["Size", "Duration", "Experience"]

Xtr2, Xte2, ytr2, yte2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
scaler_m = StandardScaler()
Xtr2_s = scaler_m.fit_transform(Xtr2)
Xte2_s = scaler_m.transform(Xte2)

print("\n🔵 MLP Neural Network (Best Model)...")
mlp = MLPRegressor(
    hidden_layer_sizes=(32, 16), activation='relu',
    solver='adam', alpha=0.001, batch_size=16,
    learning_rate='adaptive', max_iter=1000,
    early_stopping=True, validation_fraction=0.2,
    n_iter_no_change=10, random_state=42
)
mlp.fit(Xtr2_s, ytr2)
yp2 = mlp.predict(Xte2_s)
results["MLP Neural Network"] = {
    "type":  "Regression",
    "R2":    round(r2_score(yte2, yp2), 4),
    "RMSE":  round(np.sqrt(mean_squared_error(yte2, yp2)), 2),
    "MAE":   round(mean_absolute_error(yte2, yp2), 2),
    "note":  "Best Model — trained on combined dataset",
}
print(f"   R²={results['MLP Neural Network']['R2']}  RMSE={results['MLP Neural Network']['RMSE']}")
joblib.dump({"model": mlp, "scaler": scaler_m,
             "features": MLP_FEATURES}, "mlp_model.joblib")

# ═══════════════════════════════════════════════════════════
# Save test data + results for Streamlit app
# ═══════════════════════════════════════════════════════════
joblib.dump({
    # Desharnais test set (for regression charts)
    "Xte_desh":     Xte_s,
    "yte_desh":     yte_r.values,
    "Xte_desh_raw": Xte.values,
    "desh_features": DESH_FEATURES,
    # MLP test set
    "Xte_mlp":      Xte2_s,
    "yte_mlp":      yte2.values,
    "mlp_features": MLP_FEATURES,
    # All results
    "results":      results,
    "label_encoder": le,
}, "app_data.pkl")

# ── Final summary ────────────────────────────────────────────
print("\n" + "=" * 60)
print("  📋 FINAL RESULTS SUMMARY")
print("=" * 60)
for name, r in results.items():
    if r["type"] == "Regression":
        star = " ⭐ BEST" if name == "MLP Neural Network" else ""
        print(f"  {name:<22} R²={r['R2']:.4f}  RMSE={r['RMSE']:,.1f}{star}")
    else:
        print(f"  {name:<22} Accuracy={r['Accuracy']:.4f}  (Classification)")
print("=" * 60)
print("\n💾 Saved: lr_model.joblib, dt_model.joblib, rf_model.joblib")
print("💾 Saved: gnb_model.joblib, log_model.joblib, mlp_model.joblib")
print("💾 Saved: app_data.pkl")
print("\n✅ All done! Now run:  streamlit run app.py")