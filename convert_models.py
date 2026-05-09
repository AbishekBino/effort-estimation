"""
convert_models.py
Run this ONCE in your project folder:
    python convert_models.py

It reads your .joblib files and creates models.js
which the HTML website uses to run real ML predictions in the browser.
"""
import joblib, json, numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR2

def native(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.float32,np.float64,float)): return float(obj)
    if isinstance(obj, (np.int32,np.int64,int)): return int(obj)
    if isinstance(obj, dict): return {k:native(v) for k,v in obj.items()}
    if isinstance(obj, list): return [native(i) for i in obj]
    return obj

print("Loading models...")
mlp_b = joblib.load("mlp_model.joblib")
lr_b  = joblib.load("lr_model.joblib")
dt_b  = joblib.load("dt_model.joblib")
rf_b  = joblib.load("rf_model.joblib")
gnb_b = joblib.load("gnb_model.joblib")
log_b = joblib.load("log_model.joblib")
data  = joblib.load("app_data.pkl")
results = data["results"]
le      = data["label_encoder"]

# MLP — extract real weights
mlp, sc = mlp_b["model"], mlp_b["scaler"]
mlp_data = {
    "scaler_mean": sc.mean_.tolist(),
    "scaler_std":  sc.scale_.tolist(),
    "weights":     [w.tolist() for w in mlp.coefs_],
    "biases":      [b.tolist() for b in mlp.intercepts_],
    "features":    mlp_b["features"],
}
print(f"  MLP: {len(mlp.coefs_)} layers")

# Linear Regression — exact weights
lr, sc_lr = lr_b["model"], lr_b["scaler"]
lr_data = {
    "scaler_mean": sc_lr.mean_.tolist(),
    "scaler_std":  sc_lr.scale_.tolist(),
    "coef":        lr.coef_.tolist(),
    "intercept":   float(lr.intercept_),
    "features":    lr_b["features"],
}
print(f"  Linear Regression: {len(lr.coef_)} coefficients")

# For RF and DT — fit linear proxy on training predictions
df = pd.read_csv("desharnais.csv").dropna(subset=["Effort"])
drop_ids = [c for c in df.columns if any(t in c.lower() for t in ("id","project"))]
X_raw = df.drop(columns=["Effort"]+drop_ids, errors="ignore")
y_reg = df["Effort"].astype(float)
X_raw = pd.get_dummies(X_raw, drop_first=True)
X_raw = X_raw.select_dtypes(include=[np.number]).fillna(X_raw.median())
Xtr, _, ytr, _ = train_test_split(X_raw, y_reg, test_size=0.2, random_state=42)
Xtr_s = sc_lr.transform(Xtr)

# Random Forest proxy
rf = rf_b["model"]
rf_train_preds = rf.predict(Xtr_s)
proxy_rf = LR2().fit(Xtr_s, rf_train_preds)
rf_data = {
    "scaler_mean":         sc_lr.mean_.tolist(),
    "scaler_std":          sc_lr.scale_.tolist(),
    "features":            rf_b["features"],
    "feature_importances": rf.feature_importances_.tolist(),
    "proxy_coef":          proxy_rf.coef_.tolist(),
    "proxy_intercept":     float(proxy_rf.intercept_),
}
print(f"  Random Forest: proxy fitted")

# Decision Tree proxy
dt = dt_b["model"]
dt_train_preds = dt.predict(Xtr_s)
proxy_dt = LR2().fit(Xtr_s, dt_train_preds)
dt_data = {
    "scaler_mean":         sc_lr.mean_.tolist(),
    "scaler_std":          sc_lr.scale_.tolist(),
    "features":            dt_b["features"],
    "feature_importances": dt.feature_importances_.tolist(),
    "proxy_coef":          proxy_dt.coef_.tolist(),
    "proxy_intercept":     float(proxy_dt.intercept_),
}
print(f"  Decision Tree: proxy fitted")

# Gaussian NB — exact parameters
gnb, sc_gnb = gnb_b["model"], gnb_b["scaler"]
gnb_data = {
    "scaler_mean": sc_gnb.mean_.tolist(),
    "scaler_std":  sc_gnb.scale_.tolist(),
    "features":    gnb_b["features"],
    "class_prior": gnb.class_prior_.tolist(),
    "theta":       gnb.theta_.tolist(),
    "var":         gnb.var_.tolist(),
    "classes":     le.classes_.tolist(),
}
print(f"  Gaussian NB: {len(gnb.classes_)} classes")

# Logistic Regression — exact weights
log, sc_log = log_b["model"], log_b["scaler"]
log_data = {
    "scaler_mean": sc_log.mean_.tolist(),
    "scaler_std":  sc_log.scale_.tolist(),
    "features":    log_b["features"],
    "coef":        log.coef_.tolist(),
    "intercept":   log.intercept_.tolist(),
    "classes":     le.classes_.tolist(),
}
print(f"  Logistic Regression: {len(log.classes_)} classes")

# Metrics
metrics = native(results)

all_models = {
    "mlp": mlp_data, "lr": lr_data, "rf": rf_data,
    "dt": dt_data, "gnb": gnb_data, "logistic": log_data,
    "metrics": metrics,
}

js = f"const MODELS = {json.dumps(all_models, indent=2)};"
with open("models.js", "w") as f:
    f.write(js)

size_kb = len(js) // 1024
print(f"\n✅ models.js created ({size_kb} KB)")
print("   Now copy models.js and index.html to your GitHub repo and push!")