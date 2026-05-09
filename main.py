"""
main.py — Software Effort Estimation API (Fixed)
Run: uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
from typing import Dict, Any

# ── App setup ──
app = FastAPI(title="Software Effort Estimation API v2.1", version="2.1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Load models at startup ──
MODELS_DIR = "models"

def load_model_bundle(filename: str) -> Dict[str, Any]:
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Model missing: {path}")
    bundle = joblib.load(path)
    print(f"✅ Loaded {filename}: {bundle.get('model', 'N/A').__class__.__name__}")
    return bundle

# Load all model bundles
models = {
    "mlp": load_model_bundle("mlp_model.joblib"),
    "lr": load_model_bundle("lr_model.joblib"),
    "dt": load_model_bundle("dt_model.joblib"),
    "rf": load_model_bundle("rf_model.joblib"),
    "gnb": load_model_bundle("gnb_model.joblib"),
    "logistic": load_model_bundle("log_model.joblib")
}

# Load metadata
data_bundle = load_model_bundle("app_data.pkl")
results = data_bundle["results"]
le = data_bundle["label_encoder"]
DESH_FEAT = data_bundle["desh_features"]  # Fixed: use consistent feature order
MLP_FEAT = data_bundle["mlp_features"]

LANG_FACTOR = {
    "Python": 1.0, "Java": 1.1, "C++": 1.2, "JavaScript": 1.05,
    "C#": 1.1, "PHP": 1.0, "Ruby": 0.95, "Go": 1.05, "Other": 1.0
}

# ── Pydantic models ──
class PredictRequest(BaseModel):
    size: float = Field(..., gt=0, le=5000)
    duration: float = Field(..., gt=0, le=84)
    team_exp: float = Field(..., ge=0, le=9)
    manager_exp: float = Field(..., ge=0, le=15)
    transactions: float = Field(..., ge=0, le=1000)
    entities: float = Field(..., ge=0, le=500)
    points_na: float = Field(..., ge=0, le=2000)
    adjustment: float = Field(..., ge=0.5, le=1.5)
    year_end: float = Field(..., ge=1980, le=2030)
    hours_per_month: float = Field(..., gt=0, le=300)
    hourly_rate: float = Field(..., gt=0, le=300)
    language: str = Field(...)

# ── FIXED: Consistent Desharnais feature builder ──
def build_desh_features(req: PredictRequest) -> np.ndarray:
    """Build EXACT feature order used during training for Desharnais models"""
    # Create dict with ALL expected features, default to 0
    feature_dict = {feat: 0.0 for feat in DESH_FEAT}
    
    # Map request fields to feature names (case-insensitive match)
    feature_dict["TeamExp"] = req.team_exp
    feature_dict["ManagerExp"] = req.manager_exp
    feature_dict["Length"] = req.duration
    feature_dict["Transactions"] = req.transactions
    feature_dict["Entities"] = req.entities
    feature_dict["YearEnd"] = req.year_end
    feature_dict["PointsNonAdjust"] = req.points_na
    feature_dict["PointsAjust"] = req.points_na * req.adjustment
    feature_dict["Adjustment"] = req.adjustment
    
    # Language encoding (exact match to training)
    if "Language" in feature_dict:
        lang_idx = list(LANG_FACTOR.keys()).index(req.language) if req.language in LANG_FACTOR else 0
        feature_dict["Language"] = lang_idx
    
    # Convert to exact order array
    X_desh = np.array([[feature_dict[feat] for feat in DESH_FEAT]])
    print(f"Desharnais input shape: {X_desh.shape}, sample: {X_desh[0][:5]}...")  # Debug
    return X_desh

# ── FIXED: MLP-specific features ──
def build_mlp_features(req: PredictRequest) -> np.ndarray:
    """MLP expects only [size, duration, team_exp]"""
    X_mlp = np.array([[req.size, req.duration, req.team_exp]])
    print(f"MLP input shape: {X_mlp.shape}")  # Debug
    return X_mlp

# ── Main prediction endpoint ──
@app.post("/predict")
def predict_effort(req: PredictRequest):
    try:
        lang_factor = LANG_FACTOR.get(req.language, 1.0)
        
        # === MLP Prediction ===
        X_mlp = build_mlp_features(req)
        X_mlp_scaled = models["mlp"]["scaler"].transform(X_mlp)
        mlp_raw = float(models["mlp"]["model"].predict(X_mlp_scaled)[0])
        mlp_effort = max(mlp_raw * lang_factor, 1.0)
        mlp_months = mlp_effort / req.hours_per_month
        mlp_cost = mlp_effort * req.hourly_rate
        
        # === Desharnais Models (lr, dt, rf) ===
        X_desh = build_desh_features(req)
        X_desh_scaled = models["lr"]["scaler"].transform(X_desh)  # Use LR scaler for all Desh models
        
        lr_effort = max(float(models["lr"]["model"].predict(X_desh_scaled)[0]) * lang_factor, 1.0)
        dt_effort = max(float(models["dt"]["model"].predict(X_desh_scaled)[0]) * lang_factor, 1.0)
        rf_effort = max(float(models["rf"]["model"].predict(X_desh_scaled)[0]) * lang_factor, 1.0)
        
        # === Classification ===
        X_cls_scaled = models["gnb"]["scaler"].transform(X_desh)
        gnb_pred = le.inverse_transform(models["gnb"]["model"].predict(X_cls_scaled))[0]
        log_pred = le.inverse_transform(models["logistic"]["model"].predict(X_cls_scaled))[0]
        
        return {
            "status": "success",
            "mlp": {
                "effort_hours": round(mlp_effort, 2),
                "duration_months": round(mlp_months, 2),
                "cost_usd": round(mlp_cost, 2),
                "confidence_low": round(mlp_effort * 0.8, 2),
                "confidence_high": round(mlp_effort * 1.2, 2)
            },
            "linear_regression": {
                "effort_hours": round(lr_effort, 2),
                "duration_months": round(lr_effort / req.hours_per_month, 2),
                "cost_usd": round(lr_effort * req.hourly_rate, 2)
            },
            "decision_tree": {
                "effort_hours": round(dt_effort, 2),
                "duration_months": round(dt_effort / req.hours_per_month, 2),
                "cost_usd": round(dt_effort * req.hourly_rate, 2)
            },
            "random_forest": {
                "effort_hours": round(rf_effort, 2),
                "duration_months": round(rf_effort / req.hours_per_month, 2),
                "cost_usd": round(rf_effort * req.hourly_rate, 2)
            },
            "effort_category_gnb": str(gnb_pred),
            "effort_category_logistic": str(log_pred),
            "model_metrics": {
                name: {"R2": r.get("R2", r.get("Accuracy")), "type": r["type"]}
                for name, r in results.items()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ── Health + info endpoints (unchanged) ──
@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": 6, "version": "2.1.0-fixed"}

@app.get("/models")
def models_info():
    return {"status": "success", "models": {name: {"type": r["type"], "R2": r.get("R2")} for name, r in results.items()}}

@app.get("/")
def root():
    return {"status": "success", "message": "API ready", "docs": "/docs"}