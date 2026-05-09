"""
main.py — Software Effort Estimation API
Run: uvicorn main:app --reload
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
import os
from typing import Optional

# ── App setup ────────────────────────────────────────────────
app = FastAPI(
    title="Software Effort Estimation API",
    description="ML-powered software effort, duration and cost estimation using 6 trained models.",
    version="2.0.0"
)

# Allow all origins (needed for GitHub Pages frontend to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models at startup ────────────────────────────────────
MODELS_DIR = "models"

def load_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

try:
    mlp_b  = load_model("mlp_model.joblib")
    lr_b   = load_model("lr_model.joblib")
    dt_b   = load_model("dt_model.joblib")
    rf_b   = load_model("rf_model.joblib")
    gnb_b  = load_model("gnb_model.joblib")
    log_b  = load_model("log_model.joblib")
    data   = load_model("app_data.pkl")
    print("✅ All models loaded successfully")
except FileNotFoundError as e:
    print(f"❌ Error loading models: {e}")
    raise

results     = data["results"]
le          = data["label_encoder"]
DESH_FEAT   = data["desh_features"]
MLP_FEAT    = data["mlp_features"]  # ["Size", "Duration", "Experience"]

LANG_FACTOR = {
    "Python": 1.0, "Java": 1.1, "C++": 1.2,
    "JavaScript": 1.05, "C#": 1.1, "PHP": 1.0,
    "Ruby": 0.95, "Go": 1.05, "Other": 1.0
}

# ── Request / Response models ─────────────────────────────────

class PredictRequest(BaseModel):
    # MLP features
    size:           float = Field(..., gt=0,  le=5000,  description="Function points (size)", example=100)
    duration:       float = Field(..., gt=0,  le=84,    description="Project length in months", example=6)
    team_exp:       float = Field(..., ge=0,  le=9,     description="Team experience in years", example=2)
    # Desharnais features
    manager_exp:    float = Field(..., ge=0,  le=15,    description="Manager experience in years", example=3)
    transactions:   float = Field(..., ge=0,  le=1000,  description="Number of transactions", example=100)
    entities:       float = Field(..., ge=0,  le=500,   description="Number of entities", example=50)
    points_na:      float = Field(..., ge=0,  le=2000,  description="Points non-adjusted", example=100)
    adjustment:     float = Field(..., ge=0.5, le=1.5,  description="Adjustment factor", example=1.0)
    year_end:       float = Field(..., ge=1980, le=2030, description="Year project ends", example=2024)
    # Cost inputs
    hours_per_month: float = Field(..., gt=0, le=300,   description="Hours worked per month", example=160)
    hourly_rate:    float = Field(..., gt=0, le=300,    description="Hourly rate in USD", example=25)
    language:       str   = Field(...,                  description="Programming language", example="Python")

    @validator("language")
    def validate_language(cls, v):
        if v not in LANG_FACTOR:
            raise ValueError(f"Language must be one of: {list(LANG_FACTOR.keys())}")
        return v


class ModelResult(BaseModel):
    effort_hours:  float
    duration_months: float
    cost_usd:      float
    confidence_low:  float
    confidence_high: float


class PredictResponse(BaseModel):
    status:          str
    # MLP prediction (primary)
    mlp:             ModelResult
    # Other regression models
    linear_regression: dict
    decision_tree:   dict
    random_forest:   dict
    # Classification
    effort_category_gnb:      str
    effort_category_logistic: str
    # Model metrics for display
    model_metrics:   dict


# ── Helper: build Desharnais input vector ────────────────────
def build_desh_input(req: PredictRequest):
    row = {f: 0.0 for f in DESH_FEAT}
    row["TeamExp"]    = req.team_exp
    row["ManagerExp"] = req.manager_exp
    row["Length"]     = req.duration
    row["Transactions"] = req.transactions
    row["Entities"]   = req.entities
    row["YearEnd"]    = req.year_end
    if "PointsNonAdjust" in row: row["PointsNonAdjust"] = req.points_na
    if "PointsAjust"     in row: row["PointsAjust"]     = req.points_na * req.adjustment
    if "Adjustment"      in row: row["Adjustment"]       = req.adjustment
    if "Language"        in row:
        lang_list = list(LANG_FACTOR.keys())
        row["Language"] = lang_list.index(req.language) if req.language in lang_list else 0
    return np.array([[row[f] for f in DESH_FEAT]])


# ── Main prediction endpoint ──────────────────────────────────
@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Run all 6 ML models on the given project parameters.
    Returns effort (person-hours), duration (months), cost (USD),
    confidence range, and effort category predictions.
    """
    lang_factor = LANG_FACTOR[req.language]

    # ── MLP prediction ────────────────────────────────────────
    X_mlp   = np.array([[req.size, req.duration, req.team_exp]])
    X_mlp_s = mlp_b["scaler"].transform(X_mlp)
    mlp_raw = float(mlp_b["model"].predict(X_mlp_s)[0])
    mlp_eff = max(mlp_raw * lang_factor, 1.0)
    mlp_months = mlp_eff / req.hours_per_month
    mlp_cost   = mlp_eff * req.hourly_rate

    # ── Desharnais models ─────────────────────────────────────
    X_desh   = build_desh_input(req)
    X_desh_s = lr_b["scaler"].transform(X_desh)

    lr_eff  = max(float(lr_b["model"].predict(X_desh_s)[0])  * lang_factor, 1.0)
    dt_eff  = max(float(dt_b["model"].predict(X_desh_s)[0])  * lang_factor, 1.0)
    rf_eff  = max(float(rf_b["model"].predict(X_desh_s)[0])  * lang_factor, 1.0)

    # ── Classification ────────────────────────────────────────
    X_cls_s = gnb_b["scaler"].transform(X_desh)
    gnb_pred = le.inverse_transform(gnb_b["model"].predict(X_cls_s))[0]
    log_pred = le.inverse_transform(log_b["model"].predict(X_cls_s))[0]

    return PredictResponse(
        status="success",
        mlp=ModelResult(
            effort_hours    = round(mlp_eff, 2),
            duration_months = round(mlp_months, 2),
            cost_usd        = round(mlp_cost, 2),
            confidence_low  = round(mlp_eff * 0.8, 2),
            confidence_high = round(mlp_eff * 1.2, 2),
        ),
        linear_regression={
            "effort_hours":    round(lr_eff, 2),
            "duration_months": round(lr_eff / req.hours_per_month, 2),
            "cost_usd":        round(lr_eff * req.hourly_rate, 2),
        },
        decision_tree={
            "effort_hours":    round(dt_eff, 2),
            "duration_months": round(dt_eff / req.hours_per_month, 2),
            "cost_usd":        round(dt_eff * req.hourly_rate, 2),
        },
        random_forest={
            "effort_hours":    round(rf_eff, 2),
            "duration_months": round(rf_eff / req.hours_per_month, 2),
            "cost_usd":        round(rf_eff * req.hourly_rate, 2),
        },
        effort_category_gnb      = str(gnb_pred),
        effort_category_logistic = str(log_pred),
        model_metrics = {
            name: {
                "R2":   r.get("R2", r.get("Accuracy")),
                "type": r["type"],
            }
            for name, r in results.items()
        }
    )


# ── Models info endpoint ──────────────────────────────────────
@app.get("/models", tags=["Info"])
def get_model_info():
    """Returns performance metrics for all 6 trained models."""
    return {
        "status": "success",
        "models": {
            name: {
                "type":     r["type"],
                "R2":       r.get("R2"),
                "RMSE":     r.get("RMSE"),
                "MAE":      r.get("MAE"),
                "Accuracy": r.get("Accuracy"),
            }
            for name, r in results.items()
        }
    }


# ── Health check endpoint ─────────────────────────────────────
@app.get("/health", tags=["Info"])
def health_check():
    """Check if API is running and all models are loaded."""
    return {
        "status":  "healthy",
        "models_loaded": 6,
        "version": "2.0.0",
        "models": ["MLP", "Linear Regression", "Decision Tree",
                   "Random Forest", "Gaussian NB", "Logistic Regression"]
    }


# ── Root endpoint — serve the website ────────────────────────
@app.get("/", response_class=HTMLResponse, tags=["App"])
def root():
    """Serves the web application."""
    html_path = os.path.join("static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Software Effort Estimation API</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>")


# Mount static files (JS, CSS etc.)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")