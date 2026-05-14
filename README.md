# Software Effort Estimation using Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136.1-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-29.4.2-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)

**A production-ready machine learning system that predicts software development effort, project duration, and cost using 6 trained ML models — deployed as a live REST API with a web frontend.**

[ Live API](https://effort-estimation-api.onrender.com/) · [📖 API Docs](https://effort-estimation-api.onrender.com/docs) · [ Web App](https://abishekbino.github.io/effort-estimation) · [❤️ Health Check](https://effort-estimation-api.onrender.com/health)

</div>

---

## Problem Statement

Accurately estimating software development effort is one of the most challenging tasks in software project management. Manual estimation using expert judgment and traditional methods like COCOMO leads to:

- ❌ Project delays and missed deadlines
- ❌ Budget overruns (70% of projects exceed budget)
- ❌ Poor resource allocation and planning
- ❌ Subjective, inconsistent estimates across teams

This system uses **machine learning trained on real historical project data** to provide objective, automated, data-driven effort estimates.

---

##  Features

-  **6 ML Models** — Linear Regression, Decision Tree, Random Forest, MLP Neural Network, Gaussian Naive Bayes, Logistic Regression
-  **Real-time Predictions** — Sub-500ms response via FastAPI REST API
-  **Complete Comparison** — All models evaluated with R², RMSE, MAE and cross-validation
-  **Dual Task** — Regression (effort in hours) + Classification (Low / Medium / High category)
-  **Live Deployment** — Dockerized and deployed on Render Cloud
-  **Responsive Frontend** — Dark-themed web app hosted on GitHub Pages
-  **Interactive API Docs** — Auto-generated Swagger UI at `/docs`

---

##  System Architecture

```
User (Browser)
      │
      ▼
GitHub Pages (index.html)          ← Static frontend hosting
      │
      │  POST /predict (JSON payload)
      ▼
Render Cloud ─── FastAPI Backend   ← REST API (Dockerized)
      │                │
      │         loads  ▼
      │         .joblib model files ← 6 trained ML models
      │
      ▼
JSON Response → Effort (hrs) · Duration (months) · Cost (USD) · Category
```

---

## 📈 Model Performance

### Regression Models (Predict Effort in Person-Hours)

| Model | Dataset | R² Score | RMSE | MAE |
|-------|---------|----------|------|-----|
| Linear Regression | Desharnais | **0.699** | 1,959 | 1,457 |
| MLP Neural Network  | Combined | **0.646** | 4,916 | 2,456 |
| Random Forest | Desharnais | **0.606** | 2,246 | 1,813 |
| Decision Tree | Desharnais | -0.035 | 3,634 | 2,682 |

### Classification Models (Predict Effort Category)

| Model | Accuracy | Classes |
|-------|----------|---------|
| Logistic Regression | **76.5%** | Low / Medium / High |
| Gaussian Naive Bayes | 52.9% | Low / Medium / High |

>  **Note:** IEEE and ACM research papers on the same Desharnais benchmark dataset report R² values of **0.55–0.72**. Our results are well within the published research range.

---

##  Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML** | Scikit-learn 1.7.2 | Model training and evaluation |
| **Backend** | FastAPI 0.136.1 | REST API framework |
| **Server** | Uvicorn 0.46.0 | ASGI production server |
| **Validation** | Pydantic 2.13.4 | Automatic input validation |
| **Data** | Pandas + NumPy | Preprocessing and feature engineering |
| **Persistence** | Joblib 1.5.2 | Model serialization |
| **Frontend** | HTML5 + CSS3 + JS | Responsive web application |
| **Container** | Docker 29.4.2 | Application containerization |
| **Deployment** | Render Cloud | Backend cloud hosting |
| **Hosting** | GitHub Pages | Frontend static hosting |

---

##  Project Structure

```
effort-estimation/
├── main.py                 # FastAPI backend — REST API with all endpoints
├── train_all.py            # Train all 6 ML models and save .joblib files
├── convert_models.py       # Export model weights to JavaScript format
├── app.py                  # Streamlit app (Phase 1 prototype)
├── Dockerfile              # Docker container configuration
├── requirements.txt        # Python dependencies with pinned versions
├── .dockerignore           # Files excluded from Docker build
│
├── data/
│   ├── desharnais.csv      # 81 software projects (10 features)
│   └── combined_dataset.csv # 642 software projects (3 features)
│
├── models/
│   ├── mlp_model.joblib    # MLP Neural Network + StandardScaler
│   ├── lr_model.joblib     # Linear Regression + scaler
│   ├── dt_model.joblib     # Decision Tree + scaler
│   ├── rf_model.joblib     # Random Forest + scaler + feature importances
│   ├── gnb_model.joblib    # Gaussian NB + scaler + LabelEncoder
│   ├── log_model.joblib    # Logistic Regression + scaler + LabelEncoder
│   └── app_data.pkl        # Test set data + all model metrics
│
└── static/
    ├── index.html          # Complete web application (HTML + CSS + JS)
    └── models.js           # Exported model weights for browser inference
```

---

##  Quick Start

### Option 1 — Use the Live API (No setup required)

```bash
curl -X POST "https://effort-estimation-api.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "size": 100,
    "duration": 6,
    "team_exp": 2,
    "manager_exp": 3,
    "transactions": 100,
    "entities": 50,
    "points_na": 100,
    "adjustment": 1.0,
    "year_end": 2024,
    "hours_per_month": 160,
    "hourly_rate": 25,
    "language": "Python"
  }'
```

### Option 2 — Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/AbishekBino/effort-estimation.git
cd effort-estimation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train all models (generates .joblib files)
python train_all.py

# 4. Start the API server
uvicorn main:app --reload

# 5. Open in browser
# Website:  http://localhost:8000
# API Docs: http://localhost:8000/docs
# Health:   http://localhost:8000/health
```

### Option 3 — Run with Docker

```bash
# Build the Docker image
docker build -t effort-estimation-api .

# Run the container
docker run -p 8000:8000 effort-estimation-api

# Open: http://localhost:8000
```

---

##  API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the web application |
| `GET` | `/health` | API health status check |
| `GET` | `/models` | Returns all model performance metrics |
| `POST` | `/predict` | Run all 6 ML models and get predictions |
| `GET` | `/docs` | Interactive Swagger UI documentation |

### Sample API Response — `POST /predict`

```json
{
  "status": "success",
  "mlp": {
    "effort_hours": 1808.33,
    "duration_months": 11.3,
    "cost_usd": 45208.32,
    "confidence_low": 1446.67,
    "confidence_high": 2170.0
  },
  "linear_regression": {
    "effort_hours": 2145.0,
    "duration_months": 13.4,
    "cost_usd": 53625.0
  },
  "decision_tree": {
    "effort_hours": 1224.0,
    "duration_months": 7.65,
    "cost_usd": 30600.0
  },
  "random_forest": {
    "effort_hours": 2013.81,
    "duration_months": 12.59,
    "cost_usd": 50345.12
  },
  "effort_category_gnb": "Medium",
  "effort_category_logistic": "Low"
}
```

---

##  ML Pipeline

```
Raw Datasets (CSV)
        │
        ▼
Data Cleaning
├── Remove Experience < 0 (invalid rows)
├── Drop NaN values (dropna)
└── Cap top 1% outliers (Size, Effort)
        │
        ▼
Feature Engineering
├── Log_Size = log1p(Size)
├── Log_Duration = log1p(Duration)
├── Size_x_Duration = Size × Duration
├── Size_x_Experience = Size × Experience
└── One-hot encoding (Language column)
        │
        ▼
Train-Test Split (80% / 20%, random_state=42)
        │
        ▼
StandardScaler (fit on train only → prevent data leakage)
        │
        ▼
Train 6 Models + 5-Fold Cross Validation
        │
        ▼
Save with Joblib (.joblib files)
        │
        ▼
FastAPI serves predictions via REST API
```

---

##  Docker Build Output

```
[+] Building 3.8s (14/14) FINISHED
✅ All models loaded successfully
INFO: Uvicorn running on http://0.0.0.0:8000
INFO: Application startup complete.
```

---

##  Datasets

### Desharnais Dataset
- **Source:** J. Desharnais, University of Montreal, 1988
- **Projects:** 81 real software projects
- **Features:** TeamExp, ManagerExp, YearEnd, Length, Transactions, Entities, PointsNonAdjust, Adjustment, PointsAjust, Language
- **Target:** Effort (person-hours)
- **Used for:** Linear Regression, Decision Tree, Random Forest, Gaussian NB, Logistic Regression

### Combined Dataset
- **Source:** Aggregated from Desharnais, Maxwell, and other published collections
- **Projects:** 642 software projects
- **Features:** Size (function points), Duration (months), Experience (years)
- **Target:** Effort (person-hours)
- **Used for:** MLP Neural Network

---

##  Future Improvements

- [ ] **SHAP Explainability** — Show which features contributed most to each prediction
- [ ] **XGBoost + Optuna** — Hyperparameter optimization to push R² above 0.80
- [ ] **CI/CD Pipeline** — GitHub Actions for automatic retraining on new data
- [ ] **User Authentication** — Project managers save and track estimates
- [ ] **Richer Features** — Team size, complexity score, risk factor, domain type
- [ ] **Real-time Data Collection** — Feedback form to continuously grow the dataset

---

##  Team

| Name | Role |
|------|------|
| **Abishek Bino** | ML Engineering, Backend, Deployment |
| **Adarsh YL** | Data Collection, Model Evaluation |
| **Abhinand SS** | Frontend Development, Testing |
| **Aswin Jose** | Documentation, Analysis |

**Project Guide:** Prof. Priya Shekhar
**Institution:** Lourdes Matha College of Science and Technology (LMCST)
**Program:** B.Tech Computer Science Engineering — KTU (2025–26)

---

##  References

1. K. Srinivasan and D. Fisher, *"Machine Learning Approaches to Software Cost Estimation,"* IEEE Transactions on Software Engineering, vol. 21, no. 2, pp. 126–137, 2022.
2. M. Shepperd and C. Schofield, *"Estimating Software Project Effort Using Analogies,"* IEEE Transactions on Software Engineering, vol. 23, no. 11, pp. 736–743, 2023.
3. F. Pedregosa et al., *"Scikit-learn: Machine learning in Python,"* Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.
4. J. Desharnais, *"Analyse statistique de la productivité des projets informatique,"* Master's thesis, University of Montreal, 1988.

---

##  License

This project is developed as a Final Year B.Tech project for academic purposes.

---

<div align="center">

** If this project helped you, please give it a star!**

Made with by Abishek Bino and team · LMCST · KTU · 2025–26

</div>
