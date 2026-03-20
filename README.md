# ChurnGuard (Telco) — Customer Churn Prediction (ML) | FastAPI + Streamlit

Production-ready churn prediction system for subscription businesses (Telco).  
Includes a FastAPI inference API + Streamlit dashboard with batch CSV scoring, download, and explainability (top risk drivers/reducers).

---

## 🚀 Features
- ✅ **Single Prediction**: churn probability + churn label
- ✅ **Explainability**: shows top factors increasing/decreasing churn risk
- ✅ **Batch Prediction (CSV)**: upload CSV → predict all rows → download results
- ✅ **Reusable ML Pipeline**: preprocessing + model saved via `joblib`

---

## 🧰 Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (Logistic Regression + preprocessing pipeline)
- FastAPI + Uvicorn
- Streamlit

---
## 📁 Project Structure
```text
churnguard-telco/
  backend/
    main.py
  frontend/
    app.py
  ml/
    train.py
  artifacts/
    metrics.json
    (model file generated locally)
  sample_requests/
    single.json
    batch.csv
  data/
    .gitkeep
    (place dataset here locally)
  requirements.txt
  .gitignore
  README.md
📦 Dataset

This project uses the Telco Customer Churn dataset.

Place the dataset CSV here (locally):

data/telco.csv
