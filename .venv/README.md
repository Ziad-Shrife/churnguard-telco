# ChurnGuard (Telco) — Customer Churn Prediction (ML) | FastAPI + Streamlit

An end-to-end Machine Learning project to **predict customer churn** for a telecom company.

Includes:
- ✅ **FastAPI** inference API (`/predict`, `/batch_predict_csv`)
- ✅ **Streamlit Dashboard** (single prediction + batch CSV + download)
- ✅ **Explainability** (top factors increasing/decreasing churn risk)

---

## 🚀 Features
- **Single Prediction**: churn probability + churn label
- **Explainability**: top risk drivers & reducers (human-friendly)
- **Batch Prediction (CSV)**: upload CSV → predict all rows → download results
- **Reusable ML Pipeline**: preprocessing + model saved via `joblib`

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

  ---

## 📦 Dataset
This project uses the **Telco Customer Churn** dataset.

Place the dataset CSV here (locally):
```text
data/telco.csv
