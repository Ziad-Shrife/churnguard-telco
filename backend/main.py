import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

MODEL_PATH = "artifacts/churn_model.joblib"

app = FastAPI(title="ChurnGuard Telco API")

# عشان تقدر تكلم الـAPI من أي UI بسهولة (Streamlit/Frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل الموديل مرة واحدة عند تشغيل السيرفر
pipe = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

def explain_logreg(pipe, X_df, top_n=8):
    preprocess = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    Xt = preprocess.transform(X_df)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)

    coef = model.coef_[0]
    contrib = Xt[0] * coef  # contribution in log-odds

    # feature names from ColumnTransformer
    names = preprocess.get_feature_names_out()

    # sort by absolute impact
    idx = np.argsort(np.abs(contrib))[::-1][:top_n]

    def clean_name(n: str) -> str:
        # num__tenure -> tenure
        if n.startswith("num__"):
            return n.replace("num__", "")
        # cat__Contract_Month-to-month -> Contract=Month-to-month
        if n.startswith("cat__"):
            x = n.replace("cat__", "")
            if "_" in x:
                feat, val = x.split("_", 1)
                return f"{feat}={val}"
            return x
        return n

    factors = []
    for i in idx:
        factors.append({
            "feature": clean_name(str(names[i])),
            "contribution": float(contrib[i])
        })
    return factors
@app.post("/predict")
async def predict(payload: dict):
    X = pd.DataFrame([payload])
    proba = float(pipe.predict_proba(X)[:, 1][0])
    label = int(proba >= 0.5)

    top_factors = explain_logreg(pipe, X, top_n=8)

    return {
        "churn_probability": proba,
        "churn_label": label,
        "top_factors": top_factors
    }

@app.post("/batch_predict_csv")
async def batch_predict_csv(file: UploadFile = File(...)):
    """
    ارفع CSV فيه أعمدة الداتا (بدون Churn) وهيرجع تنبؤ لكل صف
    """
    df = pd.read_csv(file.file)

    # لو لسه customerID موجود شيله
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    probas = pipe.predict_proba(df)[:, 1]
    labels = (probas >= 0.5).astype(int)

    return {
        "count": len(df),
        "results": [
            {"churn_probability": float(p), "churn_label": int(l)}
            for p, l in zip(probas, labels)
        ]
    }