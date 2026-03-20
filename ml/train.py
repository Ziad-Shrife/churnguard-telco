import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report

DATA_PATH = "data/telco.csv"
MODEL_PATH = "artifacts/churn_model.joblib"
METRICS_PATH = "artifacts/metrics.json"

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Drop ID column if exists
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric (it may contain spaces)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows with missing target
    df = df.dropna(subset=["Churn"])

    # Fill missing numeric values
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical values
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown")

    # Target to 0/1
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    return df

def main():
    df = load_data()

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = LogisticRegression(max_iter=2000)

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, zero_division=0),
        "num_features": len(num_cols),
        "cat_features": len(cat_cols),
    }

    joblib.dump(pipe, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Saved model:", MODEL_PATH)
    print("✅ Saved metrics:", METRICS_PATH)
    print("ROC-AUC:", metrics["roc_auc"], "| F1:", metrics["f1"])

if __name__ == "__main__":
    main()