import streamlit as st
import requests
import json
import pandas as pd
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="ChurnGuard (Telco)", layout="wide")
st.title("📉 ChurnGuard (Telco) — Churn Prediction Dashboard")

st.caption("Enter customer details and get churn probability. Model served via FastAPI.")

# ----------------------------
# Health check
# ----------------------------
with st.sidebar:
    st.subheader("API Status")
    if st.button("Check API"):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            st.success(r.json())
        except Exception as e:
            st.error(f"API not reachable: {e}")

# ----------------------------
# Input form
# ----------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Customer Inputs")

    gender = st.selectbox("gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("SeniorCitizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])

    tenure = st.number_input("tenure (months)", min_value=0, max_value=100, value=5)

    PhoneService = st.selectbox("PhoneService", ["Yes", "No"])
    MultipleLines = st.selectbox("MultipleLines", ["No", "Yes", "No phone service"])

    InternetService = st.selectbox("InternetService", ["Fiber optic", "DSL", "None"])
    OnlineSecurity = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])

    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "PaymentMethod",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

    MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0, max_value=300.0, value=89.9)
    TotalCharges = st.number_input("TotalCharges", min_value=0.0, max_value=20000.0, value=300.5)

    payload = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    with st.expander("Payload preview (for debugging)"):
        st.code(json.dumps(payload, indent=2), language="json")

    predict_btn = st.button("Predict Churn")

with right:
    st.subheader("Prediction Output")

    if predict_btn:
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=20)
            r.raise_for_status()
            out = r.json()
            st.write(out)
            #st.stop()
            proba = out["churn_probability"]
            label = out["churn_label"]

            # -------- Explainability --------
            factors = out.get("top_factors", [])

            if factors:
                st.markdown("### 🔍 Why this prediction? (Explainability)")

                # Split to positive/negative contributions
                pos = [f for f in factors if f.get("contribution", 0) > 0]
                neg = [f for f in factors if f.get("contribution", 0) < 0]

                # Sort by strength
                pos = sorted(pos, key=lambda x: abs(x.get("contribution", 0)), reverse=True)
                neg = sorted(neg, key=lambda x: abs(x.get("contribution", 0)), reverse=True)
        # ✅ Summary for the client (before the two columns)
                top_pos = ", ".join([p.get("feature", "") for p in pos[:3]]) if pos else "N/A"
                top_neg = ", ".join([n.get("feature", "") for n in neg[:3]]) if neg else "N/A"
                st.info(f"Top risk drivers: {top_pos} | Risk reducers: {top_neg}")
                st.info(f"أهم عوامل الخطر: {top_pos} | عوامل تقلل الخطر: {top_neg}")
                
                colA, colB = st.columns(2, gap="large")

                with colA:
                    st.subheader("⬆️ Factors increasing churn risk")
                    if pos:
                        for item in pos[:5]:
                            feat = item.get("feature", "unknown")
                            val = float(item.get("contribution", 0))
                            st.write(f"**{feat}**")
                            st.progress(min(abs(val) / 2.0, 1.0))
                            st.caption(f"Impact strength: {abs(val):.3f}")
                    else:
                        st.info("No strong positive factors found.")

                with colB:
                    st.subheader("⬇️ Factors decreasing churn risk")
                    if neg:
                        for item in neg[:5]:
                            feat = item.get("feature", "unknown")
                            val = float(item.get("contribution", 0))
                            st.write(f"**{feat}**")
                            st.progress(min(abs(val) / 2.0, 1.0))
                            st.caption(f"Impact strength: {abs(val):.3f}")
                    else:
                        st.info("No strong negative factors found.")

                st.caption("Positive contributions increase churn risk; negative contributions decrease it.")

            else:
                st.info("Explainability is not available (API did not return top_factors).")

            # -------- Prediction Metrics --------
            st.metric("Churn Probability", f"{proba:.3f}")
            st.metric("Churn Label (1=Churn)", label)

            if label == 1:
                st.warning("High churn risk. Consider retention actions.")
            else:
                st.success("Low churn risk.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
st.subheader("Batch Prediction (CSV)")
st.caption("Upload a CSV containing the same feature columns (no Churn column).")

csv_file = st.file_uploader("Upload CSV", type=["csv"])
if st.button("Run Batch Prediction") and csv_file is not None:
    try:
        files = {"file": (csv_file.name, csv_file.getvalue(), "text/csv")}
        r = requests.post(f"{API_URL}/batch_predict_csv", files=files, timeout=60)
        r.raise_for_status()
        data = r.json()
        st.success(f"Processed {data['count']} rows")
        st.json(data["results"][:10])  # show first 10
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")