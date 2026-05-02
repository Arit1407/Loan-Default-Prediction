from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

from src.data.preprocessing import preprocess
from src.utils.helpers import select_final_features
from src.config.features import FINAL_FEATURES

app = FastAPI()

# -----------------------------
# LOAD MODEL + ARTIFACTS
# -----------------------------
model = joblib.load("models/model.pkl")

# 🔥 NEW (IMPORTANT)
train_cols = joblib.load("models/columns.pkl")
threshold = joblib.load("models/threshold.pkl")


# -----------------------------
# REQUIRED RAW COLUMNS
# -----------------------------
REQUIRED_COLUMNS = [
    "Client_Income","Credit_Amount","Loan_Annuity",
    "Population_Region_Relative","Age_Days",
    "Registration_Days","ID_Days","Score_Source_3",
    "Score_Source_1","Score_Source_2",
    "Credit_Bureau","Social_Circle_Default",
    "Client_Occupation","Bike_Owned",
    "Client_Gender","Client_Marital_Status",
    "Client_Education","Client_Housing_Type",
    "Loan_Contract_Type","Client_Income_Type",
    "Accompany_Client","Car_Owned","Employed_Days",
    "Type_Organization",
    "Client_Permanent_Match_Tag",
    "Client_Contact_Work_Tag",
    "Phone_Change","Child_Count",
    "Mobile_Tag","Cleint_City_Rating","Client_Family_Members"
]


# -----------------------------
# HOME
# -----------------------------
@app.get("/")
def home():
    return {"status": "API running 🚀"}


# -----------------------------
# PREDICT
# -----------------------------
@app.post("/predict")
def predict(data: dict):

    try:
        # ---------------------------------
        # 1. INPUT → DF
        # ---------------------------------
        df = pd.DataFrame([data])

        # ---------------------------------
        # 2. ENSURE SCHEMA
        # ---------------------------------
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

        # ---------------------------------
        # 3. PREPROCESS
        # ---------------------------------
        df = preprocess(df)

        # ---------------------------------
        # 4. ENCODING (FIXED)
        # ---------------------------------
        df = pd.get_dummies(df)

        # ---------------------------------
        # 5. FEATURE SELECTION
        # ---------------------------------
        df = select_final_features(df, FINAL_FEATURES)

        # ---------------------------------
        # 6. ALIGN WITH TRAINING
        # ---------------------------------
        df = df.reindex(columns=train_cols, fill_value=0)

        df = df.fillna(0).astype(float)

        # ---------------------------------
        # 7. PREDICT (WITH CALIBRATED THRESHOLD)
        # ---------------------------------
        prob = model.predict_proba(df)[0][1]

        pred = int(prob >= threshold)

        return {
            "prediction": pred,
            "default_probability": float(prob),
            "threshold_used": float(threshold)
        }

    except Exception as e:
        return {"error": str(e)}