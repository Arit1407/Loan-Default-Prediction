import streamlit as st
import requests

# -----------------------------
# CONFIG
# -----------------------------
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Loan Default Predictor", layout="centered")

st.title("💳 Loan Default Prediction")
st.markdown("Enter client details to predict loan default risk")

# -----------------------------
# INPUT SECTION
# -----------------------------
st.subheader("📊 Financial Info")

income = st.number_input("Client Income", value=30000)
credit = st.number_input("Credit Amount", value=80000)
annuity = st.number_input("Loan Annuity", value=3000)

st.subheader("👤 Personal Info")

age = st.number_input("Age (years)", value=30)
emp_years = st.number_input("Years Employed", value=5)

gender = st.selectbox("Gender", ["Male", "Female"])
marital = st.selectbox("Marital Status", ["M", "S", "D", "W"])
education = st.selectbox("Education", ["Secondary", "Graduation"])
housing = st.selectbox("Housing Type", ["Home", "Rental"])

st.subheader("🏦 Loan Info")

contract = st.selectbox("Loan Contract Type", ["CL", "RL"])
income_type = st.selectbox("Income Type", ["Service", "Commercial"])
accompany = st.selectbox("Accompany Client", ["Alone", "Partner"])

st.subheader("📉 Risk Indicators")

score1 = st.slider("Score Source 1", 0.0, 1.0, 0.5)
score2 = st.slider("Score Source 2", 0.0, 1.0, 0.6)
score3 = st.slider("Score Source 3", 0.0, 1.0, 0.55)

credit_bureau = st.number_input("Credit Bureau Count", value=1)
social_default = st.number_input("Social Circle Default", value=0)

st.subheader("🚗 Assets")

car = st.selectbox("Car Owned", [0, 1])
bike = st.selectbox("Bike Owned", [0, 1])

occupation = st.text_input("Occupation", "Sales")
organization = st.text_input("Organization Type", "Business")

# -----------------------------
# PREPARE INPUT
# -----------------------------
data = {
    "Client_Income": income,
    "Credit_Amount": credit,
    "Loan_Annuity": annuity,

    "Age_Days": -age * 365,
    "Employed_Days": -emp_years * 365,

    "Score_Source_1": score1,
    "Score_Source_2": score2,
    "Score_Source_3": score3,

    "Credit_Bureau": credit_bureau,
    "Social_Circle_Default": social_default,

    "Client_Gender": gender,
    "Client_Marital_Status": marital,
    "Client_Education": education,
    "Client_Housing_Type": housing,

    "Loan_Contract_Type": contract,
    "Client_Income_Type": income_type,
    "Accompany_Client": accompany,

    "Client_Occupation": occupation,
    "Type_Organization": organization,

    "Car_Owned": car,
    "Bike_Owned": bike
}

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("🚀 Predict Default Risk"):

    try:
        response = requests.post(API_URL, json=data)
        result = response.json()

        if "error" in result:
            st.error(f"❌ Error: {result['error']}")
        else:
            pred = result["prediction"]
            prob = result["default_probability"]

            # -----------------------------
            # RISK LABEL
            # -----------------------------
            if prob >= 0.7:
                risk = "🔴 High Risk"
                st.error(f"{risk}")
            elif prob >= 0.4:
                risk = "🟡 Medium Risk"
                st.warning(f"{risk}")
            else:
                risk = "🟢 Low Risk"
                st.success(f"{risk}")

            st.metric("Default Probability", f"{prob:.2f}")

    except Exception as e:
        st.error("⚠️ API not running or connection failed")