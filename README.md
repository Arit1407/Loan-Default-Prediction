# 💳 Credit Risk Prediction System

An end-to-end **Machine Learning pipeline** for predicting loan default risk, built with production-ready components including preprocessing, experimentation, drift monitoring, retraining, and deployment using **FastAPI** and **Streamlit**.

---

## 🚀 Project Overview

This project predicts whether a client will **default on a loan** using financial and behavioral data.

It covers the complete ML lifecycle:

* Data preprocessing (EDA-driven)
* Feature engineering & selection
* Model training (XGBoost, Random Forest)
* Experiment tracking (MLflow)
* Drift detection (PSI, CSI, KS)
* Automated retraining
* Model explainability (SHAP)
* Deployment (FastAPI + Streamlit)

---

## 🧠 Key Features

* ✔ EDA-aligned preprocessing pipeline
* ✔ Missing value handling with statistical validation
* ✔ Feature selection based on importance
* ✔ Sampling techniques for imbalanced data
* ✔ MLflow experiment tracking
* ✔ Drift detection (PSI, CSI, KS)
* ✔ Automatic retraining pipeline
* ✔ SHAP explainability
* ✔ FastAPI backend for inference
* ✔ Streamlit frontend for interaction
* ✔ Logging and exception handling

---

## 📁 Project Structure

```bash
credit-risk-project/

├── data/
│   └── raw/
│       └── train.csv

├── models/
│   └── model.pkl

├── logs/
├── mlruns/
├── reports/

├── src/
│
│   ├── config/
│   │   ├── config.py
│   │   └── features.py
│
│   ├── data/
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── experiment_xgb.py
│   │   ├── experiment_rf.py
│   │   ├── explain.py
│   │   └── retrain.py
│
│   ├── monitoring/
│   │   └── drift.py
│
│   ├── reports/
│   │   └── report_generator.py
│
│   ├── utils/
│   │   ├── helpers.py
│   │   ├── logger.py
│   │   └── exception.py
│
│   └── api/
│       └── app.py
│
├── streamlit_app.py
├── main.py
├── requirements.txt
```

---

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd credit-risk-project

pip install -r requirements.txt
```

---

## 🏋️‍♂️ Run Full Pipeline

```bash
python main.py
```

This will:

* Load and preprocess data
* Train model
* Evaluate performance
* Save model artifacts
* Detect drift
* Trigger retraining if needed
* Generate reports

---

## 🧪 Run Experiments

```bash
python -m src.models.experiment_xgb
python -m src.models.experiment_rf
```

👉 Experiments tracked using MLflow (`mlruns/`)

---

## 🌐 Run FastAPI

```bash
uvicorn src.api.app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 🎨 Run Streamlit

```bash
streamlit run streamlit_app.py
```

---

## 📊 Model Performance

* ROC-AUC: ~0.73
* Handles class imbalance using sampling
* Uses calibrated threshold for prediction

---

## 📉 Drift Detection

Implemented:

* Population Stability Index (PSI)
* Characteristic Stability Index (CSI)
* Kolmogorov–Smirnov (KS)

Retraining is triggered when drift exceeds thresholds.

---

## 🔍 Explainability

* SHAP used for feature importance
* Helps interpret model predictions

---

## 🛠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* MLflow
* FastAPI
* Streamlit
* SHAP

---

## 🧠 Key Learnings

* Building end-to-end ML pipelines
* Handling data drift in production
* Ensuring training–inference consistency
* Feature engineering & selection strategies
* Model monitoring and retraining

---

## 📌 Future Improvements

* Dockerization
* CI/CD pipeline
* Cloud deployment (AWS/GCP)
* Real-time monitoring dashboard

---

## 👨‍💻 Author

**Arit Kar**

---

## ⭐ If you like this project, give it a star!
