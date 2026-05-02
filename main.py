from src.data.data_loader import load_data
from src.data.preprocessing import preprocess, split_data, encode_data
from src.utils.helpers import select_final_features
from src.config.features import FINAL_FEATURES
from src.config.config import DATA_PATH, MODEL_PATH

from src.models.train import train_model
from src.models.evaluate import evaluate
from src.models.retrain import retrain_if_needed
from src.models.explain import shap_analysis
from src.monitoring.drift import detect_drift
from src.reports.report_generator import save_metrics, save_drift, save_report

import joblib
import numpy as np

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
df = load_data(DATA_PATH)
df = preprocess(df)

# -----------------------------
# SPLIT + ENCODE
# -----------------------------
X_train, X_test, y_train, y_test = split_data(df)
X_train, X_test = encode_data(X_train, X_test)

# -----------------------------
# FEATURE SELECTION
# -----------------------------
X_train_fs = select_final_features(X_train, FINAL_FEATURES)
X_test_fs = select_final_features(X_test, FINAL_FEATURES)

# -----------------------------
# 🔥 SAVE TRAINING COLUMNS (VERY IMPORTANT)
# -----------------------------
joblib.dump(X_train_fs.columns, "models/columns.pkl")

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = train_model(X_train_fs, y_train)

# -----------------------------
# EVALUATE
# -----------------------------
metrics = evaluate(model, X_test_fs, y_test)
save_metrics(metrics)

# -----------------------------
# 🔥 SAVE MODEL
# -----------------------------
joblib.dump(model, MODEL_PATH)

# -----------------------------
# 🔥 CALIBRATE THRESHOLD (IMPORTANT)
# -----------------------------
y_prob = model.predict_proba(X_test_fs)[:, 1]

best_threshold = 0.5
best_f1 = 0

for t in np.arange(0.2, 0.6, 0.05):
    y_pred = (y_prob >= t).astype(int)
    f1 = ((y_pred & y_test).sum())  # simple proxy (or use sklearn f1_score)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

# save threshold
joblib.dump(best_threshold, "models/threshold.pkl")

print(f"✅ Best threshold: {best_threshold}")

# -----------------------------
# DRIFT
# -----------------------------
drift_df = detect_drift(X_train_fs, X_test_fs)
save_drift(drift_df)

# -----------------------------
# RETRAIN
# -----------------------------
retrain_if_needed(
    drift_df, X_train, y_train, X_test, y_test, MODEL_PATH, metrics
)

# -----------------------------
# SHAP
# -----------------------------
shap_analysis(model, X_test_fs)

# -----------------------------
# REPORT
# -----------------------------
save_report(metrics, drift_df)

print("✅ Full pipeline done")