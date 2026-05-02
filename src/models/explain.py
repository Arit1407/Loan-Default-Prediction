import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def shap_analysis(model, X):

    # 🔥 CRITICAL FIX: force numeric
    X = X.copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    # Fill NaN (SHAP requirement)
    X = X.fillna(X.median())

    # Convert everything to float
    X = X.astype(float)

    # Sample
    X_sample = X.sample(min(1000, len(X)), random_state=42)

    # 🔥 Use predict_proba
    explainer = shap.Explainer(model.predict_proba, X_sample)

    shap_values = explainer(X_sample)

    # Binary → class 1
    values = shap_values[..., 1]

    # Plot
    shap.plots.beeswarm(values, show=False)
    plt.savefig("reports/shap_summary.png")
    plt.close()

    # Importance
    shap_df = pd.DataFrame({
        "feature": X_sample.columns,
        "importance": np.abs(values.values).mean(axis=0)
    }).sort_values(by="importance", ascending=False)

    shap_df.to_csv("reports/shap_importance.csv", index=False)

    print("✅ SHAP saved successfully")