import os
import json
import pandas as pd

os.makedirs("reports", exist_ok=True)

def save_metrics(metrics):
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

def save_drift(drift_df):
    drift_df.to_csv("reports/drift_report.csv", index=False)

def save_report(metrics, drift_df):

    drift_pct = drift_df["drift"].mean()

    with open("reports/model_report.txt", "w") as f:
        f.write("MODEL PERFORMANCE\n")
        for k,v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

        f.write("\nDRIFT SUMMARY\n")
        f.write(f"Drift %: {drift_pct:.2%}\n")