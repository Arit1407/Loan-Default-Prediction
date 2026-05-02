import joblib
from src.models.train import train_model
from src.models.evaluate import evaluate
from src.utils.helpers import select_final_features
from src.config.features import FINAL_FEATURES

def retrain_if_needed(drift_df, X_train, y_train, X_test, y_test, model_path, baseline_metrics):

    drift_pct = drift_df["drift"].mean()

    model = joblib.load(model_path)
    X_test_fs = select_final_features(X_test, FINAL_FEATURES)

    current_metrics = evaluate(model, X_test_fs, y_test)

    roc_drop = baseline_metrics["roc_auc"] - current_metrics["roc_auc"]
    rec_drop = baseline_metrics["recall"] - current_metrics["recall"]

    if drift_pct > 0.3 or (drift_pct > 0.1 and (roc_drop > 0.03 or rec_drop > 0.05)):

        print("⚠️ Retraining triggered")

        X_train_fs = select_final_features(X_train, FINAL_FEATURES)

        model = train_model(X_train_fs, y_train)
        evaluate(model, X_test_fs, y_test)

        joblib.dump(model, model_path)

        print("✅ Model retrained")

    else:
        print("✅ No retraining needed")