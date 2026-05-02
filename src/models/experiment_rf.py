import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from src.utils.helpers import select_final_features
from src.config.features import FINAL_FEATURES


def run_rf_experiments(X_train, y_train, X_test, y_test):

    mlflow.set_experiment("RF_Experiments")

    # Feature selection
    X_train = select_final_features(X_train, FINAL_FEATURES)
    X_test = select_final_features(X_test, FINAL_FEATURES)

    # 🔥 FINAL NA FIX (CRITICAL)
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())

    samplers = {
        "no_sampling": None,
        "oversampling": RandomOverSampler(random_state=42),
        "smote": SMOTE(random_state=42)
    }

    for name, sampler in samplers.items():

        with mlflow.start_run(run_name=f"rf_{name}"):

            X_tr, y_tr = X_train.copy(), y_train.copy()

            # Apply sampling
            if sampler:
                X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)

            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_tr, y_tr)

            # Predictions
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:, 1]

            # Metrics
            roc = roc_auc_score(y_test, prob)
            precision = precision_score(y_test, pred)
            recall = recall_score(y_test, pred)
            f1 = f1_score(y_test, pred)

            # Logging
            mlflow.log_param("model", "RandomForest")
            mlflow.log_param("sampling", name)

            mlflow.log_metric("roc_auc", roc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(model, name="model")

    print("✅ RF Experiments Completed")


# ENTRY POINT
if __name__ == "__main__":

    from src.data.data_loader import load_data
    from src.data.preprocessing import preprocess, split_data, encode_data
    from src.config.config import DATA_PATH

    df = load_data(DATA_PATH)
    df = preprocess(df)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test = encode_data(X_train, X_test)

    run_rf_experiments(X_train, y_train, X_test, y_test)