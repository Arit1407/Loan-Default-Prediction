from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler


def train_model(X, y):


    # Oversampling (best from experiments)
    X_res, y_res = RandomOverSampler(random_state=42).fit_resample(X, y)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )

    model.fit(X_res, y_res)

    return model