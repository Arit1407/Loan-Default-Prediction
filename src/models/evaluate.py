from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def evaluate(model, X, y):

    X = X.fillna(X.median()).astype(float)

    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y, prob),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred)
    }

    print(metrics)
    return metrics