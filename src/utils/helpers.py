def select_final_features(X, features):
    cols = []
    for f in features:
        cols += [c for c in X.columns if c.startswith(f)]

    X = X[cols]

    # Remove duplicates
    X = X.loc[:, ~X.columns.duplicated()]

    return X