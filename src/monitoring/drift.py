import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(expected, actual, bins=10):

    # Remove NaNs
    expected = expected.dropna()
    actual = actual.dropna()

    # If empty or constant → skip
    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    if expected.nunique() <= 1 or actual.nunique() <= 1:
        return 0.0

    try:
        expected_perc = np.histogram(expected, bins)[0] / len(expected)
        actual_perc = np.histogram(actual, bins)[0] / len(actual)

        return np.sum(
            (expected_perc - actual_perc) *
            np.log((expected_perc + 1e-6) / (actual_perc + 1e-6))
        )
    except:
        return np.nan


def detect_drift(train_df, new_df):

    rows = []

    for col in train_df.columns:

        train_col = train_df[col]
        new_col = new_df[col]

        # Convert bool → int
        if train_col.dtype == "bool":
            train_col = train_col.astype(int)
            new_col = new_col.astype(int)

        # Drop NaNs
        train_col = train_col.dropna()
        new_col = new_col.dropna()

        # Skip empty columns
        if len(train_col) == 0 or len(new_col) == 0:
            continue

        # PSI
        psi_val = psi(train_col, new_col)

        # KS
        try:
            ks_stat, _ = ks_2samp(train_col, new_col)
        except:
            ks_stat = np.nan

        drift_flag = (psi_val is not np.nan and psi_val > 0.2) and (ks_stat > 0.1)

        rows.append({
            "feature": col,
            "psi": psi_val,
            "ks": ks_stat,
            "drift": drift_flag
        })

    return pd.DataFrame(rows)