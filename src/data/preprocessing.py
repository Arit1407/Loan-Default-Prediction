import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config.config import TARGET, TEST_SIZE, RANDOM_STATE

# ============================================
# 1. CLEAN NUMERIC COLUMNS (VERY IMPORTANT)
# ============================================
def clean_numeric(df, cols):

    for col in cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[^0-9.-]", "", regex=True)  # remove &, commas, etc
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ============================================
# 2. PREPROCESS FUNCTION (MAIN)
# ============================================
def preprocess(df):

    # -------------------------
    # NUMERIC COLUMNS (FROM YOUR NOTEBOOK)
    # -------------------------
    num_cols = [
        "Client_Income",
        "Credit_Amount",
        "Loan_Annuity",
        "Age_Days",
        "Employed_Days",
        "Registration_Days",
        "ID_Days",
        "Score_Source_3",
        "Population_Region_Relative"
    ]

    # CLEAN
    df = clean_numeric(df, num_cols)

    # -------------------------
    # FEATURE ENGINEERING
    # -------------------------
    df["Years_Employed"] = abs(df["Employed_Days"]) / 365
    df["Age_Years"] = abs(df["Age_Days"]) / 365


    df.drop(columns=["Own_House_Age"], inplace=True,errors='ignore')

    # -------------------------
    # HANDLE MISSING VALUES
    # -------------------------

    # HIGH MISSING → KEEP + FLAG
    high_missing_cols = ['Years_Employed',
    'Social_Circle_Default',
    'Client_Occupation',
    'Credit_Bureau',
    'Score_Source_1',
    'Score_Source_3',
    'Bike_Owned']

    for col in high_missing_cols:
        df[col + "_missing"] = df[col].isnull().astype(int)

    num_cols = df.select_dtypes(include=['float64','int64']).columns

    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    cat_cols = df.select_dtypes(include='object').columns

    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna("Unknown", inplace=True)

    df_out = df.copy()

    # Population_Region_Relative should be in [0, 1]
    df_out.loc[df_out["Population_Region_Relative"] > 1, "Population_Region_Relative"] = np.nan
    df_out["Population_Region_Relative"].fillna(df_out["Population_Region_Relative"].median(), inplace=True)

    # Score_Source_2 should be in [0, 1]
    df_out.loc[df_out["Score_Source_2"] > 1, "Score_Source_2"] = np.nan
    df_out["Score_Source_2"].fillna(df_out["Score_Source_2"].median(), inplace=True)

    cap_cols = [
    "Client_Income",
    "Credit_Amount",
    "Loan_Annuity",
    "Years_Employed",
    "Registration_Days",
    "Phone_Change",
    "Credit_Bureau"
    ]

    for col in cap_cols:
        lower = df_out[col].quantile(0.01)
        upper = df_out[col].quantile(0.99)
        df_out[col] = df_out[col].clip(lower, upper)
            
    df_out["Child_Count"] = df_out["Child_Count"].clip(upper=5)

    df_out["Years_Employed"] = np.log1p(df_out["Years_Employed"])

    df_out.drop(columns=[
    "Mobile_Tag",
    "Cleint_City_Rating",
    "Client_Family_Members"], inplace=True,errors='ignore')

    return df_out


# ============================================
# 3. SPLIT
# ============================================
def split_data(df):

    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )


# ============================================
# 4. ENCODING (FROM YOUR NOTEBOOK)
# ============================================
def encode_data(X_train, X_test):

    # -------------------------
    # BINARY
    # -------------------------
    for df_ in [X_train, X_test]:
        df_["Loan_Contract_Type"] = df_["Loan_Contract_Type"].map({"CL":0,"RL":1,"Unknown":0})
        df_["Client_Permanent_Match_Tag"] = df_["Client_Permanent_Match_Tag"].map({"Yes":1,"No":0})
        df_["Client_Contact_Work_Tag"] = df_["Client_Contact_Work_Tag"].map({"Yes":1,"No":0})

    # -------------------------
    # FREQUENCY
    # -------------------------
    for col in ["Client_Occupation","Type_Organization"]:
        freq = X_train[col].value_counts(normalize=True)
        X_train[col] = X_train[col].map(freq)
        X_test[col] = X_test[col].map(freq)

    # -------------------------
    # ONE HOT
    # -------------------------
    onehot_cols = [
        "Accompany_Client",
        "Client_Income_Type",
        "Client_Marital_Status",
        "Client_Gender",
        "Client_Housing_Type",
        "Client_Education"
    ]

    X_train = pd.get_dummies(X_train, columns=onehot_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=onehot_cols, drop_first=True)

    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    return X_train, X_test