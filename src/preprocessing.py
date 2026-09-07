# ============================================
# STEP 2: Data Cleaning & Preprocessing
# Purpose: Prepare the raw AI4I dataset into a
# clean feature set ready for model training.
# NOTE: No model training happens in this file.
# ============================================

import pandas as pd

def load_and_clean_data(filepath="data/ai4i2020.csv"):
    """
    Loads the AI4I 2020 dataset and performs cleaning:
    - Drops identifier columns (no predictive value)
    - Drops failure-type flag columns (data leakage)
    - Keeps only real sensor/operating features + target
    """

    # Load raw data
    df = pd.read_csv(filepath)

    # ----------------------------------------------------
    # 1. Drop identifier columns
    # Reason: 'UDI' is just a row number, 'Product ID' is
    # a unique text label per row. Neither carries any
    # generalizable pattern about WHY a machine fails.
    # ----------------------------------------------------
    df = df.drop(columns=["UDI", "Product ID"])

    # ----------------------------------------------------
    # 2. Drop failure-type flag columns (DATA LEAKAGE)
    # Reason: TWF, HDF, PWF, OSF, RNF tell us *why* and
    # *that* a failure already happened. These wouldn't be
    # known in advance in a real prediction scenario, so
    # including them would let the model "cheat".
    # ----------------------------------------------------
    df = df.drop(columns=["TWF", "HDF", "PWF", "OSF", "RNF"])

    # ----------------------------------------------------
    # 3. Confirm no missing values
    # Reason: We already verified 0 missing values, but we
    # re-check here so the pipeline is robust and self-
    # documenting (important for reproducibility).
    # ----------------------------------------------------
    assert df.isnull().sum().sum() == 0, "Unexpected missing values found!"

    # ----------------------------------------------------
    # 4. Separate features (X) and target (y)
    # Reason: Standard supervised learning setup — inputs
    # (X) predict output (y).
    # ----------------------------------------------------
    X = df.drop(columns=["Machine failure"])
    y = df["Machine failure"]

    return X, y


if __name__ == "__main__":
    X, y = load_and_clean_data()

    print("Cleaned feature columns:")
    print(X.columns.tolist())

    print("\nShape of X (features):", X.shape)
    print("Shape of y (target):", y.shape)

    print("\nFirst 5 rows of cleaned features:")
    print(X.head())

    print("\nData types after cleaning:")
    print(X.dtypes)