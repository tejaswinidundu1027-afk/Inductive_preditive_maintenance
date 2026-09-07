# ============================================
# STEP 1: Load and Inspect the Dataset
# Purpose: Understand structure before touching
# any preprocessing or modeling code.
# ============================================

import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data/ai4i2020.csv")

# 1. Basic shape
print("Shape of dataset (rows, columns):", df.shape)

# 2. Column names
print("\nColumn names:")
print(df.columns.tolist())

# 3. Data types
print("\nData types:\n")
print(df.dtypes)

# 4. First 5 rows
print("\nFirst 5 rows:")
print(df.head())

# 5. Missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 6. Statistical summary
print("\nStatistical summary:")
print(df.describe())

# 7. Target variable balance
print("\nMachine failure value counts:")
print(df["Machine failure"].value_counts())
print("\nMachine failure percentage:")
print(df["Machine failure"].value_counts(normalize=True) * 100)