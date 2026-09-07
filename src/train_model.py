# ============================================
# FINAL STEP: Train & Save the Best Model Pipeline
# Purpose: Build the complete preprocessing + Random
# Forest pipeline (selected based on real F1-score
# comparison), train it, and save it for the Streamlit app.
# ============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import joblib

# ----------------------------------------------------
# 1. Load and clean data
# ----------------------------------------------------
df = pd.read_csv("data/ai4i2020.csv")
df = df.drop(columns=["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"])

X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

# ----------------------------------------------------
# 2. Define feature groups
# IMPORTANT: This exact order/naming must match what the
# Streamlit app sends later.
# ----------------------------------------------------
numeric_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
categorical_features = ["Type"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# ----------------------------------------------------
# 3. Stratified train/test split (same as before, random_state
# fixed so results are reproducible)
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------
# 4. Build FINAL pipeline: preprocessing + Random Forest
# Reason: Random Forest was selected because it achieved
# the best F1-score (0.6822) in our model comparison —
# the best balance of catching real failures (recall) while
# keeping false alarms low (precision).
# ----------------------------------------------------
final_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    ))
])

# ----------------------------------------------------
# 5. Train the final pipeline
# ----------------------------------------------------
final_pipeline.fit(X_train, y_train)

# ----------------------------------------------------
# 6. Re-verify performance on test set
# (confirms this matches your earlier comparison run —
# no hardcoded/invented numbers, computed fresh here)
# ----------------------------------------------------
y_pred = final_pipeline.predict(X_test)

print("===== FINAL MODEL: Random Forest =====")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall   :", round(recall_score(y_test, y_pred), 4))
print("F1-score :", round(f1_score(y_test, y_pred), 4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Failure", "Failure"]))

# ----------------------------------------------------
# 7. Save the complete pipeline (preprocessing + model)
# Reason: Saving the FULL pipeline (not just the raw
# RandomForestClassifier) means the Streamlit app can pass
# raw user input directly — scaling + one-hot encoding of
# 'Type' happens automatically inside, with zero extra code
# needed in the app.
# ----------------------------------------------------
joblib.dump(final_pipeline, "models/predictive_maintenance_model.pkl")
print("\nFinal pipeline saved to: models/predictive_maintenance_model.pkl")

# ----------------------------------------------------
# 8. Save the exact feature names/order for reference
# Reason: The Streamlit app MUST send a DataFrame with
# these exact column names, in this exact order, or the
# ColumnTransformer will raise an error.
# ----------------------------------------------------
feature_info = {
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "all_features_in_order": numeric_features + categorical_features
}
joblib.dump(feature_info, "models/feature_info.pkl")
print("Feature info saved to: models/feature_info.pkl")