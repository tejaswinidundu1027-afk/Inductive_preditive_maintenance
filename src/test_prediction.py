# ============================================
# TEST SCRIPT: Load saved pipeline and predict
# on a new, unseen machine reading.
# Purpose: Verify the .pkl file works correctly
# BEFORE connecting it to Streamlit.
# ============================================

import pandas as pd
import joblib

# ----------------------------------------------------
# 1. Load the saved pipeline and feature info
# ----------------------------------------------------
pipeline = joblib.load("models/predictive_maintenance_model.pkl")
feature_info = joblib.load("models/feature_info.pkl")

print("Model expects these features, in this order:")
print(feature_info["all_features_in_order"])

# ----------------------------------------------------
# 2. Create a new sample machine reading
# IMPORTANT: Column names must match EXACTLY
# (same spelling, spacing, brackets, units) as training data.
# This example uses realistic values within the dataset's
# observed ranges (see your EDA describe() output).
# ----------------------------------------------------
new_machine = pd.DataFrame([{
    "Air temperature [K]": 302.5,
    "Process temperature [K]": 311.0,
    "Rotational speed [rpm]": 1380,
    "Torque [Nm]": 58.0,
    "Tool wear [min]": 200,
    "Type": "L"
}])

# Reorder columns to match training order exactly (safety step)
new_machine = new_machine[feature_info["all_features_in_order"]]

# ----------------------------------------------------
# 3. Predict
# ----------------------------------------------------
prediction = pipeline.predict(new_machine)[0]
probability = pipeline.predict_proba(new_machine)[0]

print("\n--- Prediction Result ---")
print("Predicted class:", "Failure" if prediction == 1 else "No Failure")
print(f"Probability of No Failure: {probability[0]:.4f}")
print(f"Probability of Failure   : {probability[1]:.4f}")