# 🏭 Industrial Equipment Failure & Predictive Maintenance

An AI-powered **Predictive Maintenance System** that uses machine learning to predict industrial equipment failures before they occur. The system analyzes machine operating parameters, estimates failure probability, determines the risk level, and provides a recommended maintenance action.

The project includes a **Streamlit web application** for easy interaction with the trained machine learning model.


## 📌 Project Overview

Unexpected equipment failures can cause production downtime, maintenance costs, and operational losses.

This project aims to reduce such problems by using historical machine data and machine learning to identify conditions that may lead to equipment failure.

The system takes machine parameters such as:

* Machine Type
* Air Temperature
* Process Temperature
* Rotational Speed
* Torque
* Tool Wear

and predicts whether the machine is likely to experience a failure.

The application also provides:

* Failure / No-Failure prediction
* Failure probability
* Low / Medium / High risk classification
* Maintenance recommendation
* Model performance metrics


## 🎯 Objectives

* Predict potential equipment failures in advance.
* Identify machines with high failure risk.
* Reduce unexpected machine downtime.
* Support preventive maintenance decisions.
* Provide an easy-to-use interface for machine failure prediction.
* Demonstrate the use of machine learning in industrial predictive maintenance.


## 🤖 Machine Learning Model

Several classification models were evaluated, including:

* Logistic Regression
* Decision Tree
* Random Forest

**Random Forest Classifier** was selected because it achieved the best F1-score among the evaluated models and provides a good balance between identifying actual failures and reducing false alarms.

### Model Performance

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 97.95% |
| Precision | 72.13% |
| Recall    | 64.71% |
| F1 Score  | 68.22% |

These are the evaluation metrics displayed by the project's application.


## ⚙️ System Workflow

```text
Machine Data
     ↓
Feature Processing
     ↓
Random Forest Model
     ↓
Failure Prediction
     ↓
Failure Probability
     ↓
Risk Classification
     ↓
Maintenance Recommendation

## 📊 Input Parameters

The application uses the following machine parameters:

| Parameter           | Description                                                |
| ------------------- | ---------------------------------------------------------- |
| Machine Type        | Product quality variant: L, M, or H                        |
| Air Temperature     | Ambient temperature around the machine, measured in Kelvin |
| Process Temperature | Temperature of the machining process, measured in Kelvin   |
| Rotational Speed    | Machine/tool rotation speed in RPM                         |
| Torque              | Rotational force applied during machining                  |
| Tool Wear           | Tool usage time in minutes                                 |

These parameters are passed to the trained model in the required feature order before prediction.


## 🚦 Risk Classification

The system converts the predicted failure probability into three risk levels:

| Failure Probability | Risk Level     |
| ------------------- | -------------- |
| < 30%               | 🟢 Low Risk    |
| 30% – <60%          | 🟡 Medium Risk |
| ≥ 60%               | 🔴 High Risk   |

The application then provides an appropriate maintenance recommendation.

### 🟢 Low Risk

Machine operating conditions appear normal. Continue operation and follow the scheduled maintenance plan.

### 🟡 Medium Risk

Elevated operating conditions are detected. Inspection and preventive maintenance are recommended.

### 🔴 High Risk

Potential failure conditions are detected. The machine should be inspected and preventive maintenance should be considered before continued operation.


## 🖥️ Application

The project includes a **Streamlit web application**.

The application:

1. Accepts machine operating parameters.
2. Loads the pre-trained machine learning pipeline.
3. Generates a failure prediction.
4. Calculates failure probability.
5. Determines the risk level.
6. Displays a probability chart.
7. Provides a maintenance recommendation.
8. Displays model evaluation metrics.

The application loads the trained model from:

```text
models/predictive_maintenance_model.pkl
```

and feature information from:

```text
models/feature_info.pkl
```

The Streamlit application does not train the model; it uses the already-trained pipeline for inference.

---

## 📁 Project Structure

```text
Inductive_preditive_maintenance/
│
├── data/
│   └── Dataset files
│
├── models/
│   ├── predictive_maintenance_model.pkl
│   └── feature_info.pkl
│
├── notebooks/
│   └── Data analysis and model development notebooks
│
├── src/
│   └── Source code for data processing and model development
│
├── app.py
│   └── Streamlit web application
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
```

The repository currently follows this structure with separate directories for data, models, notebooks, and source code.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Random Forest**
* **Joblib**
* **Streamlit**
* **Matplotlib**
* **Seaborn**
* **Jupyter Notebook**

The repository's requirements include Scikit-learn, Pandas, NumPy, Streamlit, and other data-science dependencies.

---

