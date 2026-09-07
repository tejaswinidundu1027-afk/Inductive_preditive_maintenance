# ============================================
# app.py
# Industrial Equipment Failure & Predictive
# Maintenance System - Streamlit Web App
#
# Purpose: Load the ALREADY TRAINED pipeline
# (models/predictive_maintenance_model.pkl) and
# let a user enter machine parameters to get a
# failure prediction, risk level, probability,
# and a maintenance recommendation.
#
# IMPORTANT: This file does NOT train or modify
# any model. It only loads the saved pipeline.
# ============================================

import streamlit as st
import pandas as pd
import joblib
import os

# ----------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ----------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------
# Constants: ACTUAL evaluation metrics from training
# (obtained during model comparison in train_model.py -
# not invented or hardcoded here beyond display)
# ----------------------------------------------------
MODEL_METRICS = {
    "Accuracy": 0.9795,
    "Precision": 0.7213,
    "Recall": 0.6471,
    "F1-score": 0.6822
}

MODEL_PATH = "models/predictive_maintenance_model.pkl"
FEATURE_INFO_PATH = "models/feature_info.pkl"


# ----------------------------------------------------
# Load model + feature info (cached so it only loads once)
# ----------------------------------------------------
@st.cache_resource
def load_model_and_features():
    """
    Loads the saved pipeline and feature info.
    Returns (pipeline, feature_info, error_message).
    """
    if not os.path.exists(MODEL_PATH):
        return None, None, f"Model file not found at '{MODEL_PATH}'. Please run train_model.py first."

    if not os.path.exists(FEATURE_INFO_PATH):
        return None, None, f"Feature info file not found at '{FEATURE_INFO_PATH}'. Please run train_model.py first."

    try:
        pipeline = joblib.load(MODEL_PATH)
        feature_info = joblib.load(FEATURE_INFO_PATH)
        return pipeline, feature_info, None
    except Exception as e:
        return None, None, f"Error loading model files: {e}"


pipeline, feature_info, load_error = load_model_and_features()

# ======================================================
# CUSTOM CSS - Industrial AI Dashboard Theme
# ======================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1180px;}

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: radial-gradient(circle at 15% 0%, #101a2e 0%, #0a0f1e 45%, #070b16 100%);
        color: #dbe4f3;
    }

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0 1.1rem 0;
        border-bottom: 1px solid rgba(148, 178, 224, 0.12);
        margin-bottom: 1.8rem;
    }
    .topbar-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: 0.2px;
    }
    .topbar-title span { color: #22d3ee; }
    .topbar-sub {
        font-size: 0.72rem;
        color: #7d8bab;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 1px;
    }
    .status-pill {
        display: flex;
        align-items: center;
        gap: 7px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 1px;
        color: #4ade80;
        background: rgba(74, 222, 128, 0.08);
        border: 1px solid rgba(74, 222, 128, 0.28);
        padding: 5px 12px;
        border-radius: 20px;
    }
    .status-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: #4ade80;
        box-shadow: 0 0 8px #4ade80;
        animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; } 50% { opacity: 0.35; } 100% { opacity: 1; }
    }

    .hero {
        text-align: center;
        padding: 1.4rem 0 2rem 0;
    }
    .hero-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 1.6px;
        color: #22d3ee;
        background: rgba(34, 211, 238, 0.09);
        border: 1px solid rgba(34, 211, 238, 0.3);
        padding: 5px 14px;
        border-radius: 20px;
        margin-bottom: 1.1rem;
    }
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.18;
        color: #f8fafc;
        margin: 0 0 0.8rem 0;
        letter-spacing: -0.5px;
    }
    .hero h1 .accent {
        background: linear-gradient(90deg, #22d3ee, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero p {
        font-size: 1.0rem;
        color: #93a4c3;
        max-width: 620px;
        margin: 0 auto;
        line-height: 1.6;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.035);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(148, 178, 224, 0.14);
        border-radius: 16px;
        padding: 1.6rem 1.7rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
        height: 100%;
    }
    .card-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 1.5px;
        color: #5b6d92;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 1.1rem;
    }

    .section-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 1.6px;
        color: #22d3ee;
        text-transform: uppercase;
        text-align: center;
        margin-bottom: 0.4rem;
    }
    .section-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f1f5f9;
        text-align: center;
        margin-bottom: 1.6rem;
    }

    .result-idle {
        text-align: center;
        padding: 2.6rem 1rem;
        color: #6b7ba0;
    }
    .result-idle .icon { font-size: 2.2rem; margin-bottom: 0.8rem; opacity: 0.6; }

    .risk-banner {
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .risk-low {
        background: rgba(74, 222, 128, 0.09);
        border: 1px solid rgba(74, 222, 128, 0.35);
    }
    .risk-medium {
        background: rgba(250, 204, 21, 0.09);
        border: 1px solid rgba(250, 204, 21, 0.35);
    }
    .risk-high {
        background: rgba(248, 113, 113, 0.1);
        border: 1px solid rgba(248, 113, 113, 0.4);
    }
    .risk-badge-text {
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    .risk-low .risk-badge-text { color: #4ade80; }
    .risk-medium .risk-badge-text { color: #facc15; }
    .risk-high .risk-badge-text { color: #f87171; }
    .risk-sub { font-size: 0.8rem; color: #9fb0d0; margin-top: 2px; }

    .stat-row {
        display: flex;
        gap: 0.8rem;
        margin: 1rem 0;
    }
    .stat-box {
        flex: 1;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(148,178,224,0.12);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        text-align: center;
    }
    .stat-box .val { font-size: 1.25rem; font-weight: 700; color: #f1f5f9; }
    .stat-box .lbl { font-size: 0.68rem; color: #7d8bab; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 2px; }

    .recommendation-card {
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-size: 0.88rem;
        line-height: 1.55;
        margin-top: 0.6rem;
        border-left: 3px solid #22d3ee;
        background: rgba(34, 211, 238, 0.06);
        color: #c7d2e5;
    }

    .metric-tile {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(148,178,224,0.14);
        border-radius: 14px;
        padding: 1.3rem 0.5rem;
        text-align: center;
    }
    .metric-tile .num {
        font-size: 1.7rem;
        font-weight: 800;
        background: linear-gradient(90deg, #22d3ee, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-tile .lbl {
        font-size: 0.75rem;
        color: #8b9bc0;
        margin-top: 3px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .step-tile {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(148,178,224,0.14);
        border-radius: 12px;
        padding: 1.1rem 0.8rem;
        text-align: center;
    }
    .step-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #22d3ee;
        letter-spacing: 1px;
    }
    .step-name {
        font-size: 0.85rem;
        font-weight: 600;
        color: #e2e8f5;
        margin-top: 4px;
    }
    .step-arrow {
        color: #3b4c6e;
        font-size: 1.2rem;
        padding: 0 2px;
    }

    .glossary-row {
        display: flex;
        gap: 1rem;
        padding: 0.75rem 0;
        border-bottom: 1px solid rgba(148,178,224,0.08);
    }
    .glossary-term {
        min-width: 190px;
        font-weight: 600;
        color: #dbe4f3;
        font-size: 0.88rem;
    }
    .glossary-desc {
        color: #8b9bc0;
        font-size: 0.86rem;
        line-height: 1.5;
    }

    .app-footer {
        text-align: center;
        margin-top: 2.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(148,178,224,0.1);
        color: #5b6d92;
    }
    .app-footer .name { font-weight: 700; color: #9fb0d0; font-size: 0.9rem; }
    .app-footer .role { font-size: 0.78rem; margin-top: 2px; }
    .app-footer .stack {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        margin-top: 10px;
        color: #4a5a7e;
    }

    div[data-testid="stSelectbox"] label, div[data-testid="stNumberInput"] label {
        color: #9fb0d0 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #0ea5e9, #22d3ee) !important;
        color: #04121c !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 1rem !important;
        letter-spacing: 0.3px;
        box-shadow: 0 6px 20px rgba(34, 211, 238, 0.25);
        transition: transform 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 24px rgba(34, 211, 238, 0.35);
    }
    hr { border-color: rgba(148,178,224,0.1) !important; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# TOP BAR
# ======================================================
st.markdown("""
<div class="topbar">
    <div>
        <div class="topbar-title">🏭 Predictive Maintenance <span>AI</span></div>
        <div class="topbar-sub">Industrial Equipment Intelligence Platform</div>
    </div>
    <div class="status-pill"><span class="status-dot"></span> SYSTEM ONLINE</div>
</div>
""", unsafe_allow_html=True)

# ======================================================
# HERO SECTION
# ======================================================
st.markdown("""
<div class="hero">
    <div class="hero-badge">POWERED BY MACHINE LEARNING</div>
    <h1>Predict Equipment Failure<br><span class="accent">Before It Happens.</span></h1>
    <p>AI-powered predictive maintenance using machine operating conditions to identify
    potential equipment failures before costly downtime occurs.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Stop early with a clear error if the model failed to load
# ----------------------------------------------------
if load_error:
    st.error(f"⚠️ {load_error}")
    st.info(
        "**How to fix this:**\n\n"
        "1. Make sure you have run `python -m src.train_model` from the project root folder.\n"
        "2. Confirm that `models/predictive_maintenance_model.pkl` and `models/feature_info.pkl` exist.\n"
        "3. Make sure you are running `streamlit run app.py` from the **project root folder** "
        "(the folder that directly contains the `models/` folder), not from inside `src/`."
    )
    st.stop()

# ======================================================
# MACHINE HEALTH ANALYSIS - two-column layout
# ======================================================
left_col, right_col = st.columns([1, 1.15], gap="large")

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Input</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Machine Parameters</div>', unsafe_allow_html=True)

    machine_type = st.selectbox(
        "Machine Type",
        options=["L", "M", "H"],
        help="L = Low quality variant, M = Medium quality variant, H = High quality variant."
    )

    t1, t2 = st.columns(2)
    with t1:
        air_temp = st.number_input(
            "Air Temperature [K]",
            min_value=290.0, max_value=310.0, value=300.0, step=0.1,
            help="Ambient air temperature around the machine. Typical range: 295K - 304K."
        )
    with t2:
        process_temp = st.number_input(
            "Process Temperature [K]",
            min_value=300.0, max_value=320.0, value=310.0, step=0.1,
            help="Temperature of the machining process itself. Typical range: 305K - 314K."
        )

    s1, s2 = st.columns(2)
    with s1:
        rotational_speed = st.number_input(
            "Rotational Speed [rpm]",
            min_value=1000, max_value=3000, value=1500, step=10,
            help="Rotational speed of the tool/spindle. Typical range: 1150 - 2900 rpm."
        )
    with s2:
        torque = st.number_input(
            "Torque [Nm]",
            min_value=0.0, max_value=100.0, value=40.0, step=0.5,
            help="Rotational force applied during machining. Typical range: 3 - 77 Nm."
        )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0, max_value=260, value=100, step=1,
        help="Minutes the current cutting tool has been in use. Typical range: 0 - 253 min."
    )

    st.markdown("<div style='height: 0.4rem'></div>", unsafe_allow_html=True)
    predict_clicked = st.button("⚡ ANALYZE MACHINE", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Output</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">AI Risk Assessment</div>', unsafe_allow_html=True)

    if not predict_clicked:
        st.markdown("""
        <div class="result-idle">
            <div class="icon">◎</div>
            <div style="font-weight:600; color:#a9b8d6; margin-bottom:4px;">Ready for analysis</div>
            <div style="font-size:0.85rem;">Enter machine parameters and run the AI assessment.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        try:
            input_data = pd.DataFrame([{
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": rotational_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
                "Type": machine_type
            }])
            input_data = input_data[feature_info["all_features_in_order"]]

            prediction = pipeline.predict(input_data)[0]

            failure_probability = None
            if hasattr(pipeline, "predict_proba"):
                probabilities = pipeline.predict_proba(input_data)[0]
                failure_probability = probabilities[1]

            if failure_probability is not None:
                if failure_probability < 0.30:
                    risk_level, risk_class = "LOW RISK", "risk-low"
                elif failure_probability < 0.60:
                    risk_level, risk_class = "MEDIUM RISK", "risk-medium"
                else:
                    risk_level, risk_class = "HIGH RISK", "risk-high"
            else:
                risk_level = "HIGH RISK" if prediction == 1 else "LOW RISK"
                risk_class = "risk-high" if prediction == 1 else "risk-low"

            outcome_text = "Machine Failure" if prediction == 1 else "No Machine Failure"

            recommendations = {
                "risk-low": "Machine operating conditions appear normal. Continue operation and follow the scheduled maintenance plan.",
                "risk-medium": "Elevated operating conditions detected. Consider inspection and schedule preventive maintenance.",
                "risk-high": "Potential failure conditions detected. Inspect the machine and consider preventive maintenance before continued operation."
            }

            risk_icon = "🔴" if risk_class == "risk-high" else "🟡" if risk_class == "risk-medium" else "🟢"

            st.markdown(f"""
            <div class="risk-banner {risk_class}">
                <div>
                    <div class="risk-badge-text">{risk_level}</div>
                    <div class="risk-sub">Prediction: {outcome_text}</div>
                </div>
                <div style="font-size:1.8rem;">{risk_icon}</div>
            </div>
            """, unsafe_allow_html=True)

            if failure_probability is not None:
                st.markdown(f"""
                <div class="stat-row">
                    <div class="stat-box">
                        <div class="val">{failure_probability*100:.1f}%</div>
                        <div class="lbl">Failure Probability</div>
                    </div>
                    <div class="stat-box">
                        <div class="val">{(1-failure_probability)*100:.1f}%</div>
                        <div class="lbl">No-Failure Probability</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                prob_df = pd.DataFrame({
                    "Outcome": ["No Failure", "Failure"],
                    "Probability": [1 - failure_probability, failure_probability]
                })
                st.bar_chart(prob_df.set_index("Outcome"), height=160)

            st.markdown(f"""
            <div class="recommendation-card">
                <b>Recommended Action:</b><br>{recommendations[risk_class]}
            </div>
            """, unsafe_allow_html=True)

            with st.expander("View input data sent to the model"):
                st.dataframe(input_data, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred while making the prediction: {e}")
            st.info("Please double-check the entered values and try again.")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height: 2.4rem'></div>", unsafe_allow_html=True)

# ======================================================
# MODEL PERFORMANCE SECTION
# ======================================================
st.markdown('<div class="section-eyebrow">Evaluation</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
metric_display = [
    ("Accuracy", MODEL_METRICS["Accuracy"]),
    ("Precision", MODEL_METRICS["Precision"]),
    ("Recall", MODEL_METRICS["Recall"]),
    ("F1 Score", MODEL_METRICS["F1-score"]),
]
for col, (label, value) in zip([m1, m2, m3, m4], metric_display):
    with col:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="num">{value*100:.2f}%</div>
            <div class="lbl">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:#8b9bc0; font-size:0.86rem; margin-top:1.1rem; max-width:640px; margin-left:auto; margin-right:auto;">
Random Forest was selected because it achieved the highest F1-score among the evaluated
classification models (Logistic Regression, Decision Tree, Random Forest), balancing the
ability to catch real failures against the rate of false alarms.
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 2.6rem'></div>", unsafe_allow_html=True)

# ======================================================
# HOW THE SYSTEM WORKS
# ======================================================
st.markdown('<div class="section-eyebrow">Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">How The System Works</div>', unsafe_allow_html=True)

steps = ["Machine Data", "Feature Processing", "Random Forest AI", "Failure Risk"]
cols = st.columns([3, 0.4, 3, 0.4, 3, 0.4, 3])
step_idx = 0
for i, col in enumerate(cols):
    with col:
        if i % 2 == 0:
            step_idx += 1
            st.markdown(f"""
            <div class="step-tile">
                <div class="step-num">0{step_idx}</div>
                <div class="step-name">{steps[step_idx-1]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="step-arrow" style="text-align:center; padding-top:1.1rem;">→</div>', unsafe_allow_html=True)

st.markdown("<div style='height: 2.6rem'></div>", unsafe_allow_html=True)

# ======================================================
# MACHINE PARAMETERS EXPLANATION
# ======================================================
st.markdown('<div class="section-eyebrow">Reference</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Understanding The Machine Parameters</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
param_info = [
    ("Machine Type", "Product quality variant - L (Low), M (Medium), or H (High). Lower-quality variants tend to show a higher observed failure rate."),
    ("Air Temperature", "The ambient temperature of the air surrounding the machine, measured in Kelvin."),
    ("Process Temperature", "The temperature of the actual machining process itself, usually a few degrees above air temperature."),
    ("Rotational Speed", "How fast the tool/spindle is rotating, measured in revolutions per minute (rpm)."),
    ("Torque", "The rotational force applied during machining. Higher torque combined with lower speed is linked to increased failure risk."),
    ("Tool Wear", "Minutes the current cutting tool has been in use. Longer-used tools are associated with higher failure risk."),
]
for term, desc in param_info:
    st.markdown(f"""
    <div class="glossary-row">
        <div class="glossary-term">{term}</div>
        <div class="glossary-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height: 2.6rem'></div>", unsafe_allow_html=True)

# ======================================================
# ABOUT THE AI MODEL
# ======================================================
st.markdown('<div class="section-eyebrow">Model</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">About The AI Model</div>', unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
    <div class="card-title" style="margin-bottom:0.7rem;">Random Forest Classifier</div>
    <div style="color:#a9b8d6; font-size:0.9rem; line-height:1.7;">
        Random Forest combines the predictions of many individual decision trees, each trained on a random
        subset of the data, to reach a more stable and reliable final decision. This allows it to capture
        nonlinear relationships between machine parameters - for example, the combined effect of high torque,
        low rotational speed, and high tool wear - that a simpler linear model would miss.
        <br><br>
        It was selected as the final model for this project based on achieving the highest F1-score in a
        direct comparison against Logistic Regression and a single Decision Tree.
    </div>
</div>
""", unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<div class="app-footer">
    <div class="name">Predictive Maintenance AI</div>
    <div class="role">AI/ML Engineering Project</div>
    <div class="stack">Built with Python • Scikit-learn • Streamlit</div>
</div>
""", unsafe_allow_html=True)