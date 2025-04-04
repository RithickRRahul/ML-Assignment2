import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Win Predictor", layout="centered")
st.title("🗳️ Indian Political Candidate Win Predictor")
st.markdown("Give the candidate's party and gender to predict the election outcome!")

# Load models
baseline_rf = joblib.load("baseline_rf.pkl")
tuned_rf = joblib.load("tuned_rf.pkl")
adaboost_model = joblib.load("adaboost_model.pkl")
bagging_model = joblib.load("bagging_model.pkl")
gradient_boosting = joblib.load("gradient_boosting.pkl")

# Load expected columns
expected_columns = joblib.load("expected_columns.pkl")

# User Inputs
party = st.selectbox("Party Abbreviation", ["BJP", "INC", "DMK", "AAAP", "CPI", "AITC", "BSP", "SP"])
gender = st.selectbox("Candidate Gender", ["M", "F", "O"])

# Model selection
selected_model = st.selectbox("Choose Model", (
    "Tuned RF", "Baseline RF", "AdaBoost", "Bagging", "Gradient Boosting"))

# Predict
if st.button("Predict"):
    # Create input dict with only party and gender one-hot encoded
    input_data = {
        f"partyabbre_{party}": 1,
        f"cand_sex_{gender}": 1
    }

    # Build dataframe, fill missing expected columns
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Map model names
    model_map = {
        "Tuned RF": tuned_rf,
        "Baseline RF": baseline_rf,
        "AdaBoost": adaboost_model,
        "Bagging": bagging_model,
        "Gradient Boosting": gradient_boosting
    }

    model = model_map[selected_model]

    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        if prediction == 1:
            st.success("✅ Likely to WIN")
        else:
            st.error("❌ Likely to LOSE")

        if proba is not None:
            st.markdown(f"**Confidence:** {proba * 100:.2f}%")

    except Exception as e:
        st.error(f"⚠️ Error making prediction: {e}")

# Footer
st.markdown("---")
st.caption("Made by Rithick | 📬 rithick.r.rahul@RRRs-MacBook-Air")
