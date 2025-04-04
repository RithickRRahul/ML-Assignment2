import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Page setup
st.set_page_config(page_title="Election Prediction App", layout="centered")
st.title("🇮🇳 Indian Election Outcome Predictor")
st.markdown("Built with 💡 using Streamlit | 🗳️ Powered by ML Models")

# Load models
baseline_rf = joblib.load("baseline_rf.pkl")
tuned_rf = joblib.load("tuned_rf.pkl")
adaboost_model = joblib.load("adaboost_model.pkl")
bagging_model = joblib.load("bagging_model.pkl")
gradient_boosting = joblib.load("gradient_boosting.pkl")

# Load expected columns (from training)
expected_columns = joblib.load("expected_columns.pkl")

# Load results for comparison graph
results_df = pd.read_csv("model_results.csv")

# --- Section 1: Model Performance ---
st.subheader("📊 Model Accuracy Comparison")

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=results_df, x="Model", y="Accuracy", palette="mako", ax=ax)
plt.ylim(0, 1)
plt.title("Model Accuracies")
st.pyplot(fig)

# --- Section 2: Predict Election Outcome ---
st.subheader("🔎 Predict Election Outcome Based on Candidate Info")

with st.form("prediction_form"):
    party = st.selectbox("Party Abbreviation", ["BJP", "INC", "AAP", "DMK", "ADMK", "SP", "BSP", "CPI", "CPM", "NCP", "SHS", "OTH"])
    gender = st.selectbox("Gender", ["M", "F", "O"])
    age = st.slider("Candidate Age", min_value=25, max_value=90, value=45)
    education = st.selectbox("Education Level", ["10th", "12th", "Graduate", "Postgraduate", "Illiterate", "Doctorate"])
    criminal_cases = st.number_input("Number of Criminal Cases", min_value=0, step=1, value=0)
    assets = st.number_input("Declared Assets (in ₹ Crores)", min_value=0.0, step=0.1, value=1.0)

    selected_model = st.selectbox("Select Model", ["Tuned RF", "Baseline RF", "AdaBoost", "Bagging", "Gradient Boosting"])

    submit = st.form_submit_button("Predict")

if submit:
    # Construct input row
    input_data = pd.DataFrame([{
        "partyabbre": party,
        "cand_sex": gender,
        "age": age,
        "education": education,
        "criminal_cases": criminal_cases,
        "assets": assets
    }])

    # One-hot encode input and align with expected columns
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)

    model_map = {
        "Tuned RF": tuned_rf,
        "Baseline RF": baseline_rf,
        "AdaBoost": adaboost_model,
        "Bagging": bagging_model,
        "Gradient Boosting": gradient_boosting
    }

    model = model_map[selected_model]

    try:
        prediction = model.predict(input_encoded)[0]
        prob = model.predict_proba(input_encoded)[0][1] if hasattr(model, "predict_proba") else None

        result_text = "🟢 **Likely to WIN**" if prediction == 1 else "🔴 **Likely to LOSE**"
        st.markdown(f"### 🎯 Prediction: {result_text}")

        if prob is not None:
            st.markdown(f"**Confidence:** {prob*100:.2f}%")

    except Exception as e:
        st.error(f"⚠️ Error while predicting: {e}")

# Footer
st.markdown("---")
st.caption("Created by Rithick | 📬 rithick.r.rahul@RRRs-MacBook-Air")
