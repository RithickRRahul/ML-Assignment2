import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

# Page setup
st.set_page_config(page_title="ML Model Evaluation", layout="centered")
st.title("ML Model Performance & Prediction App")
st.markdown("Built with 💡 using Streamlit")

# Load models
baseline_rf = joblib.load("baseline_rf.pkl")
tuned_rf = joblib.load("tuned_rf.pkl")
adaboost_model = joblib.load("adaboost_model.pkl")
bagging_model = joblib.load("bagging_model.pkl")
gradient_boosting = joblib.load("gradient_boosting.pkl")

# Load expected columns used during training
expected_columns = joblib.load("expected_columns.pkl")

# Load results
results_df = pd.read_csv("model_results.csv")

# --- Section 1: Model Performance ---
st.subheader("📊 Model Performance Comparison")

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=results_df, x="Model", y="Accuracy", palette="viridis", ax=ax)
plt.ylim(0, 1)
plt.title("Accuracy of Different Models")
st.pyplot(fig)

# --- Section 2: Predict on New Data ---
st.subheader("🔎 Predict on New Data")
uploaded_file = st.file_uploader("Upload a CSV file for prediction", type="csv")

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(test_df.head())

    # One-hot encode & align columns
    test_df_encoded = pd.get_dummies(test_df)
    test_df_encoded = test_df_encoded.reindex(columns=expected_columns, fill_value=0)

    selected_model = st.selectbox("Select a model to use for prediction", (
        "Baseline RF", "Tuned RF", "AdaBoost", "Bagging", "Gradient Boosting"))

    if st.button("Predict"):
        model_map = {
            "Baseline RF": baseline_rf,
            "Tuned RF": tuned_rf,
            "AdaBoost": adaboost_model,
            "Bagging": bagging_model,
            "Gradient Boosting": gradient_boosting
        }

        model = model_map[selected_model]

        try:
            preds = model.predict(test_df_encoded)
            st.write("✅ Predictions:")
            st.write(preds)
        except Exception as e:
            st.error(f"⚠️ Failed to predict: {str(e)}")

# Footer
st.markdown("---")
st.caption("Made by Rithick | 📬 rithick.r.rahul@RRRs-MacBook-Air")
