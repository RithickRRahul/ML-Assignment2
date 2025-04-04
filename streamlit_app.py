import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Streamlit Page Config
st.set_page_config(page_title="ML Model Evaluation", layout="centered")
st.title("⚙️ ML Model Performance & Prediction App")
st.markdown("Built with 💡 using Streamlit")

# Load models and results
try:
    baseline_rf = joblib.load("baseline_rf.pkl")
    tuned_rf = joblib.load("tuned_rf.pkl")
    bagging_model = joblib.load("bagging_model.pkl")
    adaboost_model = joblib.load("adaboost_model.pkl")
    gradient_boosting = joblib.load("gradient_boosting.pkl")
    results_df = pd.read_csv("model_results.csv")
except Exception as e:
    st.error(f"❌ Error loading models or results: {e}")
    st.stop()

# Section 1: Accuracy Comparison
st.subheader("📊 Model Performance Comparison")

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=results_df, x="Model", y="Accuracy", palette="Set2", ax=ax)
ax.set_ylim(0.95, 1.0)
ax.set_title("Accuracy of Different ML Models")
st.pyplot(fig)

# Section 2: Predict on Custom Data
st.subheader("🔎 Predict on New Data")
uploaded_file = st.file_uploader("Upload a CSV file for prediction", type="csv")

if uploaded_file:
    try:
        test_df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview:")
        st.dataframe(test_df.head())

        selected_model = st.selectbox(
            "🤖 Select a model for prediction:",
            ("Baseline Random Forest", "Tuned Random Forest", "Bagging", "AdaBoost", "Gradient Boosting")
        )

        if st.button("Predict"):
            model = {
                "Baseline Random Forest": baseline_rf,
                "Tuned Random Forest": tuned_rf,
                "Bagging": bagging_model,
                "AdaBoost": adaboost_model,
                "Gradient Boosting": gradient_boosting
            }[selected_model]

            preds = model.predict(test_df)
            st.success("✅ Predictions Generated:")
            st.write(preds)
    except Exception as e:
        st.error(f"⚠️ Failed to predict: {e}")

# Footer
st.markdown("---")
st.caption("📬 Built by Rithick | rithick.r.rahul@RRRs-MacBook-Air")
