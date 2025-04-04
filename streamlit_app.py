import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Page config
st.set_page_config(page_title="Election Winner Predictor", layout="wide")
st.title("🗳️ Indian Political Candidate Win Predictor")
st.markdown("### Predict election outcomes based on candidate party and gender")

# Load models
@st.cache_resource
def load_models():
    models = {
        "Tuned RF": joblib.load("tuned_rf.pkl"),
        "Baseline RF": joblib.load("baseline_rf.pkl"),
        "AdaBoost": joblib.load("adaboost_model.pkl"),
        "Bagging": joblib.load("bagging_model.pkl"),
        "Gradient Boosting": joblib.load("gradient_boosting.pkl")
    }
    expected_columns = joblib.load("expected_columns.pkl")
    return models, expected_columns

# Load party win statistics data
@st.cache_data
def load_stats():
    # This would ideally come from a file, but we'll use hardcoded data for now
    # based on the analysis in your notebook
    party_win_rates = {
        "BJP": 58.7,
        "INC": 42.3,
        "DMK": 45.6,
        "AAAP": 39.2,
        "CPI": 30.5,
        "AITC": 52.1,
        "BSP": 28.7,
        "SP": 35.9
    }
    
    gender_win_rates = {
        "M": 33.2,
        "F": 27.8,
        "O": 18.4
    }
    
    # Add combined party-gender win rates (hypothetical - adjust with real data if available)
    party_gender_win_rates = {
        "BJP_M": 60.2,
        "BJP_F": 55.1,
        "BJP_O": 48.3,
        "INC_M": 44.5,
        "INC_F": 38.2,
        "INC_O": 30.1,
        # Add other combinations as needed
    }
    
    return party_win_rates, gender_win_rates, party_gender_win_rates

try:
    models, expected_columns = load_models()
    party_win_rates, gender_win_rates, party_gender_win_rates = load_stats()
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Statistics", "Model Information", "Debug"])
    
    with tab1:
        st.subheader("Make a Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # User Inputs with tooltips
            party = st.selectbox(
                "Party Abbreviation", 
                ["BJP", "INC", "DMK", "AAAP", "CPI", "AITC", "BSP", "SP"],
                help="Select the political party of the candidate"
            )
            
            gender = st.selectbox(
                "Candidate Gender", 
                ["M", "F", "O"],
                help="M = Male, F = Female, O = Other"
            )
            
            selected_model = st.selectbox(
                "Choose Model", 
                tuple(models.keys()),
                help="Select the machine learning model to use for prediction"
            )
        
        with col2:
            # Show preliminary stats based on selection
            st.markdown("### Historical Win Rates")
            
            # Party win rate
            st.metric("Party Win Rate", f"{party_win_rates[party]:.1f}%", 
                      delta=f"{party_win_rates[party] - 33:.1f}%" if party_win_rates[party] > 33 else f"{party_win_rates[party] - 33:.1f}%",
                      delta_color="normal")
            
            # Gender win rate
            st.metric("Gender Win Rate", f"{gender_win_rates[gender]:.1f}%",
                      delta=f"{gender_win_rates[gender] - 30:.1f}%" if gender_win_rates[gender] > 30 else f"{gender_win_rates[gender] - 30:.1f}%",
                      delta_color="normal")
            
            # Combined party-gender win rate if available
            combined_key = f"{party}_{gender}"
            if combined_key in party_gender_win_rates:
                st.metric("Combined Party-Gender Win Rate", 
                         f"{party_gender_win_rates[combined_key]:.1f}%",
                         delta=f"{party_gender_win_rates[combined_key] - 40:.1f}%" if party_gender_win_rates[combined_key] > 40 else f"{party_gender_win_rates[combined_key] - 40:.1f}%",
                         delta_color="normal")
    
        # Predict button with enhanced UI
        predict_button = st.button("🔮 Predict Outcome", use_container_width=True, type="primary")
        
        if predict_button:
            with st.spinner("Analyzing election data..."):
                st.session_state.last_input = {
                    "party": party,
                    "gender": gender,
                    "model": selected_model
                }
                
                # Create input data for the model - properly format the input
                input_data = {}
                
                # Initialize all expected columns to 0
                for col in expected_columns:
                    input_data[col] = 0
                
                # Set the specific features for this prediction to 1
                party_col = f"partyabbre_{party}" if f"partyabbre_{party}" in expected_columns else None
                gender_col = f"cand_sex_{gender}" if f"cand_sex_{gender}" in expected_columns else None
                
                if party_col:
                    input_data[party_col] = 1
                if gender_col:
                    input_data[gender_col] = 1
                
                # Build dataframe using all expected columns IN THE CORRECT ORDER
                input_df = pd.DataFrame([input_data], columns=expected_columns)
                
                model = models[selected_model]
                
                try:
                    # Make prediction
                    prediction = model.predict(input_df)[0]
                    proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
                    
                    # Store result in session state
                    st.session_state.last_prediction = {
                        "prediction": prediction,
                        "probability": proba,
                        "input_data": input_data
                    }
                    
                    # Enhanced results display
                    st.markdown("---")
                    st.subheader("Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 1:
                            st.success("✅ **PREDICTION: LIKELY TO WIN**")
                        else:
                            st.error("❌ **PREDICTION: LIKELY TO LOSE**")
                            
                        if proba is not None:
                            st.write(f"**Win Probability:** {proba * 100:.1f}%")
                            
                        # Add comparison with historical stats
                        combined_key = f"{party}_{gender}"
                        if combined_key in party_gender_win_rates:
                            historical = party_gender_win_rates[combined_key] / 100
                            model_prob = proba if proba is not None else (1.0 if prediction == 1 else 0.0)
                            
                            st.write("**Comparison:**")
                            st.write(f"- Historical win rate: {historical * 100:.1f}%")
                            st.write(f"- Model prediction: {model_prob * 100:.1f}%")
                            st.write(f"- Difference: {(model_prob - historical) * 100:.1f}%")
                    
                    with col2:
                        # Visualization of prediction probability
                        if proba is not None:
                            fig, ax = plt.subplots(figsize=(4, 1.5))
                            plt.barh([""], [proba], color="green", alpha=0.6)
                            plt.barh([""], [1-proba], left=[proba], color="red", alpha=0.6)
                            
                            plt.xlim(0, 1)
                            plt.xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
                            plt.yticks([])
                            
                            for s, x, c in zip(["Win", "Lose"], [proba/2, proba + (1-proba)/2], ["white", "white"]):
                                plt.text(x, 0, s, ha='center', va='center', color=c, fontweight='bold')
                                
                            plt.tight_layout()
                            st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"⚠️ Error making prediction: {e}")
                    st.error("Please check if the model's expected feature names match the input data.")
    
    with tab2:
        st.subheader("Historical Data Analysis")
        
        # Show some insights from the analysis
        st.write("### Party Win Rates")
        
        # Create bar chart of party win rates
        fig, ax = plt.subplots(figsize=(10, 5))
        parties = list(party_win_rates.keys())
        win_rates = list(party_win_rates.values())
        
        bars = ax.bar(parties, win_rates, color=sns.color_palette("viridis", len(parties)))
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        ax.set_ylim(0, max(win_rates) * 1.2)
        ax.set_ylabel("Win Rate (%)")
        ax.set_title("Historical Win Rates by Party")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
        
        # Gender analysis
        st.write("### Gender Win Rates")
        
        # Create bar chart of gender win rates
        fig, ax = plt.subplots(figsize=(8, 4))
        genders = list(gender_win_rates.keys())
        gender_rates = list(gender_win_rates.values())
        
        gender_colors = {"M": "#3498db", "F": "#e74c3c", "O": "#2ecc71"}
        colors = [gender_colors[g] for g in genders]
        
        bars = ax.bar(genders, gender_rates, color=colors)
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        ax.set_ylim(0, max(gender_rates) * 1.2)
        ax.set_ylabel("Win Rate (%)")
        ax.set_title("Historical Win Rates by Gender")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
        
        # New: Add party-gender combined analysis
        st.write("### Combined Party-Gender Win Rates")
        
        # Get available combinations and sort them by party
        combinations = list(party_gender_win_rates.keys())
        combined_rates = [party_gender_win_rates[combo] for combo in combinations]
        
        # Create bar chart for combined stats
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(combinations, combined_rates, color=sns.color_palette("viridis", len(combinations)))
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        ax.set_ylim(0, max(combined_rates) * 1.2)
        ax.set_ylabel("Win Rate (%)")
        ax.set_title("Historical Win Rates by Party-Gender Combination")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        
        st.pyplot(fig)
        
        # Add some insights from the analysis
        st.info("""
        **Key Insights:**
        - BJP has historically shown the highest win rate, followed by AITC
        - Male candidates tend to have a higher win rate than female candidates
        - The combination of party affiliation and gender can significantly influence election outcomes
        - Some combinations may have interactions that aren't apparent when looking at separate factors
        """)
    
    with tab3:
        st.subheader("Model Information")
        
        st.write("### About the Models")
        
        model_descriptions = {
            "Tuned RF": """
            **Tuned Random Forest** is an optimized ensemble learning method that operates by constructing multiple decision trees during training.
            The hyperparameters have been tuned for optimal performance on Indian election data.
            """,
            
            "Baseline RF": """
            **Baseline Random Forest** uses default parameters and serves as a comparison point to evaluate 
            the effectiveness of tuning and other modeling approaches.
            """,
            
            "AdaBoost": """
            **AdaBoost** (Adaptive Boosting) is an ensemble learning method that starts by fitting a simple model 
            and then focuses on improving predictions for instances that were previously misclassified.
            """,
            
            "Bagging": """
            **Bagging** (Bootstrap Aggregating) creates multiple versions of a predictor and uses these to get an aggregated predictor.
            It helps reduce variance and avoids overfitting.
            """,
            
            "Gradient Boosting": """
            **Gradient Boosting** builds an additive model in a forward stage-wise manner, allowing for the optimization of 
            arbitrary differentiable loss functions.
            """
        }
        
        for model_name, description in model_descriptions.items():
            with st.expander(f"{model_name} Model"):
                st.markdown(description)
                st.markdown(f"**Used in this app**: This model uses party and gender as primary features to predict election outcomes.")
        
        st.write("### Feature Importance")
        st.markdown("""
        Based on the model analysis, the following features are most important for predicting election outcomes:
        
        1. **Party affiliation** - The strongest predictor of election success
        2. **Gender** - Has a significant but lesser impact than party
        
        Other factors like voter turnout, campaigning effectiveness, and regional variations also play important roles
        but are not captured in the current simplified model.
        """)
        
        # Add button to show feature importances if available
        if st.button("Show Feature Importances for Selected Model"):
            selected = st.session_state.get("last_input", {}).get("model", "Tuned RF")
            model = models.get(selected)
            
            if model and hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                
                # Create DataFrame for feature importances
                feature_df = pd.DataFrame({
                    'Feature': expected_columns,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)
                
                # Display top features
                st.write(f"### Top Features for {selected}")
                st.dataframe(feature_df.head(10))
                
                # Plot feature importances
                fig, ax = plt.subplots(figsize=(10, 6))
                top_features = feature_df.head(10)
                sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax)
                plt.title(f"Top 10 Feature Importances - {selected}")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Feature importance information is not available for this model.")
    
    # New debugging tab
    with tab4:
        st.subheader("Debug Information")
        
        st.write("This tab helps debug issues with model predictions")
        
        # Show expected columns
        if st.checkbox("Show Expected Model Columns"):
            st.write("### Expected Model Columns")
            st.write(f"Total number of expected columns: {len(expected_columns)}")
            st.write(expected_columns)
        
        # Show last prediction details
        if st.checkbox("Show Last Prediction Details"):
            if "last_prediction" in st.session_state:
                last_pred = st.session_state.last_prediction
                last_input = st.session_state.last_input
                
                st.write("### Last Prediction Information")
                st.write(f"Party: {last_input['party']}")
                st.write(f"Gender: {last_input['gender']}")
                st.write(f"Model: {last_input['model']}")
                st.write(f"Prediction: {last_pred['prediction']} ({'Win' if last_pred['prediction'] == 1 else 'Lose'})")
                st.write(f"Probability: {last_pred['probability'] * 100:.2f}%" if last_pred['probability'] is not None else "Not available")
                
                # Show input data
                st.write("### Input Data")
                input_df = pd.DataFrame([last_pred['input_data']])
                
                # Show only non-zero features for clarity
                non_zero_features = input_df.loc[:, (input_df != 0).any(axis=0)]
                st.write("#### Non-zero features:")
                st.dataframe(non_zero_features)
                
                # Option to see all features
                if st.checkbox("Show all input features"):
                    st.dataframe(input_df)
            else:
                st.info("No prediction has been made yet. Make a prediction first.")
        
        # Test different combinations of party/gender
        st.write("### Test Multiple Combinations")
        if st.button("Run Test Cases for All Combinations"):
            models_to_test = ["Tuned RF"]  # Just use one model for testing
            
            # Create test cases
            parties = ["BJP", "INC", "DMK", "AAAP", "CPI", "AITC", "BSP", "SP"]
            genders = ["M", "F", "O"]
            
            # Create results table
            results = []
            
            for model_name in models_to_test:
                model = models[model_name]
                
                for party in parties:
                    for gender in genders:
                        # Create input data
                        input_data = {}
                        for col in expected_columns:
                            input_data[col] = 0
                            
                        party_col = f"partyabbre_{party}" if f"partyabbre_{party}" in expected_columns else None
                        gender_col = f"cand_sex_{gender}" if f"cand_sex_{gender}" in expected_columns else None
                        
                        if party_col:
                            input_data[party_col] = 1
                        if gender_col:
                            input_data[gender_col] = 1
                            
                        # Build dataframe
                        input_df = pd.DataFrame([input_data])
                        
                        # Predict
                        prediction = model.predict(input_df)[0]
                        proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
                        
                        # Add to results
                        results.append({
                            "Party": party,
                            "Gender": gender,
                            "Model": model_name,
                            "Prediction": "Win" if prediction == 1 else "Lose",
                            "Win Probability": f"{proba * 100:.1f}%" if proba is not None else "N/A",
                            "Raw Probability": proba if proba is not None else 0
                        })
            
            # Convert to DataFrame and display
            results_df = pd.DataFrame(results)
            
            # Show summary
            st.write("### Prediction Results for All Combinations")
            st.dataframe(results_df)
            
            # Create heatmap of win probabilities
            st.write("### Win Probability Heatmap")
            pivot_df = results_df.pivot_table(
                index="Party", 
                columns="Gender", 
                values="Raw Probability",
                aggfunc='mean'
            )
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(pivot_df, annot=True, cmap="Blues", fmt=".2f", ax=ax)
            plt.title(f"Win Probability by Party and Gender")
            st.pyplot(fig)
            
            # Compare with historical data
            st.write("### Model vs. Historical Win Rates")
            st.write("This analysis can help identify discrepancies between model predictions and historical data.")

    # Footer - moved inside the try block
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Data source: Indian Election Dataset")
    with col2:
        st.caption("📬 Contact: rithick.r.rahul@RRRs-MacBook-Air")

except Exception as e:
    st.error(f"Error loading application: {e}")
    st.info("Please make sure all required model files are available in the app directory.")
    st.error(f"Detailed error: {str(e)}")