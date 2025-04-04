import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Page config
st.set_page_config(page_title="Election Winner Predictor", layout="wide")
st.title("🗳️ Indian Political Candidate Win Predictor")
# MODIFIED: Updated description slightly
st.markdown("### Predict election outcomes based on candidate and constituency details")

# --- Load models and stats (no changes needed here) ---
@st.cache_resource
def load_models():
    # Make sure the paths to your saved model files are correct
    try:
        models = {
            "Tuned RF": joblib.load("tuned_rf.pkl"),
            "Baseline RF": joblib.load("baseline_rf.pkl"),
            "AdaBoost": joblib.load("adaboost_model.pkl"),
            "Bagging": joblib.load("bagging_model.pkl"),
            "Gradient Boosting": joblib.load("gradient_boosting.pkl")
        }
        expected_columns = joblib.load("expected_columns.pkl")
    except FileNotFoundError as e:
        st.error(f"Error loading model or column file: {e}")
        st.error("Ensure all .pkl files are in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred loading models: {e}")
        st.stop()

    # Check for required numerical columns in expected_columns
    required_numeric = ['year', 'totvotpoll', 'electors', 'vote_share']
    if not all(col in expected_columns for col in required_numeric):
        st.error(f"Error: Expected columns file is missing required numeric features: {required_numeric}. Please regenerate.")
        st.stop() # Stop execution if columns are missing
    return models, expected_columns

@st.cache_data
def load_stats():
    # This would ideally come from a file, but we'll use hardcoded data for now
    party_win_rates = {
        "BJP": 58.7, "INC": 42.3, "DMK": 45.6, "AAAP": 39.2,
        "CPI": 30.5, "AITC": 52.1, "BSP": 28.7, "SP": 35.9
    }
    gender_win_rates = {"M": 33.2, "F": 27.8, "O": 18.4}
    party_gender_win_rates = {
        "BJP_M": 60.2, "BJP_F": 55.1, "BJP_O": 48.3,
        "INC_M": 44.5, "INC_F": 38.2, "INC_O": 30.1,
    }
    return party_win_rates, gender_win_rates, party_gender_win_rates
# -------------------------------------------------------

try:
    models, expected_columns = load_models()
    party_win_rates, gender_win_rates, party_gender_win_rates = load_stats()

    tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Statistics", "Model Information", "Debug"])

    with tab1:
        st.subheader("Make a Prediction")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Candidate Details")
            party = st.selectbox(
                "Party Abbreviation",
                ["BJP", "INC", "DMK", "AAAP", "CPI", "AITC", "BSP", "SP"],
                index=0, # Default to BJP
                help="Select the political party of the candidate"
            )

            gender = st.selectbox(
                "Candidate Gender",
                ["M", "F", "O"],
                index=0, # Default to M
                help="M = Male, F = Female, O = Other"
            )

            # NEW: Add numerical inputs
            st.markdown("#### Constituency & Election Details")
            year = st.number_input(
                "Election Year",
                min_value=1977,
                max_value=2024, # Adjust max year if needed
                value=2014, # Default based on user's context
                step=1,
                help="Enter the year of the election"
            )

            electors = st.number_input(
                "Total Electors in Constituency",
                min_value=1000,
                value=1500000, # Example default, adjust based on typical constituency size
                step=1000,
                help="Estimated total number of registered voters in the constituency"
            )

            totvotpoll = st.number_input(
                "Total Votes Polled for Candidate",
                min_value=0,
                value=560000, # Example default, adjust based on user's test
                step=1000,
                help="Estimated total votes received by this candidate"
            )
            # Calculate vote_share dynamically
            # Avoid division by zero if electors is 0 or not set yet
            vote_share = (totvotpoll / electors * 100) if electors > 0 else 0.0
            # Display calculated vote share (optional, for user feedback)
            st.write(f"Calculated Vote Share: {vote_share:.2f}%")


            st.markdown("#### Model Selection")
            # --- INTEGRATED CODE: Start ---
            # Find the index of 'Gradient Boosting' in the keys list
            model_keys = list(models.keys())
            try:
                default_index = model_keys.index("Gradient Boosting")
            except ValueError:
                default_index = 0 # Default to first model if GB not found

            selected_model = st.selectbox(
                "Choose Model",
                tuple(model_keys),
                index=default_index, # Set default index
                help="Select the machine learning model to use for prediction. Gradient Boosting recommended." # Added recommendation
            )
            # --- INTEGRATED CODE: End ---

        with col2:
            st.markdown("### Historical Win Rates (Based on Party/Gender)")
            # Party win rate
            party_rate = party_win_rates.get(party, None) # Use get for safety
            if party_rate is not None:
                 st.metric("Party Win Rate", f"{party_rate:.1f}%",
                           delta=f"{party_rate - 33:.1f}%", delta_color="normal")
            # Gender win rate
            gender_rate = gender_win_rates.get(gender, None)
            if gender_rate is not None:
                st.metric("Gender Win Rate", f"{gender_rate:.1f}%",
                          delta=f"{gender_rate - 30:.1f}%", delta_color="normal")
            # Combined party-gender win rate
            combined_key = f"{party}_{gender}"
            combined_rate = party_gender_win_rates.get(combined_key, None)
            if combined_rate is not None:
                st.metric("Combined Party-Gender Win Rate",
                         f"{combined_rate:.1f}%",
                         delta=f"{combined_rate - 40:.1f}%", delta_color="normal")

            # Add a note about the importance of other factors
            st.info("ℹ️ Note: Historical rates are based only on Party/Gender. The model prediction considers other factors you provide.")


        predict_button = st.button("🔮 Predict Outcome", use_container_width=True, type="primary")

        if predict_button:
            # --- INTEGRATED CODE: Start ---
            # Add warning for Tuned RF if selected
            if selected_model == "Tuned RF":
                st.warning("⚠️ The Tuned RF model may produce 0% win probability for some inputs due to its specific training. Consider using other models like Gradient Boosting for potentially more robust predictions.")
            # --- INTEGRATED CODE: End ---

            with st.spinner("Analyzing election data..."):
                st.session_state.last_input = {
                    "party": party,
                    "gender": gender,
                    "model": selected_model,
                    "year": year,
                    "electors": electors,
                    "totvotpoll": totvotpoll,
                    "vote_share": vote_share
                }

                # Create input data for the model
                input_data = {}

                # Initialize all expected columns to 0
                for col in expected_columns:
                    input_data[col] = 0

                # Set numerical features based on user input
                input_data['year'] = year
                input_data['electors'] = electors
                input_data['totvotpoll'] = totvotpoll
                input_data['vote_share'] = vote_share

                # Set the specific categorical features for this prediction to 1
                party_col_name = f"partyabbre_{party}"
                gender_col_name = f"cand_sex_{gender}"

                # Check and set categorical columns
                found_party = False
                if party_col_name in expected_columns:
                    input_data[party_col_name] = 1
                    found_party = True
                else:
                    st.warning(f"Party column '{party_col_name}' not found in model features. Prediction might be inaccurate.")

                found_gender = False
                if gender_col_name in expected_columns:
                     input_data[gender_col_name] = 1
                     found_gender = True
                else:
                    # Handle cases where 'O' might not have been encoded or was dropped
                    if gender != 'O': # Only warn if M or F column is missing
                         st.warning(f"Gender column '{gender_col_name}' not found in model features. Prediction might be inaccurate.")

                # Build dataframe using all expected columns IN THE CORRECT ORDER
                try:
                    input_df = pd.DataFrame([input_data], columns=expected_columns)
                    # Ensure correct dtypes, especially for numerical columns
                    input_df = input_df.astype({
                         'year': 'int64',
                         'totvotpoll': 'int64',
                         'electors': 'int64',
                         'vote_share': 'float64'
                         # Add other numeric columns if needed, matching notebook dtypes
                     }, errors='ignore')
                except Exception as e:
                     st.error(f"Error creating input DataFrame: {e}")
                     st.error(f"Input data keys: {list(input_data.keys())}")
                     st.error(f"Expected columns (first 10): {expected_columns[:10]}")
                     st.stop()


                model = models[selected_model]

                try:
                    # Make prediction
                    prediction = model.predict(input_df)[0]
                    proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

                    # Store result in session state
                    st.session_state.last_prediction = {
                        "prediction": prediction,
                        "probability": proba,
                        "input_data": input_data # Store the dict used
                    }

                    # Enhanced results display
                    st.markdown("---")
                    st.subheader("Prediction Results")

                    col1_res, col2_res = st.columns(2)

                    with col1_res:
                        if prediction == 1:
                            st.success("✅ **PREDICTION: LIKELY TO WIN**")
                        else:
                            st.error("❌ **PREDICTION: LIKELY TO LOSE**")

                        if proba is not None:
                            st.write(f"**Win Probability:** {proba * 100:.1f}%")

                        # Add comparison with historical stats
                        combined_key = f"{party}_{gender}"
                        hist_rate = party_gender_win_rates.get(combined_key, None)
                        if hist_rate is not None:
                            historical = hist_rate / 100.0
                            model_prob = proba if proba is not None else (1.0 if prediction == 1 else 0.0)

                            st.write("**Comparison (Party/Gender Only):**")
                            st.write(f"- Historical win rate: {historical * 100:.1f}%")
                            st.write(f"- Model prediction probability: {model_prob * 100:.1f}%")
                            st.write(f"- Difference: {(model_prob - historical) * 100:.1f}%")
                        else:
                             st.write("_(No specific historical win rate for this Party/Gender combination)_")

                    with col2_res:
                        # Visualization of prediction probability
                        if proba is not None:
                            fig, ax = plt.subplots(figsize=(4, 1.5))
                            # Ensure proba is a valid number between 0 and 1
                            proba_val = max(0.0, min(1.0, proba))
                            lose_proba = 1.0 - proba_val

                            plt.barh([""], [proba_val], color="green", alpha=0.6)
                            plt.barh([""], [lose_proba], left=[proba_val], color="red", alpha=0.6)

                            plt.xlim(0, 1)
                            plt.xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
                            plt.yticks([])

                            # Adjust text placement based on probability value
                            win_text_pos = proba_val / 2
                            lose_text_pos = proba_val + lose_proba / 2

                            if proba_val > 0.1: # Only show 'Win' text if bar is wide enough
                                plt.text(win_text_pos, 0, "Win", ha='center', va='center', color='white', fontweight='bold')
                            if lose_proba > 0.1: # Only show 'Lose' text if bar is wide enough
                                plt.text(lose_text_pos, 0, "Lose", ha='center', va='center', color='white', fontweight='bold')

                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                             st.write("_(Probability visualization not available for this model)_")

                except Exception as e:
                    st.error(f"⚠️ Error making prediction: {e}")
                    # NEW: Provide more debug info
                    st.error("Please check if the model's expected features match the input data.")
                    st.error(f"Shape of input_df: {input_df.shape}")
                    st.error(f"Input DataFrame dtypes: {input_df.dtypes}")
                    st.dataframe(input_df.head()) # Show the dataframe being used


    # --- Tabs 2, 3, 4 remain largely the same ---

    with tab2:
        st.subheader("Historical Data Analysis")
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
        colors = [gender_colors.get(g, "#95a5a6") for g in genders] # Use get with default

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
        # ... (rest of tab3 code remains the same) ...
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
                # MODIFIED: Updated description to reflect new inputs
                st.markdown(f"**Used in this app**: This model uses party, gender, year, electors, and votes polled to predict election outcomes.")

        st.write("### Feature Importance")
        st.markdown("""
        Based on the model analysis, the following features are most important for predicting election outcomes:

        1.  **Numerical factors** like `vote_share`, `totvotpoll`, `electors` are highly influential.
        2.  **Party affiliation** - A strong predictor of election success.
        3.  **Gender** - Has a significant but lesser impact than party or numerical factors.
        4.  **Year** - Can capture temporal trends or shifts in voting patterns.
        5.  **State** (implicit in dummy variables) - Captures regional variations.

        Other factors like voter turnout dynamics (beyond electors), campaigning effectiveness, and specific local issues also play important roles
        but are not captured in the current model.
        """)

        # Add button to show feature importances if available
        if st.button("Show Feature Importances for Selected Model"):
            # MODIFIED: Use the current selected model, default to AdaBoost if none yet
            selected = st.session_state.get("last_input", {}).get("model", "Gradient Boosting") # Changed default for button
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
                st.dataframe(feature_df.head(15)) # Show top 15

                # Plot feature importances
                fig, ax = plt.subplots(figsize=(10, 8)) # Make plot taller
                top_features = feature_df.head(15) # Plot top 15
                sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax, palette="viridis")
                plt.title(f"Top 15 Feature Importances - {selected}")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning(f"Feature importance information is not available for the '{selected}' model.")


    with tab4:
        st.subheader("Debug Information")
        st.write("This tab helps debug issues with model predictions")

        # Show expected columns
        if st.checkbox("Show Expected Model Columns"):
            st.write("### Expected Model Columns")
            st.write(f"Total number of expected columns: {len(expected_columns)}")
            st.dataframe(pd.Series(expected_columns), width=800) # Display as a series for better readability

        # Show last prediction details
        if st.checkbox("Show Last Prediction Details"):
            if "last_prediction" in st.session_state and "last_input" in st.session_state:
                last_pred = st.session_state.last_prediction
                last_input = st.session_state.last_input

                st.write("### Last Prediction Information")
                st.write(f"**Inputs:**")
                st.write(f"- Party: `{last_input['party']}`")
                st.write(f"- Gender: `{last_input['gender']}`")
                st.write(f"- Year: `{last_input['year']}`")
                st.write(f"- Electors: `{last_input['electors']}`")
                st.write(f"- Votes Polled: `{last_input['totvotpoll']}`")
                st.write(f"- Vote Share: `{last_input['vote_share']:.2f}%`")
                st.write(f"- Model Used: `{last_input['model']}`")

                st.write(f"**Results:**")
                st.write(f"- Prediction: `{last_pred['prediction']} ({'Win' if last_pred['prediction'] == 1 else 'Lose'})`")
                st.write(f"- Probability: `{last_pred['probability'] * 100:.2f}%`" if last_pred['probability'] is not None else "Not available")

                # Show input data dictionary
                st.write("### Input Data Dictionary (Sent to Model)")
                # MODIFIED: Display the actual dict used to create the DataFrame
                input_data_dict = last_pred['input_data']

                # Show non-zero features first for clarity
                st.write("#### Non-zero features:")
                non_zero_dict = {k: v for k, v in input_data_dict.items() if v != 0}
                st.json(non_zero_dict, expanded=False) # Use json for better dict display

                # Option to see all features
                if st.checkbox("Show full input data dictionary"):
                    st.json(input_data_dict, expanded=False)

                # MODIFIED: Show the DataFrame created from the dict
                st.write("### Input DataFrame (Sent to Model - Head)")
                input_df_debug = pd.DataFrame([input_data_dict], columns=expected_columns)
                st.dataframe(input_df_debug.head())

            else:
                st.info("No prediction has been made yet in this session. Make a prediction first.")

        # Test multiple combinations of party/gender (This test will now be less useful without numericals)
        st.write("### Test Multiple Categorical Combinations (Numerical values fixed at 0)")
        st.warning("⚠️ This test uses default 0 for numerical features (year, electors, etc.), so predictions might not be realistic for models sensitive to these values.")
        if st.button("Run Test Cases (Categorical Only)"):
            # ... (rest of test case code remains the same, but acknowledge limitations) ...

            models_to_test = ["Gradient Boosting"]  # Test with the recommended model

            # Create test cases
            parties = ["BJP", "INC", "DMK", "AAAP", "CPI", "AITC", "BSP", "SP"]
            genders = ["M", "F", "O"]

            # Create results table
            results = []

            for model_name in models_to_test:
                model = models[model_name]

                for party_test in parties:
                    for gender_test in genders:
                        # Create input data - ALL ZERO except party/gender
                        input_data_test = {col: 0 for col in expected_columns}

                        party_col_name_test = f"partyabbre_{party_test}"
                        gender_col_name_test = f"cand_sex_{gender_test}"

                        if party_col_name_test in expected_columns:
                            input_data_test[party_col_name_test] = 1
                        if gender_col_name_test in expected_columns:
                             input_data_test[gender_col_name_test] = 1

                        # Build dataframe with correct columns
                        input_df_test = pd.DataFrame([input_data_test], columns=expected_columns)
                        # Ensure correct dtypes
                        input_df_test = input_df_test.astype({
                             'year': 'int64', 'totvotpoll': 'int64',
                             'electors': 'int64', 'vote_share': 'float64'
                         }, errors='ignore')

                        # Predict
                        try:
                            prediction_test = model.predict(input_df_test)[0]
                            proba_test = model.predict_proba(input_df_test)[0][1] if hasattr(model, "predict_proba") else None
                        except Exception as e_test:
                             st.error(f"Error testing {party_test}/{gender_test} with {model_name}: {e_test}")
                             prediction_test = "Error"
                             proba_test = None


                        # Add to results
                        results.append({
                            "Party": party_test,
                            "Gender": gender_test,
                            "Model": model_name,
                            "Prediction": "Win" if prediction_test == 1 else ("Lose" if prediction_test == 0 else "Error"),
                            "Win Probability": f"{proba_test * 100:.1f}%" if proba_test is not None else "N/A",
                            "Raw Probability": proba_test if proba_test is not None else None
                        })

            # Convert to DataFrame and display
            results_df = pd.DataFrame(results)

            # Show summary
            st.write("### Prediction Results for Categorical Combinations")
            st.dataframe(results_df)

            # Create heatmap of win probabilities
            st.write("### Win Probability Heatmap (Categorical Only)")
            # Filter out errors before pivoting
            results_df_valid = results_df[results_df["Raw Probability"].notna()]
            if not results_df_valid.empty:
                pivot_df = results_df_valid.pivot_table(
                    index="Party",
                    columns="Gender",
                    values="Raw Probability",
                    aggfunc='mean'
                )

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(pivot_df, annot=True, cmap="Blues", fmt=".2f", ax=ax, linewidths=.5)
                plt.title(f"Win Probability by Party and Gender ({models_to_test[0]}) - Numericals at 0")
                st.pyplot(fig)
            else:
                st.warning("Could not generate heatmap due to errors or missing probabilities.")

            # Compare with historical data
            st.write("### Model vs. Historical Win Rates")
            st.write("Note: Discrepancies are expected as numerical features are set to 0 in these tests.")


    # Footer
    st.markdown("---")
    col1_foot, col2_foot = st.columns(2)
    with col1_foot:
        st.caption("Data source: Indian Election Dataset (Processed)")
    with col2_foot:
        st.caption("App Version 1.2") # Increment version

except FileNotFoundError as e:
     st.error(f"Error loading model or column file: {e}")
     st.error("Please ensure all .pkl files are in the same directory as the Streamlit app.")
     st.info("If files are present, check file permissions.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.error(f"Detailed error: {str(e)}")
    import traceback
    st.text(traceback.format_exc())