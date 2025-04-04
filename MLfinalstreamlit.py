import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Page config
st.set_page_config(page_title="Election Winner Predictor", layout="wide")
st.title("🗳️ Indian Political Candidate Win Predictor")
st.markdown("### Predict election outcomes based on candidate and constituency details")

# --- Load models and stats ---
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

    # --- Prediction Section ---
    st.subheader("Make a Prediction")

    # Use columns for layout
    col1_input, col2_input = st.columns(2)

    with col1_input:
        st.markdown("#### Candidate Details")
        party = st.selectbox(
            "Party Abbreviation",
            ["BJP", "INC", "DMK", "AAAP", "CPI", "AITC", "BSP", "SP"],
            index=0 # Default to BJP
        )

        gender = st.selectbox(
            "Candidate Gender",
            ["M", "F", "O"],
            index=0 # Default to M
        )

        st.markdown("#### Constituency & Election Details")
        year = st.number_input(
            "Election Year",
            min_value=1977,
            max_value=2024,
            value=2014,
            step=1,
        )

        electors = st.number_input(
            "Total voters in Constituency",
            min_value=1000,
            value=1500000,
            step=1000,
        )

        totvotpoll = st.number_input(
            "Total Votes Polled for Candidate",
            min_value=0,
            value=560000,
            step=1000,
        )
        # Calculate vote_share dynamically
        vote_share = (totvotpoll / electors * 100) if electors > 0 else 0.0
        st.write(f"Calculated Vote Share: {vote_share:.2f}%")

    with col2_input:
        st.markdown("#### Model Selection")
        model_keys = list(models.keys())
        try:
            default_index = model_keys.index("Gradient Boosting")
        except ValueError:
            default_index = 0 # Default to first model if GB not found

        selected_model = st.selectbox(
            "Choose Model (Choose Gradient Boosting for best results)",
            tuple(model_keys),
            index=default_index,
        )

    predict_button = st.button("🔮 Predict Outcome", use_container_width=True, type="primary")

    # --- Prediction Results Display ---
    if predict_button:
        if selected_model == "Tuned RF":
            st.warning("⚠️ The Tuned RF model may produce 0% win probability for some inputs due to its specific training. Consider using other models like Gradient Boosting for potentially more robust predictions.")

        with st.spinner("Analyzing election data..."):
            # Store inputs for potential later use (removed from debug tab display)
            st.session_state.last_input = {
                "party": party, "gender": gender, "model": selected_model,
                "year": year, "electors": electors, "totvotpoll": totvotpoll,
                "vote_share": vote_share
            }

            # Create input data for the model
            input_data = {col: 0 for col in expected_columns} # Initialize all to 0

            # Set numerical features
            input_data['year'] = year
            input_data['electors'] = electors
            input_data['totvotpoll'] = totvotpoll
            input_data['vote_share'] = vote_share

            # Set categorical features
            party_col_name = f"partyabbre_{party}"
            gender_col_name = f"cand_sex_{gender}"

            if party_col_name in expected_columns:
                input_data[party_col_name] = 1
            else:
                st.warning(f"Party column '{party_col_name}' not found.")

            if gender_col_name in expected_columns:
                 input_data[gender_col_name] = 1
            else:
                 if gender != 'O':
                      st.warning(f"Gender column '{gender_col_name}' not found.")

            # Build dataframe with correct columns and dtypes
            try:
                input_df = pd.DataFrame([input_data], columns=expected_columns)
                input_df = input_df.astype({
                     'year': 'int64', 'totvotpoll': 'int64',
                     'electors': 'int64', 'vote_share': 'float64'
                 }, errors='ignore')
            except Exception as e:
                 st.error(f"Error creating input DataFrame: {e}")
                 st.stop()

            # Select model and predict
            model = models[selected_model]
            try:
                prediction = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

                # Store prediction results (removed from debug tab display)
                st.session_state.last_prediction = {
                    "prediction": prediction, "probability": proba, "input_data": input_data
                }

                # Display results
                st.markdown("---")
                st.subheader("Prediction Results")

                if prediction == 1:
                    st.success("✅ **PREDICTION: LIKELY TO WIN**")
                else:
                    st.error("❌ **PREDICTION: LIKELY TO LOSE**")

                if proba is not None:
                    st.write(f"**Win Probability:** {proba * 100:.1f}%")

                # Comparison with historical stats
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

            except Exception as e:
                st.error(f"⚠️ Error making prediction: {e}")
                st.dataframe(input_df.head()) # Show df for debugging


    # --- Statistics Section (Moved from Tab2) ---
    st.markdown("---") # Add a separator
    st.subheader("Historical Data Analysis")

    st.write("### Party Win Rates")
    fig_party, ax_party = plt.subplots(figsize=(10, 5))
    parties = list(party_win_rates.keys())
    win_rates = list(party_win_rates.values())
    bars_party = ax_party.bar(parties, win_rates, color=sns.color_palette("viridis", len(parties)))
    for bar in bars_party:
        height = bar.get_height()
        ax_party.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}%', ha='center', va='bottom')
    ax_party.set_ylim(0, max(win_rates) * 1.2 if win_rates else 100)
    ax_party.set_ylabel("Win Rate (%)")
    ax_party.set_title("Historical Win Rates by Party")
    ax_party.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig_party)

    st.write("### Gender Win Rates")
    fig_gender, ax_gender = plt.subplots(figsize=(8, 4))
    genders = list(gender_win_rates.keys())
    gender_rates = list(gender_win_rates.values())
    gender_colors = {"M": "#3498db", "F": "#e74c3c", "O": "#2ecc71"}
    colors = [gender_colors.get(g, "#95a5a6") for g in genders]
    bars_gender = ax_gender.bar(genders, gender_rates, color=colors)
    for bar in bars_gender:
        height = bar.get_height()
        ax_gender.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}%', ha='center', va='bottom')
    ax_gender.set_ylim(0, max(gender_rates) * 1.2 if gender_rates else 100)
    ax_gender.set_ylabel("Win Rate (%)")
    ax_gender.set_title("Historical Win Rates by Gender")
    ax_gender.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig_gender)

    st.write("### Combined Party-Gender Win Rates")
    combinations = list(party_gender_win_rates.keys())
    combined_rates = [party_gender_win_rates[combo] for combo in combinations]
    fig_combo, ax_combo = plt.subplots(figsize=(12, 6))
    bars_combo = ax_combo.bar(combinations, combined_rates, color=sns.color_palette("viridis", len(combinations)))
    for bar in bars_combo:
        height = bar.get_height()
        ax_combo.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}%', ha='center', va='bottom')
    ax_combo.set_ylim(0, max(combined_rates) * 1.2 if combined_rates else 100)
    ax_combo.set_ylabel("Win Rate (%)")
    ax_combo.set_title("Historical Win Rates by Party-Gender Combination")
    ax_combo.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    st.pyplot(fig_combo)

    st.info("""
    **Key Insights:**
    - BJP has historically shown the highest win rate, followed by AITC
    - Male candidates tend to have a higher win rate than female candidates
    - The combination of party affiliation and gender can significantly influence election outcomes
    - Some combinations may have interactions that aren't apparent when looking at separate factors
    """)

    # --- Footer ---
    st.markdown("---")
    col1_foot, col2_foot = st.columns(2)
    with col1_foot:
        st.caption("Data source: Indian Election Dataset (Processed)")
    with col2_foot:
        st.caption("App Version 2.0 - Simplified") # Increment version

except FileNotFoundError as e:
     st.error(f"Error loading model or column file: {e}")
     st.error("Please ensure all .pkl files are in the same directory as the Streamlit app.")
     st.info("If files are present, check file permissions.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.error(f"Detailed error: {str(e)}")
    import traceback
    st.text(traceback.format_exc())