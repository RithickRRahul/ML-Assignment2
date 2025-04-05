from flask import Flask, request, jsonify
from flask_cors import CORS # NEW: Import CORS
import pandas as pd
import joblib
import numpy as np
import traceback # For detailed error logging

app = Flask(__name__)
CORS(app) # NEW: Enable CORS for all routes on this app

# Load models and columns ONCE when the API starts
try:
    # Ensure these paths point to where your .pkl files actually are
    models = {
        "Tuned RF": joblib.load("tuned_rf.pkl"),
        "Baseline RF": joblib.load("baseline_rf.pkl"),
        "AdaBoost": joblib.load("adaboost_model.pkl"),
        "Bagging": joblib.load("bagging_model.pkl"),
        "Gradient Boosting": joblib.load("gradient_boosting.pkl")
    }
    expected_columns = joblib.load("expected_columns.pkl")
    print("Models and columns loaded successfully.")

    # Pre-load historical stats (or load from a file)
    party_gender_win_rates = {
        "BJP_M": 60.2, "BJP_F": 55.1, "BJP_O": 48.3,
        "INC_M": 44.5, "INC_F": 38.2, "INC_O": 30.1,
        # Add other combinations if they exist in your analysis
    }
    print("Historical stats loaded.")

except FileNotFoundError as e:
    print(f"FATAL ERROR loading model/column file: {e}")
    print("Ensure .pkl files are in the same directory as api.py or provide the correct path.")
    models = None
    expected_columns = None
except Exception as e:
    print(f"FATAL ERROR during initial load: {e}")
    models = None
    expected_columns = None

@app.route('/predict', methods=['POST', 'OPTIONS']) # NEW: Added OPTIONS method for CORS preflight
def predict():
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*', # Or specify your frontend origin e.g., 'http://127.0.0.1:8888'
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600' # Cache preflight response for 1 hour
        }
        return ('', 204, headers)

    # Handle actual POST request
    if request.method == 'POST':
        if not models or not expected_columns:
            return jsonify({"error": "Models not loaded properly on server"}), 500

        try:
            data = request.get_json()
            if data is None:
                return jsonify({"error": "Invalid JSON data received"}), 400
            print(f"Received data: {data}") # For debugging

            # --- Extract data (with defaults/validation if needed) ---
            party = data.get('party', 'BJP')
            gender = data.get('gender', 'M')
            year = int(data.get('year', 2014))
            electors = int(data.get('electors', 1500000))
            totvotpoll = int(data.get('totvotpoll', 560000))
            selected_model_name = data.get('selected_model', 'Gradient Boosting')

            if electors <= 0:
                 return jsonify({"error": "Total Electors must be greater than 0"}), 400

            vote_share = (totvotpoll / electors * 100) if electors > 0 else 0.0

            # --- Prepare input DataFrame (same logic as Streamlit) ---
            input_data = {col: 0 for col in expected_columns}
            # Set numerical features first
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
                 print(f"Warning: Party column '{party_col_name}' not found in expected columns.")
            if gender_col_name in expected_columns:
                input_data[gender_col_name] = 1
            else:
                # Only warn if M or F is missing, O might be legitimately absent
                if gender != 'O':
                     print(f"Warning: Gender column '{gender_col_name}' not found in expected columns.")

            # Build DataFrame
            input_df = pd.DataFrame([input_data], columns=expected_columns)
            # Set dtypes (important for consistency)
            try:
                input_df = input_df.astype({
                     'year': 'int64', 'totvotpoll': 'int64',
                     'electors': 'int64', 'vote_share': 'float64'
                     # Add other numeric dtypes if necessary
                 }, errors='ignore') # Ignore errors if a column wasn't set (e.g., missing gender/party)
            except Exception as type_e:
                 print(f"Error setting dtypes: {type_e}")
                 # Continue, but be aware types might be wrong

            # --- Select model and predict ---
            if selected_model_name not in models:
                 return jsonify({"error": f"Model '{selected_model_name}' not found"}), 400

            model = models[selected_model_name]

            prediction = model.predict(input_df)[0]
            proba_win = None
            if hasattr(model, "predict_proba"):
                 # Ensure numpy types are converted for JSON
                 probabilities = model.predict_proba(input_df)[0]
                 proba_win = float(probabilities[1]) # Probability of class 1 (Win)

            # --- Get historical rate for comparison ---
            combined_key = f"{party}_{gender}"
            hist_rate = party_gender_win_rates.get(combined_key, None)
            hist_rate_perc = float(hist_rate) if hist_rate is not None else None

            # --- Return results ---
            response_data = {
                "prediction": int(prediction),
                "win_probability": proba_win,
                "outcome_label": "LIKELY TO WIN" if prediction == 1 else "LIKELY TO LOSE",
                "historical_rate_perc": hist_rate_perc,
                "selected_model": selected_model_name,
            }
            print(f"Sending response: {response_data}") # For debugging
            return jsonify(response_data)

        except Exception as e:
            print(f"Error during prediction POST request: {e}")
            print(traceback.format_exc())
            return jsonify({"error": "An internal error occurred during prediction."}), 500

# Only run the Flask dev server if the script is executed directly
if __name__ == '__main__':
    # Run on 0.0.0.0 to be accessible from other devices on the network if needed,
    # or 127.0.0.1 for local only. Explicitly set port 5000.
    app.run(debug=True, host='127.0.0.1', port=5000)