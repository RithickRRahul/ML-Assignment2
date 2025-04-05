from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import traceback
from waitress import serve # NEW: Import serve from waitress
# import multiprocessing # REMOVED: Not strictly needed for basic waitress run

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- MODIFIED: Load only the required model and columns ---
try:
    model = joblib.load("gradient_boosting.pkl") # Load GB model directly
    expected_columns = joblib.load("expected_columns.pkl")
    print("Gradient Boosting model and columns loaded successfully.")

    # Pre-load historical stats (or load from a file)
    party_gender_win_rates = {
        "BJP_M": 60.2, "BJP_F": 55.1, "BJP_O": 48.3,
        "INC_M": 44.5, "INC_F": 38.2, "INC_O": 30.1,
    }
    print("Historical stats loaded.")

except FileNotFoundError as e:
    print(f"FATAL ERROR loading model/column file: {e}")
    print("Ensure 'gradient_boosting.pkl' and 'expected_columns.pkl' are in the same directory as api.py.")
    model = None
    expected_columns = None
except Exception as e:
    print(f"FATAL ERROR during initial load: {e}")
    model = None
    expected_columns = None
# ----------------------------------------------------------

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    # Handle actual POST request
    if request.method == 'POST':
        # MODIFIED: Check the single loaded model
        if not model or not expected_columns:
            return jsonify({"error": "Model not loaded properly on server"}), 500

        try:
            data = request.get_json()
            if data is None:
                return jsonify({"error": "Invalid JSON data received"}), 400
            print(f"Received data: {data}") # For debugging

            # --- Extract data ---
            party = data.get('party', 'BJP')
            gender = data.get('gender', 'M')
            year = int(data.get('year', 2014))
            electors = int(data.get('electors', 1500000))
            totvotpoll = int(data.get('totvotpoll', 560000))
            # selected_model_name = data.get('selected_model', 'Gradient Boosting') # REMOVED: Model is fixed

            if electors <= 0:
                 return jsonify({"error": "Total Electors must be greater than 0"}), 400

            vote_share = (totvotpoll / electors * 100) if electors > 0 else 0.0

            # --- Prepare input DataFrame ---
            input_data = {col: 0 for col in expected_columns}
            input_data['year'] = year
            input_data['electors'] = electors
            input_data['totvotpoll'] = totvotpoll
            input_data['vote_share'] = vote_share

            party_col_name = f"partyabbre_{party}"
            gender_col_name = f"cand_sex_{gender}"

            if party_col_name in expected_columns:
                input_data[party_col_name] = 1
            else:
                 print(f"Warning: Party column '{party_col_name}' not found.")
            if gender_col_name in expected_columns:
                input_data[gender_col_name] = 1
            else:
                 if gender != 'O':
                      print(f"Warning: Gender column '{gender_col_name}' not found.")

            # Build DataFrame
            input_df = pd.DataFrame([input_data], columns=expected_columns)
            try:
                input_df = input_df.astype({
                     'year': 'int64', 'totvotpoll': 'int64',
                     'electors': 'int64', 'vote_share': 'float64'
                 }, errors='ignore')
            except Exception as type_e:
                 print(f"Error setting dtypes: {type_e}")

            # --- Predict using the loaded Gradient Boosting model ---
            # REMOVED: Model selection logic
            # if selected_model_name not in models: ...
            # model = models[selected_model_name]

            prediction = model.predict(input_df)[0]
            proba_win = None
            if hasattr(model, "predict_proba"):
                 probabilities = model.predict_proba(input_df)[0]
                 proba_win = float(probabilities[1])

            # --- Historical Rate ---
            combined_key = f"{party}_{gender}"
            hist_rate = party_gender_win_rates.get(combined_key, None)
            hist_rate_perc = float(hist_rate) if hist_rate is not None else None

            # --- Response ---
            response_data = {
                "prediction": int(prediction),
                "win_probability": proba_win,
                "outcome_label": "LIKELY TO WIN" if prediction == 1 else "LIKELY TO LOSE",
                "historical_rate_perc": hist_rate_perc,
                "selected_model": "Gradient Boosting", # Return the fixed model name
            }
            print(f"Sending response: {response_data}")
            return jsonify(response_data)

        except Exception as e:
            print(f"Error during prediction POST request: {e}")
            print(traceback.format_exc())
            return jsonify({"error": "An internal error occurred during prediction."}), 500

# MODIFIED: Run Waitress server instead of Flask dev server
if __name__ == '__main__':
    if model and expected_columns: # Check if model loaded successfully
        host = '0.0.0.0' # Listen on all network interfaces
        port = 8080      # Use port 8080 for Waitress
        print(f"Starting Waitress server for Flask app on http://{host}:{port}")
        try:
            serve(app, host=host, port=port) # Removed threads for simplicity
        except Exception as e:
             print(f"Error starting Waitress server: {e}")
    else:
        print("Model/Columns could not be loaded. Server not starting.")