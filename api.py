# File: api.py
# (Keep this as a separate file)

from flask import Flask, request, jsonify, send_from_directory # Added send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import traceback
import os # Needed for send_from_directory

app = Flask(__name__)
# CORS(app) # Keep CORS enabled - Handles requests if HTML is opened via file:///

# --- Model and Stats Loading ---
try:
    model = joblib.load("gradient_boosting.pkl")
    expected_columns = joblib.load("expected_columns.pkl")
    print("Gradient Boosting model and columns loaded successfully.")
    party_gender_win_rates = {
        "BJP_M": 60.2, "BJP_F": 55.1, "BJP_O": 48.3,
        "INC_M": 44.5, "INC_F": 38.2, "INC_O": 30.1,
    }
    print("Historical stats loaded.")
except FileNotFoundError as e:
    print(f"FATAL ERROR loading model/column file: {e}")
    model = None
    expected_columns = None
except Exception as e:
    print(f"FATAL ERROR during initial load: {e}")
    model = None
    expected_columns = None
# -----------------------------

# --- Route for serving the SINGLE index.html file ---
@app.route('/', methods=['GET'])
def serve_index():
    print("Serving index.html")
    try:
        # Serve index.html from the directory where api.py is run
        return send_from_directory('.', 'index.html')
    except Exception as e:
         print(f"Error serving index.html: {e}")
         return "Error loading page.", 404

# --- Predict route (remains mostly the same) ---
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle CORS Preflight
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    # Handle POST
    if request.method == 'POST':
        if not model or not expected_columns:
            return jsonify({"error": "Model not loaded properly on server"}), 500
        try:
            data = request.get_json()
            if data is None: return jsonify({"error": "Invalid JSON data received"}), 400
            print(f"Received data: {data}")

            # Extract data
            party = data.get('party', 'BJP')
            gender = data.get('gender', 'M')
            year = int(data.get('year', 2014))
            electors = int(data.get('electors', 1500000))
            totvotpoll = int(data.get('totvotpoll', 560000))
            selected_model_name = "Gradient Boosting" # Fixed model

            if electors <= 0: return jsonify({"error": "Total Electors > 0"}), 400
            vote_share = (totvotpoll / electors * 100) if electors > 0 else 0.0

            # Prepare DataFrame
            input_data = {col: 0 for col in expected_columns}
            input_data['year'] = year
            input_data['electors'] = electors
            input_data['totvotpoll'] = totvotpoll
            input_data['vote_share'] = vote_share
            party_col_name = f"partyabbre_{party}"
            gender_col_name = f"cand_sex_{gender}"
            if party_col_name in expected_columns: input_data[party_col_name] = 1
            else: print(f"Warning: Party column '{party_col_name}' not found.")
            if gender_col_name in expected_columns: input_data[gender_col_name] = 1
            else:
                 if gender != 'O': print(f"Warning: Gender column '{gender_col_name}' not found.")

            input_df = pd.DataFrame([input_data], columns=expected_columns)
            try:
                input_df = input_df.astype({'year': 'int64', 'totvotpoll': 'int64','electors': 'int64', 'vote_share': 'float64'}, errors='ignore')
            except Exception as type_e: print(f"Error setting dtypes: {type_e}")

            # Predict
            prediction = model.predict(input_df)[0]
            proba_win = None
            if hasattr(model, "predict_proba"):
                 probabilities = model.predict_proba(input_df)[0]
                 proba_win = float(probabilities[1])

            # Historical Rate
            combined_key = f"{party}_{gender}"
            hist_rate = party_gender_win_rates.get(combined_key, None)
            hist_rate_perc = float(hist_rate) if hist_rate is not None else None

            # Response
            response_data = {
                "prediction": int(prediction),
                "win_probability": proba_win,
                "outcome_label": "LIKELY TO WIN" if prediction == 1 else "LIKELY TO LOSE",
                "historical_rate_perc": hist_rate_perc,
                "selected_model": selected_model_name,
            }
            print(f"Sending response: {response_data}")
            return jsonify(response_data)

        except Exception as e:
            print(f"Error during prediction POST request: {e}")
            print(traceback.format_exc())
            return jsonify({"error": "Internal error during prediction."}), 500

# --- REMOVED the Waitress server runner for Cloud Deployment ---
# Instead, rely on Procfile/Dockerfile + Gunicorn/Waitress command

# --- Keep this for potential local testing if needed ---
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=8080)