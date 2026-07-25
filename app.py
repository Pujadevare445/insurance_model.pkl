import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained Decision Tree Regressor model
MODEL_PATH = "decisioninsurance_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# HTML Template with Embedded CSS styling
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Insurance Premium Estimator</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --input-bg: #0f172a;
            --border-color: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-color: #38bdf8;
            --accent-hover: #0284c7;
            --success-color: #10b981;
            --error-color: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
            width: 100%;
            max-width: 650px;
            padding: 2.5rem;
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 1.875rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.25rem;
        }

        @media (max-width: 580px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .form-group.full-width {
            grid-column: 1 / -1;
        }

        label {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-secondary);
        }

        input, select {
            background-color: var(--input-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            padding: 0.75rem 1rem;
            font-size: 1rem;
            transition: border-color 0.2s, box-shadow 0.2s;
            outline: none;
        }

        input:focus, select:focus {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15);
        }

        .btn-submit {
            background-color: var(--accent-color);
            color: #0f172a;
            border: none;
            border-radius: 8px;
            padding: 0.875rem;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            transition: background-color 0.2s, transform 0.1s;
            margin-top: 1rem;
        }

        .btn-submit:hover {
            background-color: var(--accent-hover);
            color: #ffffff;
        }

        .btn-submit:active {
            transform: scale(0.99);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.25rem;
            border-radius: 8px;
            text-align: center;
            background-color: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--success-color);
        }

        .result-box h2 {
            font-size: 1rem;
            color: var(--text-secondary);
            font-weight: 500;
            margin-bottom: 0.25rem;
        }

        .result-box .amount {
            font-size: 2rem;
            font-weight: 700;
            color: var(--success-color);
        }

        .error-box {
            margin-top: 1.5rem;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            background-color: rgba(239, 68, 68, 0.1);
            border: 1px solid var(--error-color);
            color: var(--error-color);
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Insurance Cost Predictor</h1>
            <p>Enter individual parameters below to estimate medical insurance charges.</p>
        </div>

        {% if error_message %}
            <div class="error-box">
                {{ error_message }}
            </div>
        {% endif %}

        <form action="/predict" method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label for="age">Age</label>
                    <input type="number" id="age" name="age" min="1" max="100" placeholder="e.g. 28" required value="{{ inputs.age if inputs }}">
                </div>

                <div class="form-group">
                    <label for="sex">Sex</label>
                    <select id="sex" name="sex" required>
                        <option value="1" {% if inputs and inputs.sex == '1' %}selected{% endif %}>Male</option>
                        <option value="0" {% if inputs and inputs.sex == '0' %}selected{% endif %}>Female</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="bmi">BMI (Body Mass Index)</label>
                    <input type="number" step="0.1" id="bmi" name="bmi" min="10" max="60" placeholder="e.g. 26.5" required value="{{ inputs.bmi if inputs }}">
                </div>

                <div class="form-group">
                    <label for="children">Number of Children</label>
                    <input type="number" id="children" name="children" min="0" max="10" placeholder="e.g. 1" required value="{{ inputs.children if inputs }}">
                </div>

                <div class="form-group">
                    <label for="smoker">Smoker Status</label>
                    <select id="smoker" name="smoker" required>
                        <option value="0" {% if inputs and inputs.smoker == '0' %}selected{% endif %}>No</option>
                        <option value="1" {% if inputs and inputs.smoker == '1' %}selected{% endif %}>Yes</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="region">Region</label>
                    <select id="region" name="region" required>
                        <option value="0" {% if inputs and inputs.region == '0' %}selected{% endif %}>Northeast</option>
                        <option value="1" {% if inputs and inputs.region == '1' %}selected{% endif %}>Northwest</option>
                        <option value="2" {% if inputs and inputs.region == '2' %}selected{% endif %}>Southeast</option>
                        <option value="3" {% if inputs and inputs.region == '3' %}selected{% endif %}>Southwest</option>
                    </select>
                </div>

                <div class="form-group full-width">
                    <button type="submit" class="btn-submit">Calculate Insurance Premium</button>
                </div>
            </div>
        </form>

        {% if prediction_text %}
            <div class="result-box">
                <h2>Estimated Insurance Cost</h2>
                <div class="amount">${{ prediction_text }}</div>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_LAYOUT)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(
            HTML_LAYOUT, 
            error_message="Model file missing or failed to load correctly."
        )

    try:
        # Extract form parameters
        form_data = request.form
        age = float(form_data["age"])
        sex = int(form_data["sex"])
        bmi = float(form_data["bmi"])
        children = int(form_data["children"])
        smoker = int(form_data["smoker"])
        region = int(form_data["region"])

        # Arrange features matching model expectations
        features = np.array([[age, sex, bmi, children, smoker, region]])
        
        # Predict price
        prediction = model.predict(features)[0]
        formatted_prediction = f"{prediction:,.2f}"

        return render_template_string(
            HTML_LAYOUT, 
            prediction_text=formatted_prediction,
            inputs=form_data
        )
    except Exception as e:
        return render_template_string(
            HTML_LAYOUT, 
            error_message=f"Prediction Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
