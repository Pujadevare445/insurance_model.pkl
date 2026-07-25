import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Model loading with error handling
MODEL_PATH = "decisioninsurance_model.pkl"

model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: '{MODEL_PATH}' not found in current directory.")

# Feature Encoding Mappings
SEX_MAP = {"female": 0, "male": 1}
SMOKER_MAP = {"no": 0, "yes": 1}
REGION_MAP = {
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model file missing or failed to load."}), 500

    try:
        data = request.get_json() if request.is_json else request.form

        # Extract features
        age = float(data.get("age", 30))
        sex = SEX_MAP.get(str(data.get("sex")).lower(), 0)
        bmi = float(data.get("bmi", 25.0))
        children = int(data.get("children", 0))
        smoker = SMOKER_MAP.get(str(data.get("smoker")).lower(), 0)
        region = REGION_MAP.get(str(data.get("region")).lower(), 0)

        # Feature array order: ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        features = np.array([[age, sex, bmi, children, smoker, region]])
        
        # Make Prediction
        prediction = model.predict(features)[0]
        formatted_price = f"${prediction:,.2f}"

        return jsonify({
            "success": True,
            "prediction": formatted_price,
            "raw_value": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
