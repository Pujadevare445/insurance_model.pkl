import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Model path
MODEL_PATH = "insurance_model.pkl"

# Load Trained Scikit-Learn Model
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model file not found on server."}), 500

    try:
        # Extract features from form input
        age = float(request.form.get("age"))
        sex = float(request.form.get("sex"))
        bmi = float(request.form.get("bmi"))
        children = float(request.form.get("children"))
        smoker = float(request.form.get("smoker"))
        region = float(request.form.get("region"))

        # Feature array order matching feature_names_in_:
        # ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        features = np.array([[age, sex, bmi, children, smoker, region]])

        # Model prediction
        prediction = model.predict(features)[0]

        return jsonify(
            {
                "success": True,
                "prediction_text": f"${prediction:,.2f}",
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
