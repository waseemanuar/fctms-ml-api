from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model, encoders, and scaler
model = joblib.load("fctms_rf_model.pkl")
encoders = joblib.load("fctms_label_encoders.pkl")
scaler = joblib.load("fctms_scaler.pkl")

@app.route("/")
def home():
    print("🏠 Home route was called")
    return "✅ Welcome to FCTMS Workout Plan API!"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return "🚫 Use POST method to get predictions."

    try:
        data = request.get_json()
        print("🟡 Received input:", data)

        # 🔧 Fix field name mismatch
        data["Workout_Frequency (days/week)"] = data["Workout_Frequency"]

        # 🔁 Encode categorical features
        gender_encoded = encoders["Gender"].transform([data["Gender"]])[0]
        type_encoded = encoders["Type_of_Exercise"].transform([data["Type_of_Exercise"]])[0]
        type_weighted = type_encoded * 0.2  # ✅ Weighted as per notebook
        age_group_encoded = encoders["Age_Group"].transform([data["Age_Group"]])[0]

        # ✅ Build features as per model
        features = [
            data["Age"],
            gender_encoded,
            data["BMI"],
            data["Fat_Percentage"],
            data["Session_Duration_Minutes"],
            data["Workout_Frequency (days/week)"],
            data["Experience_Level"],
            type_weighted,
            age_group_encoded
        ]

        print("🧪 Encoded features:", features)

        # Scale and predict
        features_scaled = scaler.transform([features])
        probs = model.predict_proba(features_scaled)[0]
        classes = model.classes_

        top_indices = np.argsort(probs)[::-1][:3]
        top_predictions = {
            encoders["Workout_Label"].inverse_transform([classes[i]])[0]: round(float(probs[i]), 3)
            for i in top_indices
        }

        prediction = model.predict(features_scaled)[0]
        label = encoders["Workout_Label"].inverse_transform([prediction])[0]

        print("🎯 Top predictions with confidence:", top_predictions)
        print("🔢 Predicted class:", prediction)
        print("✅ Predicted label:", label)

        return jsonify({
            "prediction": label,
            "top_3_predictions": top_predictions
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
