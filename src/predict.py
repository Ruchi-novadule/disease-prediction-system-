import os
import joblib
import numpy as np

# ✅ Correct path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model", "rf_model.pkl")

# 📦 Load model
model = joblib.load(model_path)

# 📊 Feature names
feature_names = ["fever", "cough", "fatigue", "headache", "nausea", "body_pain", "breathlessness"]

def predict(symptoms_list):
    input_data = [1 if feature in symptoms_list else 0 for feature in feature_names]

    input_array = np.array(input_data).reshape(1, -1)

    prediction = model.predict(input_array)[0]
    probabilities = model.predict_proba(input_array)[0]

    top3_indices = probabilities.argsort()[-3:][::-1]
    top3_diseases = [model.classes_[i] for i in top3_indices]

    confidence = max(probabilities)

    return prediction, confidence, top3_diseases
log_prediction(symptoms, pred)