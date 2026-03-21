import streamlit as st
import joblib
import numpy as np
import os

# Path fix
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "model", "rf_model.pkl")

# Load model
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

st.set_page_config(page_title="Disease Prediction", layout="wide")
st.title("🩺 Disease Prediction System")

# ✅ IMPORTANT: same order as training data
all_symptoms = [
    "fever", "cough", "fatigue", "headache",
    "nausea", "body_pain", "breathlessness"
]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Select Symptoms")
    symptoms = st.multiselect("Symptoms", all_symptoms)

    if st.button("Predict"):

        if not symptoms:
            st.warning("Please select at least one symptom")
        else:
            try:
                # Create input
                input_data = [1 if s in symptoms else 0 for s in all_symptoms]
                input_array = np.array(input_data).reshape(1, -1)

                # Prediction
                pred = model.predict(input_array)[0]
                prob = model.predict_proba(input_array)[0]

                st.success(f"Predicted Disease: {pred}")
                st.info(f"Confidence: {round(max(prob)*100,2)}%")

                # Top 3
                top3 = model.classes_[np.argsort(prob)[-3:][::-1]]
                st.write("Top 3 Diseases:")
                for d in top3:
                    st.write(d)

            except Exception as e:
                st.error(f"Prediction error: {e}")

with col2:
    st.subheader("About")
    st.write("Machine Learning based disease prediction system.")