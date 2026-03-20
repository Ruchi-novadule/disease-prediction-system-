import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model/rf_model.pkl")

st.set_page_config(page_title="Disease Prediction", layout="wide")

st.title("🩺 Disease Prediction System")

# Symptoms list
all_symptoms = [
    "fever", "cough", "fatigue", "headache",
    "nausea", "body_pain", "breathlessness"
]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Select Symptoms")

    symptoms = st.multiselect("Symptoms", all_symptoms)

    if st.button("Predict"):

        if len(symptoms) == 0:
            st.warning("Please select at least one symptom")
        else:
            input_data = [1 if s in symptoms else 0 for s in all_symptoms]
            input_data = np.array(input_data).reshape(1, -1)

            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0]

            st.success(f"Predicted Disease: {pred}")
            st.info(f"Confidence: {round(max(prob)*100,2)}%")

            top3 = model.classes_[np.argsort(prob)[-3:][::-1]]
            st.write("Top 3 Diseases:")
            for d in top3:
                st.write(d)

with col2:
    st.subheader("About")
    st.write("Machine Learning based disease prediction system.")