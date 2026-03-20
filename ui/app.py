import streamlit as st
import requests
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Disease Prediction", layout="wide")

# Title
st.title("🩺 Disease Prediction Dashboard")

# Symptom list (same as training data)
all_symptoms = [
    "fever", "cough", "fatigue", "headache",
    "nausea", "body_pain", "breathlessness"
]

# Layout
col1, col2 = st.columns(2)

# ---------------- LEFT SIDE ----------------
with col1:
    st.subheader("Enter Symptoms")

    symptoms = st.multiselect(
        "Select Symptoms",
        all_symptoms
    )

    if st.button("Predict"):

        if len(symptoms) == 0:
            st.warning("Please select at least one symptom")
        else:
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json={"symptoms": symptoms}
                )

                result = response.json()

                # Prediction
                st.success(f"Predicted Disease: {result['prediction']}")

                # Confidence
                st.info(f"Confidence: {round(result['confidence']*100,2)}%")

                # Top 3
                st.write("Top 3 Possible Diseases:")
                for i, d in enumerate(result["top_3"], 1):
                    st.write(f"{i}. {d}")

                # Simple chart (visual feel)
                st.subheader("Feature Impact (Demo Graph)")
                features = all_symptoms[:len(symptoms)]
                values = [1/len(features)] * len(features)

                fig, ax = plt.subplots()
                ax.barh(features, values)
                st.pyplot(fig)

            except Exception as e:
                st.error("API Error: Make sure FastAPI is running")

# ---------------- RIGHT SIDE ----------------
with col2:
    st.subheader("API Response")

    if 'result' in locals():
        st.code(result, language="json")