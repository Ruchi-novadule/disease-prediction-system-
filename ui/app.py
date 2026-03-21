import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "model", "rf_model.pkl")

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error("❌ Model file not found. Please check deployment.")
    st.stop()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🩺 AI-Powered Disease Prediction System")
st.markdown("### Predict diseases based on symptoms using Machine Learning")
st.markdown("---")

# ---------------- SYMPTOMS ----------------
all_symptoms = [
    "fever", "cough", "fatigue", "headache",
    "nausea", "body_pain", "breathlessness"
]

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([2, 1])

# ================= LEFT SIDE =================
with col1:
    st.subheader("🧾 Select Your Symptoms")

    symptoms = st.multiselect(
        "Choose symptoms:",
        all_symptoms,
        placeholder="Select one or more symptoms"
    )

    if st.button("🔍 Predict Disease", use_container_width=True):

        if len(symptoms) == 0:
            st.warning("⚠️ Please select at least one symptom")

        else:
            try:
                # Prepare input
                input_data = [1 if s in symptoms else 0 for s in all_symptoms]
                input_data = np.array(input_data).reshape(1, -1)

                # Prediction
                pred = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0]

                # ---------------- RESULTS ----------------
                st.markdown("---")
                st.success(f"🩺 Predicted Disease: {pred}")
                st.info(f"📊 Confidence: {round(max(prob)*100, 2)}%")

                # ---------------- TOP 3 ----------------
                st.subheader("📌 Top 3 Possible Diseases")
                top3 = model.classes_[np.argsort(prob)[-3:][::-1]]

                for d in top3:
                    st.success(f"👉 {d}")

                # ---------------- GRAPH ----------------
                st.markdown("---")
                st.subheader("📊 Prediction Probabilities")

                prob_df = pd.DataFrame({
                    "Disease": model.classes_,
                    "Probability": prob
                })

                st.bar_chart(prob_df.set_index("Disease"))

                # ---------------- EXPLANATION ----------------
                st.markdown("---")
                st.subheader("🧠 Why this prediction?")

                st.write("The model predicted based on the selected symptoms:")

                for s in symptoms:
                    st.write(f"✔ {s}")

            except Exception as e:
                st.error(f"❌ Prediction error: {e}")

# ================= RIGHT SIDE =================
with col2:
    st.subheader("ℹ️ About App")

    st.write("""
    This is a Machine Learning-based disease prediction system.

    🔹 Enter symptoms  
    🔹 Get instant prediction  
    🔹 View confidence score  
    🔹 See top 3 possible diseases  

    Built using:
    - Python  
    - Scikit-learn  
    - Streamlit  
    """)

    st.markdown("---")

    st.subheader("⚠️ Disclaimer")
    st.write("This app is for educational purposes only and not a medical diagnosis.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("© 2026 AI Disease Prediction System | Built with ❤️ by Ruchi Tiwari")