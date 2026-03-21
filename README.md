# 🩺 Disease Prediction System

An end-to-end Machine Learning project that predicts diseases based on user symptoms.  
This system uses a trained Random Forest model and provides real-time predictions through an interactive Streamlit dashboard.

---

## 🚀 Live Demo
(Add your deployed app link here after deployment)

---

## 📌 Features

- Predict disease based on selected symptoms
- Displays prediction confidence score
- Shows top 3 probable diseases
- Interactive and user-friendly UI
- Real-time predictions using trained ML model

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Machine Learning:** Scikit-learn (Random Forest)  
- **Frontend/UI:** Streamlit  
- **Backend (optional):** FastAPI  
- **Libraries:** Pandas, NumPy, Matplotlib, Joblib  

---

## 📂 Project Structure
disease-prediction-system/
│
├── data/ # Dataset files
├── model/ # Trained model (.pkl)
├── src/ # ML training & prediction scripts
├── api/ # FastAPI backend
├── ui/ # Streamlit frontend
│ └── app.py
│
├── requirements.txt # Dependencies
├── .gitignore # Ignored files
└── README.md # Project documentation


---

## ⚙️ How It Works

1. User selects symptoms from the UI  
2. Symptoms are converted into numerical format (0/1 encoding)  
3. The trained Random Forest model processes the input  
4. System predicts:
   - Disease name  
   - Confidence score  
   - Top 3 possible diseases  

---

## ▶️ Run Locally

### 1️⃣ Clone the repository
git clone https://github.com/Ruchi-novadule/disease-prediction-system-.git

cd disease-prediction-system

---

### 2️⃣ Install dependencies
pip install -r requirements.txt


---

### 3️⃣ Run the application
streamlit run ui/app.py

---

## 📊 Model Details

- Algorithm: Random Forest Classifier  
- Input Features:
  - fever
  - cough
  - fatigue
  - headache
  - nausea
  - body_pain
  - breathlessness  

- Output:
  - Predicted Disease
  - Confidence Score
  - Top 3 Predictions  

---

## 📸 Screenshot

(Add your app screenshot here)

Example:

---

## 🔮 Future Improvements

- Add more diseases & symptoms  
- Improve model accuracy with larger dataset  
- Deploy FastAPI backend separately  
- Add SHAP for explainable AI  
- User authentication system  

## 🔍 Additional Features
- Explainable predictions (based on selected symptoms)
- Logging system for monitoring predictions
- Clean and interactive UI
---

## 👩‍💻 Author

**Ruchi Tiwari**  
- Aspiring Data Analyst / ML Enthusiast  

---

## ⭐ Acknowledgement

This project is built for learning and demonstrating Machine Learning, API development, and UI integration in a real-world use case.