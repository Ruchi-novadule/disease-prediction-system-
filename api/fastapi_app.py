from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# ✅ src folder ko path me add karo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import predict

# ✅ FastAPI app create
app = FastAPI()

# 📥 Input schema
class Symptoms(BaseModel):
    symptoms: list

# 🔮 API endpoint
@app.post("/predict")
def get_prediction(data: Symptoms):
    prediction, confidence, top3 = predict(data.symptoms)

    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "top_3": top3
    }