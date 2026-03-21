import logging

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_prediction(symptoms, prediction):
    logging.info(f"Symptoms: {symptoms} -> Prediction: {prediction}")