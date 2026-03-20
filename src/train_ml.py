import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

from preprocess import load_data, preprocess_data

# 📥 Load data
df = load_data("../data/raw/dataset.csv")
df = preprocess_data(df)

# 🎯 Features & Target
X = df.drop("disease", axis=1)
y = df["disease"]
# 🤖 Model
model = RandomForestClassifier()
model.fit(X, y)

# 💾 Save model
joblib.dump(model, "../model/rf_model.pkl")

print("✅ Model trained and saved successfully!")