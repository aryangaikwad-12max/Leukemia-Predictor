# ==========================================
# app.py — Leukemia Prediction API
# ==========================================

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# -------------------------------
# 1️⃣ Define the FastAPI app
# -------------------------------
app = FastAPI(
    title="🩸 Leukemia Prediction API",
    description="A machine learning API that predicts leukemia based on CBC parameters.",
    version="1.0.0"
)

# -------------------------------
# 2️⃣ Load the trained model
# -------------------------------
model = joblib.load("leukemia_rf_model.pkl")

# -------------------------------
# 3️⃣ Define input schema
# -------------------------------
class CBCData(BaseModel):
    gender: int  # 0 = Male, 1 = Female
    WBC: float
    RBC: float
    Hemoglobin: float
    Platelet: float
    MCV: float
    MCH: float
    MCHC: float

# -------------------------------
# 4️⃣ Root endpoint
# -------------------------------
@app.get("/")
def home():
    return {"message": "🩸 Leukemia Prediction API is Running!"}

# -------------------------------
# 5️⃣ Prediction endpoint
# -------------------------------
@app.post("/predict")
def predict(data: CBCData):
    """
    Predict leukemia (1 = Leukemia Detected, 0 = Normal)
    """
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Human-readable output
    result = "Leukemia Detected" if prediction == 1 else "No Leukemia Detected"

    return {
        "prediction": int(prediction),
        "result": result,
        "input_data": data.dict()
    }

# -------------------------------
# 6️⃣ Example request for docs
# -------------------------------
@app.get("/example")
def example():
    return {
        "example_input": {
            "gender": 1,
            "WBC": 12.3,
            "RBC": 4.5,
            "Hemoglobin": 13.2,
            "Platelet": 250,
            "MCV": 88,
            "MCH": 30,
            "MCHC": 34
        },
        "endpoint": "/predict"
    }
