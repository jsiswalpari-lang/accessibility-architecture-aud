from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(title="Real-Time ML Inference API")

model = joblib.load("model.pkl")

class PredictionInput(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "ML API is running"}

@app.post("/predict")
def predict(data: PredictionInput):
    prediction = model.predict_proba([data.features])
    return {"probabilities": prediction.tolist()}
