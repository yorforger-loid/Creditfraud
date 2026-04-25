from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")   # ← load scaler

@app.get("/")
def home():
    return {"message": "Fraud Detection API running"}

@app.post("/predict")
def predict(data: dict):
    input_data = np.array(data["input"]).reshape(1, -1)
    
    # Scale Amount (index 29) before prediction
    input_data[0][29] = scaler.transform([[input_data[0][29]]])[0][0]
    
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}
