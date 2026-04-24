from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("fraud_model.pkl")

@app.get("/")
def home():
    return {"message": "Fraud Detection API running"}

@app.post("/predict")
def predict(data: dict):
    input_data = np.array(data["input"]).reshape(1, -1)
    prediction = model.predict(input_data)
    
    return {"prediction": int(prediction[0])}
