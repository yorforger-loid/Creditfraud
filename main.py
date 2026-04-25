from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from math import radians, cos, sin, asin, sqrt

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load all saved files
model       = joblib.load("fraud_model.pkl")
scaler      = joblib.load("scaler.pkl")
le_category = joblib.load("le_category.pkl")
le_gender   = joblib.load("le_gender.pkl")


class Transaction(BaseModel):
    amt: float          # Transaction amount
    category: str       # e.g. "grocery_pos", "shopping_net"
    gender: str         # "M" or "F"
    city_pop: int       # Population of city
    hour: int           # Hour of transaction (0-23)
    age: int            # Age of cardholder
    distance: float     # Distance from home to merchant in km


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(tx: Transaction):
    # Encode categoricals
    try:
        cat_enc = le_category.transform([tx.category])[0]
    except ValueError:
        return {"error": f"Unknown category: {tx.category}"}

    try:
        gen_enc = le_gender.transform([tx.gender])[0]
    except ValueError:
        return {"error": f"Unknown gender: {tx.gender}"}

    # Build feature array — must match training order:
    # ['amt', 'category_enc', 'gender_enc', 'city_pop', 'hour', 'age', 'distance']
    features = np.array([[
        tx.amt,
        cat_enc,
        gen_enc,
        tx.city_pop,
        tx.hour,
        tx.age,
        tx.distance
    ]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]  # fraud probability

    return {
        "prediction": int(prediction),
        "label": "Fraud" if prediction == 1 else "Normal",
        "fraud_probability": round(float(probability) * 100, 1)
    }


@app.get("/categories")
def get_categories():
    return {"categories": le_category.classes_.tolist()}
