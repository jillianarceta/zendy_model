from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("zendy_model.pkl")

@app.get("/")
def home():
    return {"message": "Zendy API working"}

@app.post("/predict")
def predict(data: dict):
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]

    return {
        "prediction": int(prediction),
        "label": "At Risk" if prediction == 1 else "Not at Risk"
    }