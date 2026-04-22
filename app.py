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
    try:
        features = np.array(data["features"]).reshape(1, -1)

        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]

        return {
            "prediction": int(prediction),
            "label": "At Risk" if prediction == 1 else "Not at Risk",
            "confidence": float(proba)
        }

    except Exception as e:
        return {"error": str(e)}
