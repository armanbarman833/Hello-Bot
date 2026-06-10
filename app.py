from fastapi import FastAPI
import joblib

app = FastAPI()

# Load model
model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "ML API Running"}

@app.get("/predict/{hours}")
def predict(hours: float):

    prediction = model.predict([[hours]])

    return {
        "study_hours": hours,
        "predicted_marks": round(prediction[0], 2)
    }