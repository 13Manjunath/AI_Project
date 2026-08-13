from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load trained model
model = joblib.load("house_price_model.pkl")


# Input data structure
class HouseData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running"
    }


@app.post("/predict")
def predict(data: HouseData):

    # Convert input into DataFrame
    input_data = pd.DataFrame({
        "MedInc": [data.MedInc],
        "HouseAge": [data.HouseAge],
        "AveRooms": [data.AveRooms],
        "AveBedrms": [data.AveBedrms],
        "Population": [data.Population],
        "AveOccup": [data.AveOccup],
        "Latitude": [data.Latitude],
        "Longitude": [data.Longitude]
    })

    # Prediction
    prediction = model.predict(input_data)

    return {
        "predicted_house_value": float(prediction[0])
    }
