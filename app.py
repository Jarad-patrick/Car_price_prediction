from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd 

model = joblib.load('car_price_prediction_model.pkl')


app = FastAPI(title = 'Car prediction API')

class Car(BaseModel):
    levy: int
    manufacturer: str
    model: str
    prod_year: int
    category: str
    engine_volume: float
    is_turbo: int
    mileage: int
    gear_box_type: str
    drive_wheels: str
    doors: str
    wheel: str
    colour: str
    airbags: int

@app.post("/predict")
def predict_price(car: Car):
    data = pd.DataFrame([car.dict()])
    prediction = model.predict(data)[0]

    return {"predicted_price": round(float(prediction), 2)}