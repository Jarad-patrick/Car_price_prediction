from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd 

model = joblib.load('car_price_prediction.pkl')


app = FastAPI(title = 'Car prediction API')

from pydantic import BaseModel

class Car(BaseModel):
    # numerical features
    levy: int
    prod_year: int
    cylinders: int
    airbags: int
    engine_volume: float
    mileage: int
    is_turbo: int

    # categorical features
    manufacturer: str
    model: str
    category: str
    gear_box_type: str
    drive_wheels: str
    doors: str
    wheel: str
    color: str
    leather_interior: str
    fuel_type: str


@app.post("/predict")
def predict_price(car: Car):
    data = pd.DataFrame([car.dict()])
    prediction = model.predict(data)[0]

    return {"predicted_price": round(float(prediction), 2)}