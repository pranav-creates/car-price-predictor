from fastapi import FastAPI
import joblib

app=FastAPI()

model = joblib.load("car_price_model.pkl")
columns = joblib.load("columns.pkl")

@app.get("/")
def home():
    return {"status" : "ok"}


from pydantic import BaseModel
from datetime import date
import pandas as pd

class CarInput(BaseModel):
    mnfg_year : int
    mnfg_month : int
    mileage : float
    engine : float
    km_driven : int
    max_power : int
    seats : int
    brand : str
    seller_type : str
    fuel_type : str
    transmission_type : str

@app.post("/predict")
def predict(car : CarInput):
    today=date.today()
    months = (today.year - car.mnfg_year)*12 + (today.month - car.mnfg_month)

    row = dict.fromkeys(columns, 0)
    row["vehicle_age"] = max(round(months/12), 0)
    row["mileage"] = car.mileage
    row["engine"] = car.engine
    row["km_driven"] = min(car.km_driven, 180000)
    row["max_power"] = car.max_power
    row["seats"] = car.seats

    for key in [
        f"brand_{car.brand}",
        f"seller_type_{car.seller_type}",
        f"fuel_type_{car.fuel_type}",
        f"transmission_type_{car.transmission_type}",
    ]:
        if key in row:
            row[key]=1;


    df=pd.DataFrame([row])[columns]
    price=model.predict(df)[0]
    return {"predict_price" : float(price)}
