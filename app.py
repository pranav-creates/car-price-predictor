import joblib
import pandas as pd
import streamlit as st

from datetime import date

model = joblib.load("car_price_model.pkl")
columns = list(joblib.load("columns.pkl"))

brands = ["Audi"] 
for c in columns:
    if c.startswith("brand_"):
        brands.append(c.replace("brand_",""))

st.title("Car Price Predictor")
brand = st.selectbox("Brand", sorted(brands))

today = date.today()

mnfg_year = st.number_input("Manufaturing Year",1995, today.year, today.year - 5)
mnfg_month = st.selectbox("Manufacturing Month",list(range(1,13)))

km_driven = st.number_input("KM Driven ", 0, 300000, 50000)
milege = st.number_input("Milege", 2, 45, 18)
engine = st.number_input("Engine (cc)", 500, 50000, 1200)
max_power = st.number_input("Max Power(bhp) ", 30, 600, 100)
seats = st.number_input("Total Seats ", 1, 10, 5)
seller_type = st.selectbox("Seller Type" , ["Dealer", "Individual", "Trustmark Dealer"])
fuel_type = st.selectbox("Fuel Type ", ["CNG", "Petrol", "Diesel", "Electric", "LPG"])
transmission =st.selectbox("Transmission", ["Automatic", "Manual"])



if st.button("Predict"):

    row = dict.fromkeys(columns, 0)

    months = (today.year - mnfg_year) * 12 + (today.month - mnfg_month)
    vehicle_age = max(round(months/12), 0)


    row["vehicle_age"] = vehicle_age
    row["milege"] = milege
    row["engine"] = engine
    row["km_driven"] = min(km_driven,180000)
    row["max_power"] = max_power
    row["seats"] = seats

    for key in [
        f"brand_{brand}",
        f"seller_type_{seller_type}",
        f"fuel_type_{fuel_type}",
        f"transmission_type_{transmission}"
    ]:
        if key in row:
            row[key]=1


    X = pd.DataFrame([row], columns = columns)
    price = model.predict(X)[0]
    st.success(f"Estimated Price Rs. {price :,.0f}")