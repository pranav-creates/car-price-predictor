# Car Price Predictor

Predicts the selling price of a used car from its details (brand, age, engine, power, fuel type and more). It is built with Python and scikit-learn. It includes a Streamlit web app and a FastAPI service.

## Results

| Model | Test R² |
|---|---|
| Linear Regression (raw price) | 0.78 |
| Linear Regression (log price) | 0.83 |
| Random Forest (log price, tuned) | 0.93 |

### 5-fold cross-validation (R²)

| Model | Mean | Std |
|---|---|---|
| Linear Regression | 0.806 | 0.026 |
| Random Forest | 0.935 | 0.004 |

Random Forest scores higher, and its low std means the result is stable across folds.

## What this project shows

- **Data cleaning:** dropped index and high-cardinality columns, capped `km_driven` at the 99th percentile (180,000), removed the top 1% of prices.
- **Encoding:** one-hot encoding with `get_dummies(drop_first=True)`.
- **Model comparison:** Linear Regression vs Random Forest, with a train vs test R² check for over and underfitting.
- **Log transform of price:** reduces skew and improves Linear Regression a lot. Applied with `TransformedTargetRegressor` (`log1p` / `expm1`).
- **Hyperparameter tuning:** `RandomizedSearchCV` on Random Forest.
- **Cross-validation:** 5-fold `KFold` to check the model is stable.
- **Feature importance:** `max_power`, `vehicle_age` and `engine` matter most.
- **Deployment:** model saved with `joblib` and served through Streamlit and FastAPI.

## Best parameters

```
n_estimators=200, max_depth=30, min_samples_split=7,
min_samples_leaf=1, max_features=0.5
```

## Project structure

```
car_price_predictor.ipynb   # data cleaning, modelling, tuning, evaluation
app.py                      # Streamlit web app
api.py                      # FastAPI service
car_price_model.pkl         # saved model
columns.pkl                 # training column order
cardekho_dataset.csv        # dataset
```

## Dataset

CarDekho used car dataset (from Kaggle).


## How to run

Install the libraries:

```
pip install pandas numpy scikit-learn matplotlib joblib streamlit fastapi uvicorn
```

Run the notebook: open `car_price_predictor.ipynb` and choose **Run All**.

Run the web app:

```
streamlit run app.py
```

Run the API:

```
uvicorn api:app --reload
```

Then open `http://127.0.0.1:8000/docs` to try the `/predict` route.

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib, joblib, Streamlit, FastAPI