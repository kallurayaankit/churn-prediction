from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI()

# --- Load the model and preprocessing objects from local files ---
# (These files are copied into the Docker image at /app/)
scaler = joblib.load("preprocess/scaler.pkl")
numeric_cols = joblib.load("preprocess/numeric_cols.pkl")
feature_columns = joblib.load("preprocess/feature_columns.pkl")
label_encoders = joblib.load("preprocess/label_encoders.pkl")
multi_cat_cols = joblib.load("preprocess/multi_cat_cols.pkl")
binary_cols = joblib.load("preprocess/binary_cols.pkl")
model = joblib.load("model/model.pkl")

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict(customer: CustomerData):
    df = pd.DataFrame([customer.dict()])

    # Preprocess exactly as during training
    for col in binary_cols:
        if col == 'Churn':
            continue
        df[col] = label_encoders[col].transform(df[col])

    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Predict
    prediction = model.predict(df)
    churn = bool(prediction[0])
    return {"churn": churn}