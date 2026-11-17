"""FastAPI app that serves the Telco churn pipeline saved by discovery.py.

Run with:  uv run fastapi dev main.py   then open http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ml = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the fitted pipeline once, when the app starts.
    ml["model"] = joblib.load("model.joblib")
    yield
    ml.clear()


app = FastAPI(
    title="Telco Churn Predictor",
    description="Predicts whether a telecom customer will churn (Yes/No).",
    lifespan=lifespan,
)

YesNo = Literal["Yes", "No"]
NoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1] = Field(description="1 if the customer is 65 or older")
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, description="Months the customer has been with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: NoInternet
    OnlineBackup: NoInternet
    DeviceProtection: NoInternet
    TechSupport: NoInternet
    StreamingTV: NoInternet
    StreamingMovies: NoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "No",
                    "Dependents": "No",
                    "tenure": 2,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "No",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 70.70,
                    "TotalCharges": 151.65,
                }
            ]
        }
    }


class Prediction(BaseModel):
    churn: str
    churn_probability: float


@app.get("/")
def root():
    return {"message": "Telco churn model is running. See /docs."}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    model = ml["model"]
    row = pd.DataFrame([customer.model_dump()])
    label = model.predict(row)[0]
    yes_index = list(model.classes_).index("Yes")
    proba = model.predict_proba(row)[0][yes_index]
    return Prediction(churn=str(label), churn_probability=round(float(proba), 4))
