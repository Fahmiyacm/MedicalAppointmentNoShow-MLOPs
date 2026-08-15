from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Medical Appointment No-Show Prediction API",
    description="Predicts whether a patient is likely to miss a medical appointment.",
    version="1.0.0",
)


# --------------------------------------------------
# Load trained ML pipeline
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "src" / "models" / "model_pipeline.pkl"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class PatientData(BaseModel):
    Gender: str = Field(..., description="Patient gender: F or M")
    Age: int = Field(..., ge=0, description="Patient age")
    Neighbourhood: str = Field(..., description="Appointment neighbourhood")

    Scholarship: int = Field(..., ge=0, le=1)
    Hipertension: int = Field(..., ge=0, le=1)
    Diabetes: int = Field(..., ge=0, le=1)
    Alcoholism: int = Field(..., ge=0, le=1)
    Handcap: int = Field(..., ge=0, le=1)
    SMS_received: int = Field(..., ge=0, le=1)

    days_until_appointment: int = Field(
        ...,
        ge=0,
        description="Number of days between scheduling and appointment",
    )

    scheduled_day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description="0=Monday, 6=Sunday",
    )

    appointment_day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description="0=Monday, 6=Sunday",
    )

    appointment_month: int = Field(
        ...,
        ge=1,
        le=12,
    )

    scheduled_hour: int = Field(
        ...,
        ge=0,
        le=23,
    )


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Medical Appointment No-Show Prediction API is running"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(data: PatientData):

    # Convert Pydantic object to dictionary
    input_data = pd.DataFrame([data.model_dump()])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability of no-show
    probability = model.predict_proba(input_data)[0][1]

    return {
        "prediction": int(prediction),
        "prediction_label": (
            "No-show" if prediction == 1 else "Attended"
        ),
        "probability": round(float(probability), 4),
    }