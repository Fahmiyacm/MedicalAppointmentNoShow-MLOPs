import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Medical Appointment No-Show Prediction",
    page_icon="🏥",
    layout="centered"
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent / "src" / "models" / "model_pipeline.pkl"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🏥 Medical Appointment No-Show Prediction")

st.write(
    "Enter the patient's appointment details to predict "
    "whether the patient is likely to miss the appointment."
)


# --------------------------------------------------
# Patient information
# --------------------------------------------------

st.subheader("Patient Information")

gender = st.selectbox(
    "Gender",
    ["F", "M"]
)

age = st.number_input(
    "Age",
    min_value=0,
    max_value=120,
    value=30
)

neighbourhood = st.text_input(
    "Neighbourhood",
    value="JARDIM DA PENHA"
)

scholarship = st.selectbox(
    "Scholarship",
    [0, 1]
)

hypertension = st.selectbox(
    "Hypertension",
    [0, 1]
)

diabetes = st.selectbox(
    "Diabetes",
    [0, 1]
)

alcoholism = st.selectbox(
    "Alcoholism",
    [0, 1]
)

handicap = st.selectbox(
    "Handicap",
    [0, 1]
)

sms_received = st.selectbox(
    "SMS Received",
    [0, 1]
)

days_until_appointment = st.number_input(
    "Days Until Appointment",
    min_value=0,
    value=10
)

scheduled_day_of_week = st.number_input(
    "Scheduled Day of Week",
    min_value=0,
    max_value=6,
    value=6
)

appointment_day_of_week = st.number_input(
    "Appointment Day of Week",
    min_value=0,
    max_value=6,
    value=6
)

appointment_month = st.number_input(
    "Appointment Month",
    min_value=1,
    max_value=12,
    value=1
)

scheduled_hour = st.number_input(
    "Scheduled Hour",
    min_value=0,
    max_value=23,
    value=10
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict", type="primary"):

    input_data = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Neighbourhood": neighbourhood,
        "Scholarship": scholarship,
        "Hipertension": hypertension,
        "Diabetes": diabetes,
        "Alcoholism": alcoholism,
        "Handcap": handicap,
        "SMS_received": sms_received,
        "days_until_appointment": days_until_appointment,
        "scheduled_day_of_week": scheduled_day_of_week,
        "appointment_day_of_week": appointment_day_of_week,
        "appointment_month": appointment_month,
        "scheduled_hour": scheduled_hour
    }])

    try:

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:

            st.error("Prediction: No-show")

        else:

            st.success("Prediction: Show")

        st.metric(
            "No-show Probability",
            f"{probability:.2%}"
        )

    except Exception as e:

        st.error(f"Prediction error: {e}")
