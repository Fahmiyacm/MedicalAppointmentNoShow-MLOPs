import streamlit as st
import requests

st.set_page_config(
    page_title="Medical Appointment No-Show Prediction",
    page_icon="🏥"
)

st.title("🏥 Medical Appointment No-Show Prediction")
st.write("Enter the patient's appointment details to predict whether the patient is likely to miss the appointment.")

# FastAPI URL
API_URL = "http://localhost:8000/predict"

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

if st.button("Predict"):

    data = {
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
    }

    try:
        response = requests.post(
            API_URL,
            json=data
        )

        if response.status_code == 200:

            result = response.json()

            st.success("Prediction completed!")

            st.write(
                "### Prediction:",
                result["prediction_label"]
            )

            st.write(
                "### Probability:",
                f'{result["probability"]:.2%}'
            )

        else:
            st.error(
                f"API Error: {response.status_code}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to FastAPI. "
            "Please make sure the FastAPI server is running."
        )