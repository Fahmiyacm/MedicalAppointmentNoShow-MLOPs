from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_predict():
    payload = {
        "Gender": "F",
        "Age": 45,
        "Neighbourhood": "CENTRO",
        "Scholarship": 0,
        "Hipertension": 1,
        "Diabetes": 0,
        "Alcoholism": 0,
        "Handcap": 0,
        "SMS_received": 1,
        "days_until_appointment": 10,
        "scheduled_day_of_week": 1,
        "appointment_day_of_week": 3,
        "appointment_month": 5,
        "scheduled_hour": 10
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "prediction_label" in result
    assert "probability" in result

    assert result["prediction"] in [0, 1]
    assert 0 <= result["probability"] <= 1


def test_invalid_age():
    payload = {
        "Gender": "F",
        "Age": -5,
        "Neighbourhood": "CENTRO",
        "Scholarship": 0,
        "Hipertension": 1,
        "Diabetes": 0,
        "Alcoholism": 0,
        "Handcap": 0,
        "SMS_received": 1,
        "days_until_appointment": 10,
        "scheduled_day_of_week": 1,
        "appointment_day_of_week": 3,
        "appointment_month": 5,
        "scheduled_hour": 10
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422