# Medical Appointment No-Show Prediction – MLOps Project

## Project Overview

This project implements a simplified MLOps pipeline for predicting whether a patient is likely to miss a scheduled medical appointment.

The project demonstrates the complete machine learning workflow including:

- Machine learning model training
- Data preprocessing
- Model serialization
- FastAPI REST API
- Unit testing with Pytest
- Continuous Integration using GitHub Actions
- Docker containerization
- Docker Hub image publishing
- Streamlit user interface
- Cloud deployment of the Streamlit application

The trained model predicts whether an appointment will be:

- `Show`
- `No-show`

It also provides the probability of a no-show prediction.

---

# 1. Project Objectives

The main objectives of this project are:

1. Build a machine learning model for medical appointment no-show prediction.
2. Store the project using Git and GitHub.
3. Create a FastAPI API to serve the trained model.
4. Write automated tests for the API.
5. Configure GitHub Actions for Continuous Integration.
6. Containerize the application using Docker.
7. Build and publish the Docker image to Docker Hub.
8. Create a Streamlit interface for interacting with the model.
9. Deploy the Streamlit application to the cloud.

---

# 2. Technologies Used

## Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

## API

- FastAPI
- Uvicorn
- Pydantic

## Testing

- Pytest
- FastAPI TestClient
- HTTPX

## MLOps / DevOps

- Git
- GitHub
- GitHub Actions
- Docker
- Docker Hub

## User Interface

- Streamlit

---

# 3. Dataset

The project uses the Medical Appointment No-Show dataset.

The dataset contains information about patients and their scheduled appointments.

Important features include:

- Gender
- Age
- Neighbourhood
- Scholarship
- Hypertension
- Diabetes
- Alcoholism
- Handicap
- SMS Received
- Days until appointment
- Scheduled day of week
- Appointment day of week
- Appointment month
- Scheduled hour

The target variable indicates whether the patient attended the appointment or did not attend.

---

# 4. Machine Learning Model

The machine learning pipeline performs preprocessing and prediction.

The preprocessing includes handling numerical and categorical features.

The `Neighbourhood` column is encoded using One-Hot Encoding.

Unknown neighbourhood values are handled using:

python
OneHotEncoder(handle_unknown="ignore")  



# 5. Project Structure

MedicalAppointmentNoShow-MLOPs/
│
├── .github/
│   └── workflows/
│       └── test.yml
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   └── raw/
│       └── Medical.csv
│
├── src/
│   ├── train.py
│   └── models/
│       └── model_pipeline.pkl
│
├── test/
│   └── test_api.py
│
├── streamlit_app.py
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md

# 6. Step 1 – Version Control
# 6. Step 1 – Version Control

Git and GitHub are used for version control of the project.

The project source code, machine learning code, API, tests, Docker configuration, and Streamlit application are maintained in a public GitHub repository.

## GitHub Repository

**Repository:**  
https://github.com/Fahmiyacm/MedicalAppointmentNoShow-MLOPs

**Main Branch:**

```text
main


# Docker Containerization

Docker is used to containerize the Medical Appointment No-Show Prediction API.

Containerization packages the application, Python runtime, dependencies, source code, and trained machine learning model into a portable Docker image.

This allows the same application to run on another computer or server without manually installing all project dependencies.

---

## Docker Architecture

The Docker workflow used in this project is:

```text
Dockerfile
    ↓
docker build
    ↓
Docker Image
    ↓
docker run
    ↓
Docker Container
    ↓
FastAPI Application
    ↓
Machine Learning Model
    ↓
Prediction