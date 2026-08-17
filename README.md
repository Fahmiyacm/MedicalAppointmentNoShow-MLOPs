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

Git and GitHub are used for version control of the project.

The project source code, machine learning code, API, tests, Docker configuration, and Streamlit application are maintained in a public GitHub repository.

## GitHub Repository

**Repository:**  
https://github.com/Fahmiyacm/MedicalAppointmentNoShow-MLOPs

**Main Branch:**

```text
main

#7. FastAPI Model API

FastAPI is used to expose the trained machine learning model as a REST API.

The API provides the following endpoints.

Home Endpoint
GET /

This endpoint confirms that the API is running.

Prediction Endpoint
POST /predict

The user sends patient information as JSON.

Example request:

{
  "Gender": "F",
  "Age": 30,
  "Neighbourhood": "JARDIM DA PENHA",
  "Scholarship": 0,
  "Hipertension": 0,
  "Diabetes": 0,
  "Alcoholism": 0,
  "Handcap": 0,
  "SMS_received": 0,
  "days_until_appointment": 10,
  "scheduled_day_of_week": 6,
  "appointment_day_of_week": 6,
  "appointment_month": 1,
  "scheduled_hour": 10
}

Example response:

{
  "prediction": 1,
  "prediction_label": "No-show",
  "probability": 0.5739
}

The probability represents the estimated probability that the patient will miss the appointment.

8. Running FastAPI Locally

Activate the virtual environment and run:

uvicorn app.main:app --reload

The API will be available at:

http://localhost:8000

FastAPI automatically provides interactive Swagger documentation at:

http://localhost:8000/docs

The OpenAPI specification is available at:

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

# 10. Building the Docker Image

The Docker image was built using:

docker build -t medical-no-show-api .

This command reads the Dockerfile and creates a Docker image containing:

Python runtime
Python dependencies
FastAPI application
Machine learning model
Required source files

The image can be viewed using:

docker images
# 11. Running the Docker Container

The Docker container can be started using:

docker run -p 8000:8000 medical-no-show-api

The API can then be accessed at:

http://localhost:8000

Swagger documentation:

http://localhost:8000/docs

The Docker container allows the application to run without requiring the user to manually install all project dependencies.

12. Docker Image

The Docker image was tagged with the Docker Hub username:

fahmiyacm2025/medical-no-show-api

The Docker image was pushed to Docker Hub using:

docker push fahmiyacm2025/medical-no-show-api:latest

Docker Hub repository:

https://hub.docker.com/r/fahmiyacm2025/medical-no-show-api

The published image can be pulled using:

docker pull fahmiyacm2025/medical-no-show-api:latest

Then it can be run using:

docker run -p 8000:8000 fahmiyacm2025/medical-no-show-api:latest

This demonstrates that the application can be distributed as a Docker image and run on another machine that has Docker installed.

13. Step 3 – Automated Testing

Pytest is used to test the FastAPI application.

Tests are located in:

test/test_api.py

The tests use FastAPI's TestClient to send requests to the API without manually starting a web server.

Example:

pytest -v

The test suite checks that the API responds correctly to valid requests.

# 14. Continuous Integration – GitHub Actions

GitHub Actions is used to automatically run tests whenever code is pushed to GitHub or a pull request is created.

The workflow file is:

.github/workflows/test.yml

The workflow performs the following steps:

Checks out the GitHub repository.
Installs Python 3.12.
Installs project dependencies.
Checks the project structure.
Configures PYTHONPATH.
Runs the Pytest test suite.
# 15. GitHub Actions CI Workflow

The CI workflow is:

name: CI Tests


on:
  push:
  pull_request:


jobs:
  test:
    runs-on: ubuntu-latest


    steps:
      - name: Checkout code
        uses: actions/checkout@v4


      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"


      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt


      - name: Check project structure
        run: |
          pwd
          ls -la
          ls -la app
          python -c "import os; print(os.getcwd())"
          python -c "import sys; print(sys.path)"
          python -c "import app; print(app.__file__)"


      - name: Run tests
        env:
          PYTHONPATH: ${{ github.workspace }}
        run: |
          python -m pytest -v
# 16. How CI Works

Whenever code is pushed:

Developer
    ↓
git push
    ↓
GitHub
    ↓
GitHub Actions starts
    ↓
Checkout repository
    ↓
Install Python 3.12
    ↓
Install requirements
    ↓
Run Pytest
    ↓
Tests pass / fail

If all tests pass, GitHub Actions displays a green check mark.

If a test fails, the workflow fails and the developer can inspect the logs to identify the problem.

This provides automated validation of the project whenever changes are pushed.

# 17. CI Test Result

The GitHub Actions workflow successfully executed the test suite.

The workflow completed successfully with a green check mark.

The tests were executed using:

python -m pytest -v

This confirms that the automated testing pipeline is working.

18. Streamlit User Interface

A Streamlit application is included to provide a simple user interface for the machine learning model.

The Streamlit application is:

streamlit_app.py

The application allows users to enter patient appointment information and receive a prediction.

The interface displays:

Prediction: No-show
No-show Probability: 57.39%

The Streamlit application directly loads:

src/models/model_pipeline.pkl

This allows the public demonstration application to make predictions without requiring a separate FastAPI server.


