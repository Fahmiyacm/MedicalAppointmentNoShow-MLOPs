import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib


# -----------------------------
# 1. Load dataset
# -----------------------------
DATA_PATH = r"C:\Users\fahmi\PycharmProjects\MedicalAppointmentMLOPs\data\raw\Medical.csv"
MODEL_PATH = "models/model_pipeline.pkl"

df = pd.read_csv(DATA_PATH)

print("Original shape:", df.shape)


# -----------------------------
# 2. Clean column names
# -----------------------------
df.columns = df.columns.str.strip()


# -----------------------------
# 3. Remove invalid ages
# -----------------------------
df = df[df["Age"] >= 0].copy()


# -----------------------------
# 4. Convert dates
# -----------------------------
df["ScheduledDay"] = pd.to_datetime(
    df["ScheduledDay"],
    errors="coerce",
    utc=True
)

df["AppointmentDay"] = pd.to_datetime(
    df["AppointmentDay"],
    errors="coerce",
    utc=True
)


# Remove rows where dates could not be converted
df = df.dropna(
    subset=["ScheduledDay", "AppointmentDay"]
).copy()


# -----------------------------
# 5. Feature engineering
# -----------------------------

# Days between scheduling and appointment
df["days_until_appointment"] = (
    df["AppointmentDay"] - df["ScheduledDay"]
).dt.days


# Remove invalid negative waiting times
df = df[df["days_until_appointment"] >= 0].copy()


# Scheduled day of week
df["scheduled_day_of_week"] = (
    df["ScheduledDay"].dt.dayofweek
)

# Appointment day of week
df["appointment_day_of_week"] = (
    df["AppointmentDay"].dt.dayofweek
)

# Appointment month
df["appointment_month"] = (
    df["AppointmentDay"].dt.month
)

# Scheduled hour
df["scheduled_hour"] = (
    df["ScheduledDay"].dt.hour
)


# -----------------------------
# 6. Target encoding
# -----------------------------
df["No-show"] = df["No-show"].map({
    "No": 0,
    "Yes": 1
})


# -----------------------------
# 7. Features
# -----------------------------
features = [
    "Gender",
    "Age",
    "Neighbourhood",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received",
    "days_until_appointment",
    "scheduled_day_of_week",
    "appointment_day_of_week",
    "appointment_month",
    "scheduled_hour"
]

X = df[features]
y = df["No-show"]


# -----------------------------
# 8. Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# 9. Preprocessing
# -----------------------------
categorical_features = [
    "Gender",
    "Neighbourhood"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# -----------------------------
# 10. Model
# -----------------------------
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)


# -----------------------------
# 11. Complete ML Pipeline
# -----------------------------
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# -----------------------------
# 12. Train
# -----------------------------
print("Training model...")

pipeline.fit(X_train, y_train)


# -----------------------------
# 13. Evaluate
# -----------------------------
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(
    "ROC-AUC:",
    roc_auc_score(y_test, y_prob)
)


# -----------------------------
# 14. Save model
# -----------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\nModel saved to:", MODEL_PATH)