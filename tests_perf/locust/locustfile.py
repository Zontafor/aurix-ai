# locustfile.py
# Locust load test for the AURIX-AI churn service.
#
# UI:
#   locust -f locust/locustfile.py --host http://127.0.0.1:8000
#
# Headless example:
#   locust -f locustfile.py --headless -u 2000 -r 8 -t 30m --host=http://localhost:8080 --loglevel=INFO
#   gunicorn -k uvicorn.workers.UvicornWorker -w 8 -b 0.0.0.0:8080 src.api.main:app
#   lsof -ti tcp:8080 | xargs kill -9

import random
import copy
import pandas as pd
from typing import Tuple
from pathlib import Path
from payloads import random_payload
from locust import HttpUser, task, between
from locust.exception import RescheduleTask

# BASE_PAYLOAD = {
#     "customerID": "locust-user",
#     "tenure": 18,
#     "MonthlyCharges": 65.0,
#     "TotalCharges": 1170.0,
#     "Contract": "Month-to-month",
#     "InternetService": "Fiber optic",
#     "PaymentMethod": "Electronic check",
#     "PaperlessBilling": "Yes",
#     "gender": "Male",
#     "SeniorCitizen": 0,
#     "Partner": "No",
#     "Dependents": "No",
#     "PhoneService": "Yes",
#     "MultipleLines": "No",
#     "OnlineSecurity": "No",
#     "OnlineBackup": "No",
#     "DeviceProtection": "No",
#     "TechSupport": "No",
#     "StreamingTV": "Yes",
#     "StreamingMovies": "Yes",
# }

# def random_payload():
#     """Return a slightly randomized payload so the service isn't trivially cached."""
#     p = copy.deepcopy(BASE_PAYLOAD)

#     # Make customerID unique-ish per request
#     p["customerID"] = f"locust-{random.randint(1, 10_000_000)}"

#     # Core numeric variability
#     tenure = random.randint(0, 72)
#     monthly = round(random.uniform(20.0, 120.0), 2)
#     total = round(monthly * max(tenure, 1) * random.uniform(0.5, 1.2), 2)

#     p["tenure"] = tenure
#     p["MonthlyCharges"] = monthly
#     p["TotalCharges"] = total

#     # Categorical variability (keep within known Telco values)
#     p["Contract"] = random.choice(["Month-to-month", "One year", "Two year"])
#     p["InternetService"] = random.choice(["DSL", "Fiber optic", "No"])
#     p["PaymentMethod"] = random.choice([
#         "Electronic check",
#         "Mailed check",
#         "Bank transfer (automatic)",
#         "Credit card (automatic)",
#     ])
#     p["PaperlessBilling"] = random.choice(["Yes", "No"])
#     p["gender"] = random.choice(["Male", "Female"])
#     p["SeniorCitizen"] = random.choice([0, 1])
#     p["Partner"] = random.choice(["Yes", "No"])
#     p["Dependents"] = random.choice(["Yes", "No"])

#     # Keep services broadly consistent with InternetService="No"
#     if p["InternetService"] == "No":
#         for k in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
#             p[k] = "No"
#     else:
#         for k in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
#             p[k] = random.choice(["Yes", "No"])

#     return p

# def payload():
#     p = copy.deepcopy(BASE_PAYLOAD)
#     p["customerID"] = f"locust-{random.randint(1, 10_000_000)}"
#     return p

# class AurixUser(HttpUser):
#     host = "http://127.0.0.1:8080"

#     # Think time between requests: adjust as you like for realism
#     # wait_time = between(0.2, 1.2)
#     wait_time = between(0.1, 0.5)

#     # If your endpoint differs, change it here
#     PREDICT_PATH = "/predict_risk"  # or "/predict" or "/predict_churn"

#     @task(10)
#     def predict_risk(self):
#         # try:
#         #     payload = random_payload()
#         # except Exception as e:
#         #     print("payload error:", repr(e))
#         #     raise RescheduleTask()

#         payload = random_payload()

#         # self.client.post("/predict_risk", json=payload, name="/predict_risk")

#         # catch_response lets us mark “soft failures” (wrong schema) as failures too
#         with self.client.post(self.PREDICT_PATH, json=payload, name=self.PREDICT_PATH, catch_response=True) as resp:
#             if resp.status_code != 200:
#                 resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
#                 return

#             # "Happy path" schema checks (matches your HW4 requirement)
#             try:
#                 data = resp.json()
#             except Exception:
#                 resp.failure("Response was not valid JSON")
#                 return

#             # Update these keys to match your API's actual response contract
#             with self.client.post("/predict_risk", json=payload, name="/predict_risk", catch_response=True) as r:
#                 if resp.status_code != 200:
#                     resp.failure(f"HTTP {r.status_code}: {r.text[:200]}")
#                     return
#                 data = resp.json()
#                 if "churn_probability" not in data:
#                     resp.failure(f"Missing churn_probability. keys={list(data.keys())}")
#                     return
#                 p = data["churn_probability"]
#                 if not (0.0 <= float(p) <= 1.0):
#                     resp.failure(f"Bad churn_probability={p}")
#                     return
                
#             resp.success()

#     @task(1)
#     def health(self):
#         # Remove if you don't have /health
#         self.client.get("/health", name="/health")

BASE_PAYLOAD = {
    "customerID": "locust-user",
    "tenure": 18,
    "MonthlyCharges": 65.0,
    "TotalCharges": 1170.0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    "PaperlessBilling": "Yes",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
}

# -------------------------------------------------------------------
# Payload generator
# -------------------------------------------------------------------
def make_payload():
    p = copy.deepcopy(BASE_PAYLOAD)

    p["customerID"] = f"locust-{random.randint(1, 10_000_000)}"

    tenure = random.randint(0, 72)
    monthly = round(random.uniform(20.0, 120.0), 2)
    total = round(monthly * max(tenure, 1) * random.uniform(0.5, 1.2), 2)

    p["tenure"] = tenure
    p["MonthlyCharges"] = monthly
    p["TotalCharges"] = total

    p["Contract"] = random.choice(["Month-to-month", "One year", "Two year"])
    p["InternetService"] = random.choice(["DSL", "Fiber optic", "No"])
    p["PaymentMethod"] = random.choice([
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ])
    p["PaperlessBilling"] = random.choice(["Yes", "No"])
    p["gender"] = random.choice(["Male", "Female"])
    p["SeniorCitizen"] = random.choice([0, 1])
    p["Partner"] = random.choice(["Yes", "No"])
    p["Dependents"] = random.choice(["Yes", "No"])

    if p["InternetService"] == "No":
        for k in [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]:
            p[k] = "No"
    else:
        for k in [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]:
            p[k] = random.choice(["Yes", "No"])

    return p

# -------------------------------------------------------------------
# Locust user
# -------------------------------------------------------------------
class AurixUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict_risk(self):
        payload = make_payload()

        with self.client.post(
            "/predict_risk",
            json=payload,
            name="/predict_risk",
            timeout=5,
            catch_response=True,
        ) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}: {r.text[:200]}")
                return

            try:
                data = r.json()
            except Exception:
                r.failure("Response not valid JSON")
                return

            if "churn_probability" not in data:
                r.failure(f"Missing churn_probability: {list(data.keys())}")
                return

            p = data["churn_probability"]
            if not isinstance(p, (int, float)) or not (0.0 <= p <= 1.0):
                r.failure(f"Invalid churn_probability={p}")
                return

            r.success()

    @task(1)
    def health(self):
        self.client.get("/health", name="/health", timeout=2)

def engineer_aurix_features(df: pd.DataFrame, output_file: Path, target_col: str = "churn_yes") -> Tuple[pd.DataFrame, pd.Series]:
    """Engineer features for AURIX-AI churn prediction from cleaned Telco dataset.

    Args:
        df (pd.DataFrame): Cleaned Telco dataset.
        output_file (Path): Path to save engineered dataset CSV.
        target_col (str): Name of the target column indicating churn.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix X and target vector y.
    """
    # Create binary churn_flag
    df["churn_flag"] = df[target_col].astype(int)