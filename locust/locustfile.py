"""Escenario de carga: usuarios que consultan /predict con pacientes aleatorios."""
import random

from locust import HttpUser, between, task

AGES = ["[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)"]
MED = ["No", "Steady", "Up", "Down"]


def random_patient():
    return {
        "race": random.choice(["Caucasian", "AfricanAmerican", "Hispanic", "Other"]),
        "gender": random.choice(["Male", "Female"]),
        "age": random.choice(AGES),
        "admission_type_id": random.choice([1, 2, 3]),
        "discharge_disposition_id": random.choice([1, 3, 6]),
        "admission_source_id": random.choice([1, 7]),
        "time_in_hospital": random.randint(1, 14),
        "medical_specialty": random.choice(["InternalMedicine", "Cardiology", "?"]),
        "num_lab_procedures": random.randint(1, 100),
        "num_procedures": random.randint(0, 6),
        "num_medications": random.randint(1, 60),
        "number_outpatient": random.randint(0, 5),
        "number_emergency": random.randint(0, 3),
        "number_inpatient": random.randint(0, 6),
        "diag_1": random.choice(["428", "414", "786", "250.83", "V57"]),
        "diag_2": random.choice(["250", "427", "276"]),
        "diag_3": random.choice(["401", "250", "?"]),
        "number_diagnoses": random.randint(1, 16),
        "max_glu_serum": random.choice(["None", "Norm", ">200", ">300"]),
        "a1cresult": random.choice(["None", "Norm", ">7", ">8"]),
        "metformin": random.choice(MED), "glipizide": random.choice(MED),
        "glyburide": random.choice(MED), "insulin": random.choice(MED),
        "change": random.choice(["No", "Ch"]),
        "diabetesmed": random.choice(["Yes", "No"]),
    }


class PredictUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(10)
    def predict(self):
        self.client.post("/predict", json=random_patient(), name="/predict")

    @task(1)
    def model_info(self):
        self.client.get("/model-info", name="/model-info")
