# archivo: locustfile.py
# -----------------------------------------------------------
# Carga al endpoint /predict con el formato correcto:
# {"records": [ {race, gender, age} ]}
# -----------------------------------------------------------
import random
from locust import HttpUser, task, between

RACE_OPTS = [
    "Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "?"
]
GENDER_OPTS = ["Male", "Female", "Unknown/Invalid"]
AGE_OPTS = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]


class InferenceUser(HttpUser):
    """
    Locust arrancará tantos usuarios concurrentes como indiques
    y cada uno enviará un POST /predict cada 1-2 segundos.
    Usa --host http://api:8000 o la URL que corresponda.
    """
    wait_time = between(1, 2)

    @task
    def predict(self):
        payload = {
            "records": [
                {
                    "race": random.choice(RACE_OPTS),
                    "gender": random.choice(GENDER_OPTS),
                    "age": random.choice(AGE_OPTS),
                }
            ]
        }
        self.client.post("/predict", json=payload)
