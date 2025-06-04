"""
DAG: test_mlflow_s3
• Entrena un modelo de clasificación (Iris) con scikit-learn
• Registra parámetros, métricas y artefactos en MLflow
• Descarga el artefacto para verificar lectura S3 (MinIO)
"""

from __future__ import annotations
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------------
# Ajusta si cambiaste credenciales o endpoint en tu .env
# ------------------------------------------------------------------
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
S3_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
AWS_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "minioadmin")
AWS_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin123")
EXPERIMENT_NAME = "test_s3_artifact_store"
# ------------------------------------------------------------------


def _train_and_log(**context: dict[str, Any]) -> str:
    """Entrena modelo, lo sube a MLflow y devuelve run_id."""
    # ––– credenciales para boto3 dentro del worker –––
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_KEY
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = S3_ENDPOINT

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="rf_iris") as run:
        run_id = run.info.run_id

        # Datos
        iris = load_iris(as_frame=True)
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.3, random_state=42
        )

        # Modelo
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # Métricas
        preds = rf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("n_estimators", 100)

        # Artefacto adicional (feature importances)
        fi_path = Path(tempfile.mkdtemp()) / "feature_importances.csv"
        pd.Series(rf.feature_importances_, index=iris.feature_names).to_csv(fi_path)
        mlflow.log_artifact(str(fi_path), artifact_path="analysis")

        # Serializa modelo
        mlflow.sklearn.log_model(rf, artifact_path="model")

    # Devolver run_id para la task siguiente
    context["ti"].xcom_push(key="run_id", value=run_id)
    return run_id


def _verify_artifact(**context: dict[str, Any]) -> None:
    """Descarga el CSV logeado y muestra su tamaño, prueba lectura S3."""
    run_id = context["ti"].xcom_pull(key="run_id", task_ids="train_and_log")

    # credenciales otra vez (por si es otro worker)
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_KEY
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = S3_ENDPOINT

    mlflow.set_tracking_uri(MLFLOW_URI)
    csv_local = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="analysis/feature_importances.csv",
    )
    size = Path(csv_local).stat().st_size
    print(f"✔ Artefacto descargado ({size} bytes): {csv_local}")


with DAG(
    dag_id="test_mlflow_s3",
    start_date=days_ago(1),
    schedule_interval=None,  # se dispara manualmente
    catchup=False,
    tags=["mlflow", "minio", "test"],
) as dag:
    train_and_log = PythonOperator(
        task_id="train_and_log",
        python_callable=_train_and_log,
        provide_context=True,
    )

    verify_artifact = PythonOperator(
        task_id="verify_artifact",
        python_callable=_verify_artifact,
        provide_context=True,
    )

    train_and_log >> verify_artifact
