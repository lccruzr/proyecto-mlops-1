from __future__ import annotations
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any, List
import logging

import mlflow
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.dates import days_ago
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import sparse

# ------------ ajustes rápidos ---------------
RAW_FOLDER = Path("/opt/airflow/data/raw")
MODEL_NAME = "realtor_rf"
TARGET_COL = "price"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
N_ESTIM = 10
SEED = 42
SHAP_FOLDER = Path("/opt/airflow/data/shap")
SHAP_FOLDER.mkdir(parents=True, exist_ok=True)
# --------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _latest_parquet() -> str:
    files = sorted(
        RAW_FOLDER.glob("*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError("Sin archivos parquet en RAW_FOLDER")
    return str(files[0])


def _md5(path: str, block: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _decide(**ctx) -> str:
    logging.info("🔎 Decidiendo si hay que re-entrenar…")
    data_path = ctx["ti"].xcom_pull(key="data_path", task_ids="get_data")
    data_hash = _md5(data_path)

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    prod = [
        v
        for v in client.search_model_versions(f"name='{MODEL_NAME}'")
        if v.current_stage == "Production"
    ]

    ctx["ti"].xcom_push(key="prev_ver", value=prod[0].version if prod else None)

    if not prod:
        logging.info("🆕 No existe modelo en producción — se entrenará uno nuevo.")
        return "retrain"
    if prod[0].tags.get("data_hash") != data_hash:
        logging.info("📊 Los datos cambiaron — se entrenará una nueva versión.")
        return "retrain"

    logging.info("✅ Datos sin cambios — se omite re-entrenamiento.")
    return "skip"


def _train_register(**ctx) -> None:
    logging.info("🚀 Iniciando proceso de entrenamiento…")
    data_path = ctx["ti"].xcom_pull(key="data_path", task_ids="get_data")
    data_hash = _md5(data_path)
    prev_ver = ctx["ti"].xcom_pull(key="prev_ver", task_ids="decide")

    df = pd.read_parquet(data_path)
    logging.info(f"📥 Datos cargados: {df.shape}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    cat_cols: List[str] = X.select_dtypes(include="object").columns.tolist()
    num_cols: List[str] = [c for c in X.columns if c not in cat_cols]

    prepro = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
        ]
    )

    rf = RandomForestRegressor(n_estimators=N_ESTIM, random_state=SEED, n_jobs=-1)
    pipe = Pipeline(steps=[("prep", prepro), ("rf", rf)])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    logging.info(f"🟢 Split entrenamiento/validación — train: {X_tr.shape}, test: {X_te.shape}")

    pipe.fit(X_tr, y_tr)
    mae = mean_absolute_error(y_te, pipe.predict(X_te))
    logging.info(f"📈 Modelo entrenado — MAE: {mae:.2f}")

    example_df = X_tr.head(5)
    signature = infer_signature(example_df, pipe.predict(example_df))

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("realtor_training")
    logging.info("🔗 MLflow configurado, iniciando run…")

    with mlflow.start_run(run_name=f"rf_{datetime.utcnow():%Y%m%d%H%M}") as run:
        mlflow.log_metric("mae", mae)
        mlflow.log_param("n_estimators", N_ESTIM)
        mlflow.log_param("train_rows", len(X_tr))

        client = MlflowClient()
        if MODEL_NAME not in [m.name for m in client.list_registered_models()]:
            client.create_registered_model(MODEL_NAME)
            logging.info("📚 Modelo registrado en MLflow Model Registry.")

        # -------- SHAP ------------
        logging.info("🧐 Calculando SHAP sobre 100 muestras…")
        X_explain = X_te.sample(n=100, random_state=SEED)
        transformed = prepro.transform(X_explain)
        X_explain_trans = (
            transformed.toarray() if sparse.issparse(transformed) else transformed
        ).astype(float)

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_explain_trans, check_additivity=False)
        feature_names = prepro.get_feature_names_out()

        shap_df = pd.DataFrame(X_explain_trans, columns=feature_names)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            shap_df,
            plot_type="bar",
            show=False
        )
        shap_png = SHAP_FOLDER / f"shap_summary_{run.info.run_id}.png"
        fig.savefig(shap_png, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(shap_png), artifact_path="shap")
        logging.info("🎨 Gráfico SHAP guardado y subido a MLflow.")

        sv_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_parquet = SHAP_FOLDER / f"shap_values_{run.info.run_id}.parquet"
        sv_df.to_parquet(shap_parquet, index=False)
        mlflow.log_artifact(str(shap_parquet), artifact_path="shap")

        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=example_df.iloc[:1],
            extra_pip_requirements=["scikit-learn", "pandas", "numpy", "shap"],
            serialization_format="cloudpickle",
        )
        logging.info("📦 Pipeline y metadata logueados en MLflow.")

        new_ver = max(
            v.version for v in client.search_model_versions(f"name='{MODEL_NAME}'")
        )
        client.set_model_version_tag(MODEL_NAME, new_ver, "data_hash", data_hash)
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new_ver,
            stage="Production",
            archive_existing_versions=True,
        )
        logging.info(f"🏷️ Versión {new_ver} promovida a Production.")

        if prev_ver:
            logging.info(f"📦 Versión previa {prev_ver} archivada.")
        logging.info("✅ Run completado.")


with DAG(
    dag_id="train_realtor_model",
    start_date=days_ago(1),
    schedule_interval="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["realtor", "training", "mlflow"],
) as dag:

    get_data = PythonOperator(
        task_id="get_data",
        python_callable=lambda **c: c["ti"].xcom_push(
            key="data_path", value=_latest_parquet()
        ),
    )

    decide = BranchPythonOperator(
        task_id="decide", python_callable=_decide, provide_context=True
    )

    retrain = PythonOperator(
        task_id="retrain", python_callable=_train_register, provide_context=True
    )

    skip = EmptyOperator(task_id="skip")

    get_data >> decide
    decide >> retrain
    decide >> skip
