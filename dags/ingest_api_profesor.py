"""
DAG: ingest_api_profesor
Descarga datos del API del profesor:
  GET http://localhost:8989/data?group_number=4&day=Tuesday
Guarda Parquet + metadatos JSON en /opt/airflow/data/raw
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from airflow import DAG, settings
from airflow.hooks.base import BaseHook
from airflow.models import Connection
from airflow.operators.python import PythonOperator
from airflow.sensors.http_sensor import HttpSensor
from airflow.utils.dates import days_ago
from sqlalchemy.exc import IntegrityError

# -------------------- CONFIGURACIÓN RÁPIDA --------------------
CONN_ID = "PROFESOR_API"
HOST = "http://host.docker.internal"      # o host-gateway si lo necesitas
PORT = 8989
ENDPOINT = "/data"
GROUP_NUMBER = 4
DAY = "Tuesday"                           # fijo (día de la clase)
#EXTRA = {"timeout": 30}
RAW_FOLDER = Path("/opt/airflow/data/raw")
RAW_FOLDER.mkdir(parents=True, exist_ok=True)
SCHEDULE = "@daily"
# --------------------------------------------------------------


def _ensure_connection() -> None:
    """Recrea PROFESOR_API sin campo timeout para evitar el header inválido."""
    session = settings.Session()

    # Elimina si existe
    session.query(Connection).filter(Connection.conn_id == CONN_ID).delete()

    # Crea de nuevo, limpio
    conn = Connection(
        conn_id=CONN_ID,
        conn_type="http",
        host=HOST,
        port=PORT,
        # extra=None  ←  sin timeout
    )
    session.add(conn)
    session.commit()
    print(f"✔ Conexión {CONN_ID} recreada sin 'timeout'")


def _download_and_store(**context):
    """
    Llama al API, toma la lista dentro de la clave "data", guarda Parquet
    y un JSON con metadatos.
    """
    conn = BaseHook.get_connection(CONN_ID)
    base = f"{conn.host}:{conn.port}" if conn.port else conn.host
    url  = f"{base.rstrip('/')}{ENDPOINT}"

    params = {"group_number": GROUP_NUMBER, "day": DAY}
    resp   = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()

    payload = resp.json()

    # --- extraemos la lista que realmente contiene las filas ---
    try:
        rows = payload["data"]
    except (TypeError, KeyError):
        raise ValueError("La respuesta no contiene la clave 'data'")

    df = pd.DataFrame(rows)

    exec_date    = context["ds_nodash"]           # AAAAMMDD
    parquet_path = RAW_FOLDER / f"{exec_date}_g{GROUP_NUMBER}.parquet"
    df.to_parquet(parquet_path, index=False)

    meta = {
        "group_number": GROUP_NUMBER,
        "day": DAY,
        "rows": len(df),
        "cols": list(df.columns),
        "url": url,
        "params": params,
        "ingest_utc": datetime.utcnow().isoformat(),
        "run_id": context["run_id"],
        "file": str(parquet_path),
    }
    with open(parquet_path.with_suffix(".json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)

    context["ti"].xcom_push(key="metadata", value=meta)



with DAG(
    dag_id="ingest_api_profesor",
    start_date=days_ago(1),
    schedule_interval=SCHEDULE,
    catchup=False,
    max_active_runs=1,
    tags=["ingest", "profesor_api"],
) as dag:

    ensure_connection = PythonOperator(
        task_id="ensure_connection",
        python_callable=_ensure_connection,
    )

    wait_for_api = HttpSensor(
        task_id="wait_for_api",
        http_conn_id=CONN_ID,
        endpoint=ENDPOINT,
        request_params={"group_number": GROUP_NUMBER, "day": DAY},
        poke_interval=30,
        timeout=300,
        mode="reschedule",
    )

    download_and_store = PythonOperator(
        task_id="download_and_store",
        python_callable=_download_and_store,
        provide_context=True,
    )

    ensure_connection >> wait_for_api >> download_and_store
