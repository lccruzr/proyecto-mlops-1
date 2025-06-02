"""
DAG: simulate_profesor_api
Crea un dataset sintético de listados inmobiliarios para pruebas locales
cuando el API real no está disponible.

• Genera 1 000 filas con variables típicas de Realtor
• Guarda Parquet + JSON de metadatos
• El nombre del archivo replica al usado por ingest_api_profesor
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# -------- CONFIG RÁPIDA --------
RAW_FOLDER = Path("/opt/airflow/data/raw")
RAW_FOLDER.mkdir(parents=True, exist_ok=True)

GROUP_NUMBER = 4          # para el sufijo _g4.parquet
N_ROWS = 1_000            # tamaño del dataset sintético
SEED = 42                 # reproducibilidad
SCHEDULE = None           # disparo manual; cámbialo a "@daily" si quieres
# ------------------------------


def _generate_dataset(**context: dict[str, Any]) -> None:
    """Crea datos ficticios tipo Realtor y los guarda en Parquet + JSON."""
    rng = np.random.default_rng(SEED)

    df = pd.DataFrame({
        "AREA":        rng.uniform(40, 350, N_ROWS).round(1),          # m²
        "BEDROOMS":    rng.integers(1, 6, N_ROWS),
        "BATHROOMS":   rng.integers(1, 5, N_ROWS),
        "ZIPCODE":     rng.choice(["75001", "33101", "90001", "10001"], N_ROWS),
        "YEAR_BUILT":  rng.integers(1950, 2024, N_ROWS),
        "LAT":         rng.uniform(25.0, 49.0, N_ROWS).round(6),
        "LON":         rng.uniform(-124.0, -67.0, N_ROWS).round(6),
        # Precio correlacionado con área, recámaras y ubicación
        "price":       lambda: None,   # placeholder; calculamos abajo
    })

    df["price"] = (
        df["AREA"] * rng.uniform(900, 1_400, N_ROWS) +
        df["BEDROOMS"] * rng.uniform(10_000, 50_000, N_ROWS) -
        (2025 - df["YEAR_BUILT"]) * rng.uniform(400, 900, N_ROWS)
    ).round(-3)  # redondea a millar

    # ------------- guarda -------------
    exec_date = context["ds_nodash"] if "ds_nodash" in context else \
                datetime.utcnow().strftime("%Y%m%d")

    parquet_path = RAW_FOLDER / f"{exec_date}_g{GROUP_NUMBER}.parquet"
    df.to_parquet(parquet_path, index=False)

    meta = {
        "generated": True,
        "rows": N_ROWS,
        "columns": list(df.columns),
        "group_number": GROUP_NUMBER,
        "exec_utc": datetime.utcnow().isoformat(),
        "file": str(parquet_path),
    }
    with open(parquet_path.with_suffix(".json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)

    print(f"✔ Dataset sintético guardado en {parquet_path}")


with DAG(
    dag_id="simulate_profesor_api",
    start_date=days_ago(1),
    schedule_interval=SCHEDULE,   # dispara manualmente
    catchup=False,
    tags=["dummy", "profesor_api"],
) as dag:

    generate = PythonOperator(
        task_id="generate_dummy_data",
        python_callable=_generate_dataset,
        provide_context=True,
    )
