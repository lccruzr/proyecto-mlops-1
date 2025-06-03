"""
DAG: simulate_profesor_api
Genera un dataset sintético con el MISMO esquema que la API del profesor
cuando el túnel/servicio real no está disponible.

• 1 000 filas de variables tipo Realtor + precio
• Guarda Parquet + JSON de metadatos en /opt/airflow/data/raw
• El nombre de archivo replica el usado por ingest_api_profesor (YYYYMMDD_g4.parquet)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# ────────────────────────────
# CONFIG RÁPIDA
# ────────────────────────────
RAW_FOLDER = Path("/opt/airflow/data/raw")
RAW_FOLDER.mkdir(parents=True, exist_ok=True)

GROUP_NUMBER = 4          # sufijo _g4.parquet
N_ROWS = 1_000            # tamaño del dataset sintético
SEED = 42                 # reproducibilidad
SCHEDULE = None           # “None” => solo manual
# ────────────────────────────


def _generate_dataset(**context: dict[str, Any]) -> None:
    """Crea datos ficticios con el mismo esquema de la API del profesor."""
    rng = np.random.default_rng(SEED)

    # Combos (city, state, zip_code) predefinidos
    combos = np.array(
        [
            ("Denver",      "Colorado",   "80206"),
            ("Dallas",      "Texas",      "75001"),
            ("Miami",       "Florida",    "33101"),
            ("Los Angeles", "California", "90001"),
            ("New York",    "New York",   "10001"),
        ],
        dtype=object,
    )
    chosen = combos[rng.integers(0, len(combos), N_ROWS)]
    city = chosen[:, 0]
    state = chosen[:, 1]
    zip_code = chosen[:, 2]

    status_choices = np.array(["for_sale", "sold", "pending", "other"])

    df = pd.DataFrame(
        {
            "brokered_by": rng.integers(1_000, 50_000, N_ROWS),
            "status": status_choices[rng.integers(0, len(status_choices), N_ROWS)],
            "bed": rng.integers(1, 7, N_ROWS),
            "bath": rng.integers(1, 5, N_ROWS),
            "acre_lot": np.round(rng.uniform(0.05, 5.0, N_ROWS), 2),
            "street": rng.integers(100_000, 999_999, N_ROWS),
            "city": city,
            "state": state,
            "zip_code": zip_code,
            "house_size": rng.integers(500, 6_000, N_ROWS),
            "prev_sold_date": [
                (datetime(1995, 1, 1) + timedelta(days=int(d))).date()
                for d in rng.integers(0, 30 * 365, N_ROWS)
            ],
        }
    )

    # ---------- Calcular precio (target) ----------
    base_price_m2 = rng.uniform(900, 1_400, N_ROWS)
    bedroom_prem  = rng.uniform(10_000, 50_000, N_ROWS)
    lot_prem      = df["acre_lot"] * 80_000
    age_penalty   = (
        (2025 - pd.to_datetime(df["prev_sold_date"]).dt.year)
        * rng.uniform(200, 800, N_ROWS)
    )

    df["price"] = (
        df["house_size"] * base_price_m2
        + df["bed"] * bedroom_prem
        + lot_prem
        - age_penalty
    ).round(-3)

    # ---------- Guardar archivos ----------
    exec_date = context.get("ds_nodash") or datetime.utcnow().strftime("%Y%m%d")
    parquet_path = RAW_FOLDER / f"{exec_date}_g{GROUP_NUMBER}.parquet"
    df.to_parquet(parquet_path, index=False)

    meta = {
        "generated": True,
        "rows": len(df),
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
    schedule_interval=SCHEDULE,
    catchup=False,
    tags=["dummy", "profesor_api", "synthetic_data"],
) as dag:

    generate_dummy_data = PythonOperator(
        task_id="generate_dummy_data",
        python_callable=_generate_dataset,
        provide_context=True,
    )
