# app.py
import os
import mlflow
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from typing import List

# -------------------------------------------------------------------------
# Configuración de entorno / constantes
# -------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME         = os.getenv("MODEL_NAME", "realtor_rf")
MODEL_STAGE        = os.getenv("MODEL_STAGE", "Production")

app = FastAPI(title="Realtor Price Predictor", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")

# -------------------------------------------------------------------------
# Esquema fijo de cada registro (definido en Pydantic)
# -------------------------------------------------------------------------
class Record(BaseModel):
    brokered_by: int
    status: str
    bed: int
    bath: int
    acre_lot: float
    street: int
    city: str
    state: str
    zip_code: int
    house_size: int
    prev_sold_date: str  # Formato "YYYY-MM-DD"

class Records(BaseModel):
    records: List[Record]


# -------------------------------------------------------------------------
# Variables globales para cachear el modelo y la versión cargada
# -------------------------------------------------------------------------
_model: mlflow.pyfunc.PyFuncModel = None
_loaded_version: str = None


def _load_model():
    """
    Carga o recarga la última versión en 'Production' del modelo registrado.
    """
    global _model, _loaded_version

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    versions = client.get_latest_versions(name=MODEL_NAME, stages=[MODEL_STAGE])
    if not versions:
        raise RuntimeError(f"No se encontró modelo '{MODEL_NAME}' en stage '{MODEL_STAGE}'")

    latest = versions[0]
    if latest.version != _loaded_version:
        _model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
        _loaded_version = latest.version
        logger.info(f"Modelo '{MODEL_NAME}' versión {_loaded_version} cargado")


# Inicializar al arrancar la aplicación
try:
    _load_model()
except Exception as e:
    _model, _loaded_version = None, None
    logger.warning(f"No se pudo cargar modelo al iniciar: {e}")


# -------------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------------
@app.get("/health")
def health():
    """
    Retorna el estado del servicio y la versión del modelo cargado.
    """
    status = "UP" if _model is not None else "DOWN"
    return {"status": status, "model_version": _loaded_version}


@app.post("/predict")
def predict(payload: Records):
    """
    Espera un JSON con:
    {
      "records": [
        {
          "brokered_by": 23594,
          "status": "for_sale",
          "bed": 2,
          "bath": 1,
          "acre_lot": 0.11,
          "street": 414327,
          "city": "Denver",
          "state": "Colorado",
          "zip_code": 80206,
          "house_size": 2072,
          "prev_sold_date": "2009-10-16"
        }
      ]
    }
    Convierte cada campo numérico explícitamente a float y retorna predicciones.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    # Intentar recargar si cambió la versión en Registry
    try:
        _load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al recargar modelo: {e}")

    # Convertir payload a DataFrame
    try:
        df = pd.DataFrame([r.dict() for r in payload.records])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al convertir payload a DataFrame: {e}")

    # --- Aquí forzamos a float64 las columnas numéricas que causaban el error ---
    numeric_cols = [
        "brokered_by",
        "bed",
        "bath",
        "acre_lot",
        "street",
        "zip_code",
        "house_size",
    ]
    try:
        df[numeric_cols] = df[numeric_cols].astype("float64")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al castear a float: {e}")

    # Predecir
    try:
        preds = _model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al predecir: {e}")

    return JSONResponse({"predictions": preds.tolist()})
