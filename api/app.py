# app.py
import os
import mlflow
import mlflow.tracking # Usado para MlflowClient
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
    allow_origins=["*"], # Permite todos los orígenes
    allow_methods=["*"], # Permite todos los métodos
    allow_headers=["*"], # Permite todos los headers
)

logger = logging.getLogger("uvicorn.error") # Logger para uvicorn

# -------------------------------------------------------------------------
# Esquema fijo de cada registro (definido en Pydantic)
# -------------------------------------------------------------------------
class Record(BaseModel):
    brokered_by: str
    status: str
    bed: int
    bath: int
    acre_lot: float
    street: int
    city: str
    state: str
    zip_code: int
    house_size: int
    prev_sold_date: str  # Asumimos formato "YYYY-MM-DD" o compatible con pd.to_datetime

class Records(BaseModel):
    records: List[Record]

# -------------------------------------------------------------------------
# Variables globales para cachear el modelo y la versión cargada
# -------------------------------------------------------------------------
_model: mlflow.pyfunc.PyFuncModel = None
_loaded_model_version: str = None # Renombrado para claridad
_model_load_attempted_at_startup: bool = False
_startup_error_message: str = None

def _load_model():
    """
    Carga o recarga la última versión en 'Production' del modelo registrado.
    Actualiza las variables globales _model, _loaded_model_version, y _startup_error_message.
    """
    global _model, _loaded_model_version, _model_load_attempted_at_startup, _startup_error_message
    
    if not _model_load_attempted_at_startup: # Solo marcar la primera vez que se intenta al inicio
        _model_load_attempted_at_startup = True

    current_call_error_message = None # Para errores específicos de esta llamada

    try:
        logger.info(f"Intentando conectar a MLflow en: {MLFLOW_TRACKING_URI}")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        logger.info(f"Buscando modelo '{MODEL_NAME}' en stage '{MODEL_STAGE}'.")
        versions = client.get_latest_versions(name=MODEL_NAME, stages=[MODEL_STAGE])
        
        if not versions:
            current_call_error_message = f"No se encontró ninguna versión del modelo '{MODEL_NAME}' en stage '{MODEL_STAGE}'."
            logger.warning(current_call_error_message)
            # Si no hay versiones, el modelo debe ser None
            if _model is not None or _loaded_model_version is not None: # Si había un modelo cargado
                logger.info("Descargando modelo previamente cargado ya que no se encontraron versiones en producción.")
                _model = None
                _loaded_model_version = None
            # Actualizar _startup_error_message solo si es el intento inicial
            if _model_load_attempted_at_startup and _startup_error_message is None:
                 _startup_error_message = current_call_error_message
            raise RuntimeError(current_call_error_message)

        latest_version_info = versions[0]
        
        # Recargar solo si es una nueva versión o el modelo no está cargado
        if _model is None or latest_version_info.version != _loaded_model_version:
            logger.info(f"Cargando modelo '{MODEL_NAME}' versión {latest_version_info.version} desde MLflow.")
            model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
            _model = mlflow.pyfunc.load_model(model_uri)
            _loaded_model_version = latest_version_info.version
            logger.info(f"Modelo '{MODEL_NAME}' versión {_loaded_model_version} cargado exitosamente.")
            _startup_error_message = None # Limpiar el mensaje de error de inicio si la carga fue exitosa
        else:
            logger.info(f"Modelo '{MODEL_NAME}' versión {_loaded_model_version} ya está cargado. No se requiere recarga.")

    except Exception as e:
        # En caso de cualquier error durante la carga, el modelo se considera no disponible.
        _model = None
        _loaded_model_version = None
        current_call_error_message = str(e)
        logger.error(f"Error crítico al cargar o recargar el modelo: {current_call_error_message}")
        # Actualizar _startup_error_message solo si es el intento inicial y no hay ya un error más específico
        if _model_load_attempted_at_startup and _startup_error_message is None:
            _startup_error_message = current_call_error_message
        raise # Re-lanzar la excepción para que el llamador sepa que falló


# Intento de carga inicial del modelo al arrancar la aplicación
# Esto se hace una vez cuando el módulo se carga por primera vez.
logger.info("Iniciando aplicación FastAPI y intentando carga inicial del modelo...")
try:
    _load_model()
except Exception:
    # El error ya se registra y se almacena en _startup_error_message dentro de _load_model
    # No es necesario hacer nada más aquí, el estado de _model y _loaded_model_version ya está actualizado.
    logger.warning("La carga inicial del modelo falló. La API iniciará sin un modelo cargado.")


# -------------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------------
@app.get("/health", summary="Verifica el estado de la API y del modelo")
def health():
    """
    Retorna el estado del servicio API y la información del modelo de predicción.
    - `api_status`: Siempre "UP" si este endpoint responde.
    - `mlflow_tracking_uri`: El URI configurado para MLflow.
    - `model_status`: Estado de la carga del modelo ("LOADED", "LOAD_ATTEMPTED_FAILED", "NOT_LOADED_YET").
    - `model_name`: Nombre del modelo que se intenta cargar.
    - `model_stage`: Stage del modelo que se intenta cargar.
    - `loaded_model_version`: Versión del modelo actualmente cargada (si alguna).
    - `details`: Mensaje adicional, como un error si la carga falló.
    """
    details_message = "El modelo aún no se ha intentado cargar o no hay información de error."
    current_model_status = "NOT_LOADED_YET"

    if _model is not None and _loaded_model_version is not None:
        current_model_status = "LOADED"
        details_message = f"Modelo '{MODEL_NAME}' versión {_loaded_model_version} está activo."
    elif _model_load_attempted_at_startup: # Si se intentó cargar al inicio
        if _startup_error_message:
            current_model_status = "LOAD_ATTEMPTED_FAILED"
            details_message = f"Intento de carga inicial del modelo falló: {_startup_error_message}"
        else:
            # Esto podría pasar si _load_model se llamó pero no encontró versiones y
            # _startup_error_message se reseteó o no se seteó por alguna lógica no cubierta.
            # O si el modelo se descargó después de un intento exitoso.
            current_model_status = "NOT_LOADED" 
            details_message = "Se intentó cargar el modelo al inicio, pero no está disponible actualmente o no se encontraron versiones."
    
    # Si _model es None pero no se intentó cargar al inicio (escenario menos probable con el flujo actual)
    elif _model is None and not _model_load_attempted_at_startup:
        current_model_status = "LOAD_NOT_ATTEMPTED_AT_STARTUP"
        details_message = "La carga inicial del modelo no se ha realizado o está pendiente."


    return {
        "api_status": "UP",
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "model_status": current_model_status,
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "loaded_model_version": _loaded_model_version,
        "details": details_message
    }


@app.post("/predict", summary="Realiza predicciones usando el modelo cargado")
def predict(payload: Records):
    """
    Espera un JSON con una lista de registros para predecir.
    Cada registro debe cumplir con el esquema definido en `Record`.
    """
    global _model # Necesario para potencialmente reasignar _model si _load_model tiene éxito

    if _model is None:
        logger.warning("El modelo no está cargado. Intentando cargar/recargar para esta solicitud de predicción...")
        try:
            _load_model() # Intenta cargar el modelo ahora
        except Exception as e:
            logger.error(f"Fallo al intentar cargar el modelo durante la solicitud de predicción: {e}")
            raise HTTPException(
                status_code=503, # Service Unavailable
                detail=f"Modelo no disponible. Intento de carga para la solicitud falló: {e}"
            )
    
    # Si después del intento anterior, _model sigue siendo None, entonces el modelo no está realmente disponible.
    if _model is None:
        logger.error("El modelo sigue sin estar disponible después del intento de carga para la predicción.")
        raise HTTPException(
            status_code=503, 
            detail="Modelo no disponible o no se pudo cargar. Verifique el estado en /health."
        )

        # Convertir payload a DataFrame
    try:
        data_list = [record.model_dump() for record in payload.records] # Usa .dict() para Pydantic V1
        df = pd.DataFrame(data_list)
    except Exception as e:
        logger.error(f"Error al convertir el payload a DataFrame: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error procesando los datos de entrada: {e}")

    # Columnas que el modelo espera como float64
    # Asegúrate de que esta lista es correcta según la firma de tu modelo.
    # 'brokered_by' se asume categórica (str) según tu Pydantic Record.
    cols_to_ensure_float64 = {
        "bed", "bath", "acre_lot", "street", "zip_code", "house_size",
    }
    
    try:
        for col in df.columns:
            if col in cols_to_ensure_float64:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if not pd.api.types.is_float_dtype(df[col]): # Si no es float (ej. es int)
                    df[col] = df[col].astype('float64') # Forzar a float64
            # Aquí podrías añadir otras conversiones específicas si son necesarias, ej.:
            # elif col == 'prev_sold_date':
            #     df[col] = pd.to_datetime(df[col], errors='coerce')

        # Verificar NaNs en columnas críticas después de la conversión
        check_nan_cols = [c for c in cols_to_ensure_float64 if c in df.columns]
        if df[check_nan_cols].isnull().any().any():
            nan_columns_details = df.columns[df[check_nan_cols].isnull().any()].tolist()
            logger.error(f"Se encontraron valores NaN después de la conversión de tipos en columnas: {nan_columns_details}")
            raise ValueError(f"Error en la conversión de tipos: se generaron NaNs en columnas críticas: {nan_columns_details}")

    except Exception as e:
        logger.error(f"Error durante la conversión de tipos de datos para el DataFrame: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error en la preparación de datos para el modelo: {e}")

    logger.info(f"Tipos de datos del DataFrame antes de predecir: \n{df.dtypes}")

    # Realizar predicciones
    try:
        logger.info(f"Realizando predicciones con el modelo '{MODEL_NAME}' versión {_loaded_model_version} para {len(df)} registros.")
        predictions = _model.predict(df)
        logger.info("Predicciones realizadas exitosamente.")
    except Exception as e:
        logger.error(f"Error durante la predicción del modelo: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno al realizar la predicción: {e}")

    return JSONResponse(content={"predictions": predictions.tolist()})


if __name__ == "__main__":
    import uvicorn
    # Esta configuración es solo para ejecución local directa del script,
    # no afecta a cómo se ejecuta con un servidor ASGI como Uvicorn en producción/docker.
    uvicorn.run(app, host="0.0.0.0", port=8000)