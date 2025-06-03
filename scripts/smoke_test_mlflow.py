import os
import mlflow
from mlflow.tracking import MlflowClient

# Lee las variables de entorno
# MLFLOW_URI para el script de smoke test (podría ser diferente al MLFLOW_URI para los servicios)
# Usaremos localhost:5000 como en tu intento original, asumiendo que es accesible desde el runner
mlflow_tracking_uri = "http://localhost:5000"
model_name = os.environ.get("MODEL_NAME_ENV") # Usaremos una nueva variable para claridad

if not model_name:
    print("Error: La variable de entorno MODEL_NAME_ENV no está configurada.")
    exit(1)

print(f"Intentando conectar a MLflow en: {mlflow_tracking_uri}")
print(f"Buscando modelo: {model_name}")

try:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()
    
    # Intenta listar experimentos como una prueba de conexión básica
    experiments = client.search_experiments()
    print(f"Conexión a MLflow exitosa. Encontrados {len(experiments)} experimentos.")

    versions = client.search_model_versions(f"name='{model_name}'")
    print(f"Encontradas {len(versions)} versiones para el modelo '{model_name}'.")

except Exception as e:
    print(f"Error durante el smoke test de MLflow: {e}")
    exit(1)