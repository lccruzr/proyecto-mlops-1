import os
import requests
import streamlit as st
from datetime import date, datetime
import tempfile
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

# -------------------------------------------------------------------------
# Configuración
# -------------------------------------------------------------------------
API_URL      = os.getenv("API_URL", "http://localhost:8000")
MODEL_NAME   = os.getenv("MODEL_NAME", "realtor_rf")
MODEL_STAGE  = os.getenv("MODEL_STAGE", "Production")
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Campos de entrada del modelo Realtor
INPUT_COLUMNS = [
    "brokered_by", "status", "bed", "bath", "acre_lot",
    "street", "city", "state", "zip_code", "house_size", "prev_sold_date"
]

st.set_page_config(page_title="Realtor Price Predictor 🏡", page_icon="🏡")

# ────────────────────────────
# Obtener versión de modelo
# ────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_model_version():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.ok:
            return r.json().get("model_version", "unknown")
    except Exception:
        pass
    return "unavailable"

MODEL_VERSION = fetch_model_version()

# ────────────────────────────
# Título y metadatos del modelo
# ────────────────────────────
st.title("🏡 Realtor Price Predictor")
st.caption(f"**Modelo activo:** `{MODEL_NAME}` · stage `{MODEL_STAGE}` · versión `{MODEL_VERSION}`")

# Mostrar columnas de entrada
st.subheader("Esquema de entrada del modelo")
st.write("El modelo espera un JSON con los siguientes campos:")
for col in INPUT_COLUMNS:
    st.write(f"- `{col}`")
st.markdown("---")

# ────────────────────────────
# Formulario de entrada para cada campo
# ────────────────────────────
st.subheader("Ingresa los valores para predecir")

brokered_by    = st.text_input("brokered_by (ID corredor)", "23594.0")
status         = st.selectbox("status", ["for_sale", "sold", "pending", "other"])
bed            = st.number_input("bed (número de habitaciones)", min_value=0, value=2, step=1)
bath           = st.number_input("bath (número de baños)", min_value=0, value=1, step=1)
acre_lot       = st.number_input(
    "acre_lot (tamaño del lote en acres)", 
    min_value=0.0, value=0.11, step=0.01, format="%.2f"
)
street         = st.number_input("street (ID calle)", min_value=0, value=414327, step=1)
city           = st.text_input("city", "Denver")
state          = st.text_input("state", "Colorado")
zip_code       = st.number_input("zip_code", min_value=0, value=80206, step=1)
house_size     = st.number_input("house_size (m²)", min_value=0, value=2072, step=1)
prev_sold_date = st.date_input("prev_sold_date", date(2009, 10, 16))

# ────────────────────────────
# Botón para predecir
# ────────────────────────────
prediction = None
if st.button("Predecir precio"):
    payload = {
        "records": [
            {
                "brokered_by": brokered_by,
                "status": status,
                "bed": int(bed),
                "bath": int(bath),
                "acre_lot": float(acre_lot),
                "street": int(street),
                "city": city,
                "state": state,
                "zip_code": int(zip_code),
                "house_size": int(house_size),
                "prev_sold_date": prev_sold_date.strftime("%Y-%m-%d")
            }
        ]
    }
    with st.spinner("Consultando modelo…"):
        try:
            res = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if res.ok:
                result = res.json().get("predictions", [])
                if result:
                    prediction = result[0]
                    st.success(f"Predicción de precio: {prediction:,.2f}")
                else:
                    st.warning("Respuesta vacía del modelo")
            else:
                st.error(f"Error {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"Error al conectar con API: {e}")

st.markdown("---")

# ────────────────────────────
# Historial de modelos y métricas
# ────────────────────────────
st.subheader("📜 Historial de Modelos")

@st.cache_data(show_spinner=False)
def fetch_model_history():
    """
    Obtiene del Model Registry el historial de versiones de MODEL_NAME,
    con su estado, fecha, métrica MAE y razón (producción/archive).
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception:
        return pd.DataFrame(), "error"

    history = []
    for v in versions:
        version = int(v.version)
        stage = v.current_stage
        run_id = v.run_id

        # Obtener métrica MAE del run
        try:
            run_data = client.get_run(run_id).data
            mae = run_data.metrics.get("mae", None)
            timestamp = client.get_run(run_id).info.start_time / 1000
            date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            mae = None
            date_str = "desconocida"

        if stage == "Production":
            reason = "Versión activa en Producción"
        else:
            reason = "Archivado por nueva versión"

        history.append({
            "version": version,
            "stage": stage,
            "mae": mae,
            "fecha": date_str,
            "razón": reason
        })

    df = pd.DataFrame(history)
    if not df.empty:
        df = df.sort_values("version", ascending=False).reset_index(drop=True)
    return df, "ok"

with st.spinner("Cargando historial de modelos..."):
    history_df, status_hist = fetch_model_history()
    if status_hist == "error":
        st.error("No se pudo conectar a MLflow para obtener historial.")
    else:
        if not history_df.empty:
            st.table(history_df)
        else:
            st.info("No se encontraron versiones registradas.")

st.markdown("---")

# ────────────────────────────
# Explicabilidad con SHAP
# ────────────────────────────
st.subheader("🔍 Explicabilidad del modelo (SHAP)")

@st.cache_data(show_spinner=False)
def download_shap_plot(model_name: str, stage: str):
    """
    Descarga el artefacto SHAP (summary plot) desde MLflow y devuelve
    la ruta local del PNG para mostrarlo en Streamlit.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()
        latest = client.get_latest_versions(model_name, [stage])[0]
        run_id = latest.run_id
        artifact_path = f"shap/shap_summary_{run_id}.png"
        local_dir = tempfile.mkdtemp()
        local_file = client.download_artifacts(run_id, artifact_path, dst_path=local_dir)
        return local_file, "ok"
    except Exception:
        return "", "error"

with st.spinner("Cargando gráfico SHAP..."):
    shap_file, status_shap = download_shap_plot(MODEL_NAME, MODEL_STAGE)
    if status_shap == "error":
        st.error("No se pudo conectar a MLflow para descargar gráfico SHAP.")
    else:
        if shap_file and os.path.exists(shap_file):
            st.image(shap_file, caption="SHAP summary plot (importancia global)")
        else:
            st.info("Gráfico SHAP no disponible (aún no generado o error de descarga).")
