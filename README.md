
# Proyecto MLOps - Sistema de Entrenamiento y Despliegue Automatizado con Kubernetes y Argo CD

Este proyecto implementa una solución completa de MLOps utilizando herramientas como Kubernetes, Argo CD, MLflow, Airflow, Streamlit, FastAPI y GitHub Actions.

## Estructura General

```
proyecto-mlops/
├── api/                   # API REST con FastAPI
├── ui/                    # Interfaz visual con Streamlit
├── ml/                    # Código de entrenamiento y modelo
├── dags/                  # DAGs de Airflow
├── infra/                 # Infraestructura con Helm + ArgoCD
│   └── argo-cd/
│       ├── argocd/        # Manifest YAML de la aplicación ArgoCD
│       └── apps/
│           └── umbrella/  # Umbrella Helm chart
├── .github/workflows/     # CI/CD con GitHub Actions
├── docker-compose.yml
└── README.md
```

---

## Tecnologías principales

- **Airflow**: Orquestación del entrenamiento
- **MLflow**: Registro de experimentos y artefactos
- **MinIO**: Almacenamiento S3-compatible
- **PostgreSQL**: Metadata
- **Grafana & Prometheus**: Monitoreo
- **FastAPI**: Servicio de predicción
- **Streamlit**: Visualización del modelo y SHAP
- **GitHub Actions**: CI/CD
- **Argo CD**: GitOps para despliegue continuo
- **Helm**: Manejo de charts

---

## ¿Cómo levantar todo?

### 1. Clonar el repositorio

```bash
git clone https://github.com/<usuario>/proyecto-mlops.git
cd proyecto-mlops
```

### 2. Crear imágenes y subir a GHCR (vía GitHub Actions)

Al hacer push en `main`, se activan los workflows que:
- Construyen imágenes Docker para `api` y `ui`
- Las publican en GitHub Container Registry (GHCR)
- Actualizan automáticamente `values.yaml` con el nuevo SHA
- Argo CD sincroniza automáticamente

### 3. Crear secret para GHCR en Kubernetes

```bash
kubectl create ns mlops
kubectl -n mlops create secret docker-registry ghcr \
  --docker-username=<tu_usuario_github> \
  --docker-password=<tu_token_personal> \
  --docker-email=ci@github
```

### 4. Aplicar la aplicación en Argo CD

```bash
kubectl apply -f infra/argo-cd/argocd/proyecto-mlops-app.yaml
```

Luego sincroniza en la interfaz de Argo CD.

### 5. Acceder a los servicios

```bash
kubectl -n mlops port-forward svc/api 8000:8000
kubectl -n mlops port-forward svc/ui 8501:8501
kubectl -n mlops port-forward svc/mlflow 5000:5000
```

- API (FastAPI): http://localhost:8000/docs
- UI (Streamlit): http://localhost:8501
- MLflow: http://localhost:5000

---

## Flujo MLOps

1. **Airflow** simula datos, entrena y registra modelos en MLflow.
2. El modelo se expone a través de una API (`api`) y una interfaz (`ui`).
3. **Argo CD** mantiene el estado del clúster sincronizado con Git.
4. Todo se orquesta desde GitHub Actions (CI/CD).

---

## Estado final esperado

- Aplicación `proyecto-mlops` en Argo CD → `Synced / Healthy`
- Todos los pods corriendo en `mlops`
- Modelos registrados en MLflow
- API y UI funcionando
- Dashboards activos (opcional en Grafana)

---

## 🧑‍💻 Autores

- **lccruzr**
- **SubjectumJC**