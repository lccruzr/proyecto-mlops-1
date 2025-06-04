# proyecto-mlops-manifests

Declarative Kubernetes manifests and Helm charts for the **proyecto-mlops** stack, managed by Argo CD.

## Quick start

```bash
# 1. Install Argo CD
kubectl create namespace argocd
helm repo add argo https://argo...
helm upgrade --install argocd argo/argo-cd --namespace argocd --set configs.params."server\.insecure"=true

# 2. Install the Application
kubectl apply -f argocd/proyecto-mlops-app.yaml

# 3. Watch sync
kubectl -n argocd get applications -w
```

Update image tags by committing to the `main` branch of the code repo; the CI pipeline will
open a PR here bumping `.global.sha`.
