#!/usr/bin/env bash
# Despliega todo en Kubernetes y muestra el estado. Ejecutar en la VM desde la raíz del repo.
set -euo pipefail
kubectl apply -k .
echo "Esperando a que los pods estén listos (puede tardar varios minutos la primera vez)..."
kubectl -n mlops wait --for=condition=available deployment --all --timeout=900s || true
kubectl -n mlops get pods -o wide
IP=$(hostname -I | awk '{print $1}')
cat <<URLS

  Airflow     http://$IP:30080   (admin / admin)
  MLflow      http://$IP:30500
  MinIO       http://$IP:30901   (minioadmin / minioadmin123)
  API (docs)  http://$IP:30800/docs
  Streamlit   http://$IP:30851
  Locust      http://$IP:30889
  Prometheus  http://$IP:30909
  Grafana     http://$IP:30300   (admin / admin)
URLS
