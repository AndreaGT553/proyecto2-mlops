# Alternativa a deploy.sh para PowerShell (Windows). Ejecutar desde la raíz del repo.
kubectl apply -k .
Write-Host "Esperando a que los pods estén listos (la primera vez puede tardar 10+ minutos)..."
kubectl -n mlops wait --for=condition=available deployment --all --timeout=900s
kubectl -n mlops get pods
Write-Host @"

  Airflow     http://localhost:30080   (admin / admin)
  MLflow      http://localhost:30500
  MinIO       http://localhost:30901   (minioadmin / minioadmin123)
  API (docs)  http://localhost:30800/docs
  Streamlit   http://localhost:30851
  Locust      http://localhost:30889
  Prometheus  http://localhost:30909
  Grafana     http://localhost:30300   (admin / admin)
"@
