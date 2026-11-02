# Proyecto 2 – MLOps en Kubernetes (Readmisión hospitalaria en pacientes con diabetes)

Arquitectura MLOps completa desplegada en Kubernetes: ingesta incremental por lotes con **Airflow**,
capas **RAW / CLEAN / INFERENCE** en PostgreSQL, experimentos y registro de modelos en **MLflow**
(PostgreSQL + MinIO), API de inferencia con **FastAPI**, interfaz en **Streamlit**, pruebas de carga
con **Locust** y observabilidad con **Prometheus + Grafana**.

## 1. Arquitectura

```
                 ┌──────────────────────── Kubernetes (namespace mlops) ────────────────────────┐
 diabetic_data   │  FASE DE ENTRENAMIENTO                                                       │
 .csv ──────────►│  Airflow (webserver+scheduler) ──► postgres-data: raw.* / clean.*            │
 (en la imagen)  │        │                                                                     │
                 │        └──► MLflow server ──► postgres-meta (mlflowdb)  + MinIO (artefactos) │
                 │                  ▲                                                           │
                 │  FASE DE INFERENCIA                                                          │
                 │  Streamlit ──► API FastAPI ──(models:/diabetes-readmission@champion)         │
                 │  Locust ─────►     │  └──► postgres-data: inference.predictions            │
                 │                    └──/metrics──► Prometheus ──► Grafana                     │
                 └──────────────────────────────────────────────────────────────────────────────┘
```

| Componente | Recurso K8s | Puerto (NodePort) | Credenciales |
|---|---|---|---|
| postgres-data (datos) | StatefulSet + PVC | interno 5432 | `datauser / datapass` |
| postgres-meta (metadatos MLflow + Airflow) | StatefulSet + PVC | interno 5432 | `metauser / metapass` |
| MinIO | Deployment + PVC | consola **30901** | `minioadmin / minioadmin123` |
| MLflow | Deployment | **30500** | – |
| Airflow | Deployment (webserver + scheduler) + PVC logs | **30080** | `admin / admin` |
| API de inferencia | Deployment | **30800** (`/docs`) | – |
| Streamlit | Deployment | **30851** | – |
| Locust | Deployment | **30889** | – |
| Prometheus | Deployment | **30909** | – |
| Grafana | Deployment | **30300** | `admin / admin` |

Todos los contenedores definen `resources.requests/limits`, `readinessProbe` y `livenessProbe`;
la configuración no sensible va en **ConfigMaps** y las credenciales en **Secrets**.

### Estructura del repositorio
```
common/preprocessing.py     limpieza + ingeniería de características (compartido DAG/API)
notebooks/                  análisis exploratorio y justificación de decisiones
sql/                        scripts de creación de BD y tablas (se montan como ConfigMap)
airflow/dags/               DAG principal (10 tareas)
api/  ui/  locust/  mlflow/ código + Dockerfile de cada componente
observability/              prometheus.yml, datasource y dashboard de Grafana (JSON)
k8s/  kustomization.yaml    manifiestos (kubectl apply -k .)
build_push.sh  deploy.sh    construir/publicar imágenes y desplegar
```

## 2. Decisiones técnicas

**Capas de datos.** Una instancia `postgres-data` con tres esquemas: `raw.diabetes` (todo como TEXT,
sin transformar, con `batch_id`, `load_timestamp`, `source_file`, `source_row`, `row_hash`, `status`,
`dag_run_id`), `clean.diabetes` (features listas para entrenar + `split`) e `inference.predictions`.
Los metadatos de MLflow y Airflow viven en **otra** instancia (`postgres-meta`), separada de los datos del negocio.

**Problema y métrica.** Target binario `readmitted_30` (readmisión < 30 días, ~11 % positivos).
Métrica de selección: **F1 de la clase positiva**. Accuracy es engañosa (predecir siempre "no" da ~89 %)
y recall solo se maximiza marcando a todos como positivos. F1 equilibra ambos. También se registran
recall, precision, ROC-AUC y accuracy (ver `notebooks/01_eda_preprocesamiento.ipynb`).

**DAG `diabetes_mlops_pipeline`** (cada 10 min, `max_active_runs=1`):
`check_source_file → load_raw_batch → validate_quality → preprocess_data → store_clean_data → split_data →
train_models → register_best_model → compare_with_champion → promote_model`.
- Cada ejecución carga **un lote de máx. 15.000 filas** (101.766 filas = 7 ejecuciones). Cuando no quedan filas, la tarea se marca *skipped*.
- **Sin duplicados:** `row_hash` es UNIQUE (`INSERT ... ON CONFLICT DO NOTHING`); si se reejecuta la misma corrida se reutiliza su lote (`dag_run_id`); `clean` se reescribe por lote (DELETE + INSERT dentro de una transacción).
- **Split estable** 70/15/15 por hash del registro: un registro nunca cambia de subconjunto entre reentrenamientos.
- Se entrenan Regresión Logística y Random Forest (`class_weight` balanceado). Se registra el mejor por F1 de validación, se compara contra el `@champion` actual **sobre el mismo conjunto de test** y, si es igual o mejor, se le asigna el alias **`champion`**.
- Si una tarea falla, Airflow la reintenta 1 vez y el error queda visible; las escrituras son transaccionales.

**Carga del modelo en la API.** Al iniciar carga `models:/diabetes-readmission@champion`. Luego, como caché,
cada 60 s consulta (llamada liviana) qué versión tiene el alias y solo descarga el modelo si cambió.
`POST /reload` fuerza la recarga. No hay rutas ni versiones fijas en el código.

---

## 3. Guía de despliegue paso a paso

### Paso 0 – Conectarse a la VM
1. Conecta la VPN de la universidad.
2. `ssh estudiante@10.43.97.99`
3. Verifica recursos: `free -h` y `nproc` (se recomiendan ≥ 8 GB de RAM y 4 CPU).

### Paso 1 – Tener Kubernetes en la VM
Primero revisa si ya existe: `kubectl get nodes`. Si responde con un nodo `Ready`, salta al Paso 2.
Si no existe, una opción liviana para Rocky Linux es **k3s**:
```bash
curl -sfL https://get.k3s.io | sh -
sudo chmod 644 /etc/rancher/k3s/k3s.yaml
mkdir -p ~/.kube && sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config && sudo chown $USER ~/.kube/config
kubectl get nodes            # debe mostrar el nodo en estado Ready
# Abrir los NodePorts si firewalld está activo:
sudo firewall-cmd --permanent --add-port=30000-32767/tcp && sudo firewall-cmd --reload
```

### Paso 2 – Clonar el repositorio en la VM
```bash
git clone https://github.com/AndreaGT553/proyecto2-mlops.git
cd proyecto2-mlops
```

### Paso 3 – Construir y publicar las imágenes en DockerHub
Desde tu máquina (Docker Desktop) o desde la VM si tiene Docker:
```bash
docker login
./build_push.sh <tu_usuario_dockerhub>
```
El script construye las 5 imágenes para `linux/amd64`, las sube a DockerHub y reemplaza
`DOCKERHUB_USER` en los manifiestos. **Haz commit y push de ese cambio** y luego `git pull` en la VM.
Evidencia para la entrega: captura de las imágenes en hub.docker.com.

### Paso 4 – Desplegar
```bash
./deploy.sh
kubectl -n mlops get pods        # todos deben quedar Running / 2/2 en airflow
```
Si un pod no arranca: `kubectl -n mlops describe pod <pod>` y `kubectl -n mlops logs <pod> -c <contenedor>`.

### Paso 5 – Ejecutar el DAG
1. Abre Airflow `http://10.43.97.99:30080` (admin/admin).
2. Activa (unpause) `diabetes_mlops_pipeline` y pulsa ▶ *Trigger DAG*.
3. Cada corrida carga el siguiente lote. Ejecútalo varias veces (o deja que corra cada 10 min) hasta cargar los 7 lotes.
4. Verifica los datos:
```bash
kubectl -n mlops exec -it postgres-data-0 -- psql -U datauser -d datadb -c \
 "SELECT batch_id, status, COUNT(*) FROM raw.diabetes GROUP BY 1,2 ORDER BY 1;"
kubectl -n mlops exec -it postgres-data-0 -- psql -U datauser -d datadb -c \
 "SELECT split, COUNT(*), AVG(readmitted_30) FROM clean.diabetes GROUP BY 1;"
```

### Paso 6 – Revisar MLflow
`http://10.43.97.99:30500` → experimento `diabetes_readmission` (runs con parámetros, métricas, reporte y modelo)
→ pestaña *Models* → `diabetes-readmission`: la versión con alias **champion** es la productiva.
En MinIO (`:30901`) se ven los artefactos en el bucket `mlflow-artifacts`.

### Paso 7 – Inferencia
- Swagger: `http://10.43.97.99:30800/docs` → `/predict`, `/model-info`, `/health`, `/metrics`.
- Streamlit: `http://10.43.97.99:30851` → *Cargar ejemplo* → *Predecir*.
- Si el champion cambió y quieres verlo de inmediato: `curl -X POST http://10.43.97.99:30800/reload`.
- Registro de inferencias:
```bash
kubectl -n mlops exec -it postgres-data-0 -- psql -U datauser -d datadb -c \
 "SELECT created_at, prediction, round(probability::numeric,3), model_version, round(latency_ms::numeric,1) FROM inference.predictions ORDER BY id DESC LIMIT 5;"
```

### Paso 8 – Prueba de carga y observabilidad
1. Grafana `http://10.43.97.99:30300` → carpeta *MLOps* → **Proyecto 2 - API de inferencia**.
2. Locust `http://10.43.97.99:30889` (host ya configurado `http://inference-api:8000`).
3. Escenario sugerido: arrancar con 10 usuarios (spawn rate 2), luego 50, 100, 200.
4. Observa en Grafana RPS, latencia p50/p95/p99, tasa de error, CPU y memoria. El **límite operativo** es el
   punto en que el RPS deja de crecer al subir usuarios mientras p95 se dispara (la API tiene límite de 1 CPU).
5. Registra en Locust: usuarios, spawn rate, total de solicitudes, fallidas, latencia promedio y percentiles.

## 4. Resultados de referencia (notebook, dataset completo)
| Modelo | Accuracy | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (siempre 0) | 0.886 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.674 | 0.547 | 0.185 | 0.277 | 0.669 |
| Random Forest | 0.686 | 0.538 | 0.190 | 0.281 | 0.674 |

## 5. Video de sustentación
Enlace YouTube: _pendiente_
