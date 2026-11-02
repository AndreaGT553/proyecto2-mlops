"""
API de inferencia - Proyecto 2 MLOps.

Estrategia de carga del modelo: CACHÉ CON VERIFICACIÓN PERIÓDICA
- Al iniciar, la API pide a MLflow el modelo models:/<MODEL_NAME>@<MODEL_ALIAS>.
- Cada MODEL_REFRESH_SECONDS consulta (llamada liviana) qué versión tiene el alias.
  Si cambió, descarga la nueva versión. Nunca descarga el modelo en cada request.
- POST /reload fuerza la recarga inmediata.
"""
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

import mlflow
import mlflow.sklearn
import pandas as pd
import psycopg2
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import Response
from mlflow import MlflowClient
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               generate_latest)
from pydantic import BaseModel, Field

from preprocessing import INPUT_COLUMNS, build_features

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "diabetes-readmission")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
REFRESH_SECONDS = int(os.getenv("MODEL_REFRESH_SECONDS", "60"))
THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
DB_DSN = os.getenv("DATA_DB_DSN")  # postgresql://user:pass@host:5432/datadb

# ------------------------------------------------------------ métricas -----
REQUESTS = Counter("api_requests_total", "Solicitudes HTTP",
                   ["endpoint", "method", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Latencia por endpoint",
                    ["endpoint"],
                    buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10))
PREDICTIONS = Counter("api_predictions_total", "Predicciones por clase", ["prediction"])
ERRORS = Counter("api_errors_total", "Errores", ["endpoint", "type"])
MODEL_VERSION = Gauge("api_model_version", "Versión del modelo cargado")
MODEL_LOADED = Gauge("api_model_loaded", "1 si hay modelo cargado")
INFERENCE_LOG_ERRORS = Counter("api_inference_log_errors_total",
                               "Fallos al guardar inferencias en la BD")


class ModelHolder:
    def __init__(self):
        self.model = None
        self.version = None
        self.run_id = None
        self.loaded_at = None
        self.last_check = 0.0
        self.error = None
        self.lock = threading.Lock()

    def refresh(self, force=False):
        if not force and time.time() - self.last_check < REFRESH_SECONDS:
            return
        with self.lock:
            if not force and time.time() - self.last_check < REFRESH_SECONDS:
                return
            self.last_check = time.time()
            try:
                client = MlflowClient()
                mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
                if force or mv.version != self.version:
                    model = mlflow.sklearn.load_model(
                        f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
                    self.model, self.version, self.run_id = model, mv.version, mv.run_id
                    self.loaded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    MODEL_VERSION.set(int(mv.version))
                    print(f"Modelo cargado: {MODEL_NAME} v{mv.version}")
                self.error = None
            except Exception as e:  # MLflow caído o sin champion todavía
                self.error = str(e)
                print(f"No se pudo cargar el modelo: {e}")
            MODEL_LOADED.set(1 if self.model is not None else 0)


holder = ModelHolder()


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFLOW_URI)
    holder.refresh(force=True)
    yield


app = FastAPI(title="API Readmisión Diabetes", version="1.0", lifespan=lifespan)


# ------------------------------------------------------------- esquemas ----
class Patient(BaseModel):
    race: Literal["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"] = "Caucasian"
    gender: Literal["Male", "Female"] = "Female"
    age: Literal["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)",
                 "[60-70)", "[70-80)", "[80-90)", "[90-100)"] = "[70-80)"
    admission_type_id: int = Field(1, ge=1, le=8)
    discharge_disposition_id: int = Field(1, ge=1, le=30)
    admission_source_id: int = Field(7, ge=1, le=26)
    time_in_hospital: int = Field(4, ge=1, le=14)
    medical_specialty: str = "InternalMedicine"
    num_lab_procedures: int = Field(44, ge=0, le=200)
    num_procedures: int = Field(1, ge=0, le=10)
    num_medications: int = Field(16, ge=0, le=100)
    number_outpatient: int = Field(0, ge=0, le=100)
    number_emergency: int = Field(0, ge=0, le=100)
    number_inpatient: int = Field(0, ge=0, le=100)
    diag_1: str = "428"
    diag_2: str = "250"
    diag_3: str = "401"
    number_diagnoses: int = Field(9, ge=1, le=20)
    max_glu_serum: Literal["None", "Norm", ">200", ">300"] = "None"
    a1cresult: Literal["None", "Norm", ">7", ">8"] = "None"
    metformin: Literal["No", "Steady", "Up", "Down"] = "No"
    glipizide: Literal["No", "Steady", "Up", "Down"] = "No"
    glyburide: Literal["No", "Steady", "Up", "Down"] = "No"
    insulin: Literal["No", "Steady", "Up", "Down"] = "Steady"
    change: Literal["No", "Ch"] = "Ch"
    diabetesmed: Literal["Yes", "No"] = "Yes"


class PredictionOut(BaseModel):
    request_id: str
    prediction: int
    label: str
    probability: float
    model_name: str
    model_version: Optional[str]
    model_alias: str
    processing_time_ms: float


# ------------------------------------------------------- middleware --------
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    path = request.url.path
    if path == "/metrics":
        return await call_next(request)
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        LATENCY.labels(path).observe(time.perf_counter() - start)
        REQUESTS.labels(path, request.method, str(status)).inc()
        if status >= 400:
            ERRORS.labels(path, str(status)).inc()


# --------------------------------------------------------- persistencia ----
def log_inference(row: dict):
    if not DB_DSN:
        return
    try:
        with psycopg2.connect(DB_DSN) as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO inference.predictions
                   (request_id, input_data, prediction, probability, model_name,
                    model_version, model_alias, latency_ms)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                (row["request_id"], json.dumps(row["input"]), row["prediction"],
                 row["probability"], row["model_name"], row["model_version"],
                 row["model_alias"], row["latency_ms"]))
        conn.close()
    except Exception as e:
        INFERENCE_LOG_ERRORS.inc()
        print(f"Error guardando inferencia: {e}")


# ------------------------------------------------------------ endpoints ----
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": holder.model is not None}


@app.get("/ready")
def ready():
    # Readiness: la API está viva aunque aún no exista modelo champion.
    return {"status": "ready"}


@app.get("/model-info")
def model_info():
    holder.refresh()
    return {"model_name": MODEL_NAME, "model_alias": MODEL_ALIAS,
            "model_version": holder.version, "run_id": holder.run_id,
            "loaded_at": holder.loaded_at,
            "status": "loaded" if holder.model is not None else "not_loaded",
            "last_error": holder.error, "refresh_seconds": REFRESH_SECONDS}


@app.post("/reload")
def reload_model():
    holder.refresh(force=True)
    return model_info()


@app.post("/predict", response_model=PredictionOut)
def predict(patient: Patient, background: BackgroundTasks):
    start = time.perf_counter()
    holder.refresh()
    if holder.model is None:
        raise HTTPException(503, f"No hay modelo productivo disponible: {holder.error}")
    data = patient.model_dump()
    raw = pd.DataFrame([{c: str(data[c]) for c in INPUT_COLUMNS}])
    X = build_features(raw)
    proba = float(holder.model.predict_proba(X)[0, 1])
    pred = int(proba >= THRESHOLD)
    elapsed = (time.perf_counter() - start) * 1000
    PREDICTIONS.labels(str(pred)).inc()
    out = PredictionOut(
        request_id=str(uuid.uuid4()), prediction=pred,
        label="Readmisión < 30 días" if pred else "Sin readmisión temprana",
        probability=round(proba, 4), model_name=MODEL_NAME,
        model_version=holder.version, model_alias=MODEL_ALIAS,
        processing_time_ms=round(elapsed, 2))
    background.add_task(log_inference, {
        "request_id": out.request_id, "input": data, "prediction": pred,
        "probability": proba, "model_name": MODEL_NAME,
        "model_version": holder.version, "model_alias": MODEL_ALIAS,
        "latency_ms": elapsed})
    return out


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
