"""
DAG principal del Proyecto 2 - MLOps (Diabetes 130-US hospitals).

Flujo (10 tareas):
 1. check_source_file      -> valida que el CSV exista y tenga las columnas esperadas
 2. load_raw_batch         -> carga el SIGUIENTE lote (<= 15.000 filas) en raw.diabetes
 3. validate_quality       -> marca filas 'validated' / 'rejected'
 4. preprocess_data        -> limpieza + ingeniería de características
 5. store_clean_data       -> guarda el lote limpio en clean.diabetes
 6. split_data             -> train / val / test (asignación estable por hash)
 7. train_models           -> entrena LogisticRegression y RandomForest
 8. register_best_model    -> registra el mejor run en el Model Registry de MLflow
 9. compare_with_champion  -> compara contra el modelo productivo actual
10. promote_model          -> asigna el alias 'champion' si el nuevo es mejor

Cada ejecución procesa UN lote. 101.766 filas / 15.000 = 7 ejecuciones.
Cuando ya no quedan filas, la tarea 2 se marca como "skipped" y el resto también.
"""
import hashlib
import json
import os
from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import PythonOperator
from sqlalchemy import create_engine, text

from preprocessing import (FEATURES, INPUT_COLUMNS, MODEL_CATEGORICAL,
                           MODEL_NUMERIC, TARGET, clean_training_frame)

# ---------------------------------------------------------------- config ---
DATA_DB_URI = os.environ["DATA_DB_URI"]
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
SOURCE_FILE = os.environ.get("SOURCE_FILE", "/opt/airflow/data/diabetic_data.csv")
BATCH_SIZE = min(int(os.environ.get("BATCH_SIZE", "15000")), 15000)
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "diabetes_readmission")
MODEL_NAME = os.environ.get("MODEL_NAME", "diabetes-readmission")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
PRIMARY_METRIC = "f1"  # F1 de la clase positiva (readmisión < 30 días)
WORK_DIR = "/opt/airflow/tmp"

EXPECTED_COLUMNS = {
    "encounter_id", "patient_nbr", "race", "gender", "age", "readmitted",
    "time_in_hospital", "num_medications", "diag_1", "a1cresult",
}


def engine():
    return create_engine(DATA_DB_URI, pool_pre_ping=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace("-", "_") for c in df.columns]
    return df


def read_source() -> pd.DataFrame:
    # keep_default_na=False conserva textos como 'None' o '?' tal como vienen
    df = pd.read_csv(SOURCE_FILE, dtype=str, keep_default_na=False)
    df = normalize_columns(df)
    df.insert(0, "source_row", range(1, len(df) + 1))
    return df


# ------------------------------------------------------------------ tareas -
def check_source_file(**_):
    if not os.path.isfile(SOURCE_FILE):
        raise FileNotFoundError(f"No existe el archivo fuente: {SOURCE_FILE}")
    header = pd.read_csv(SOURCE_FILE, nrows=0)
    cols = set(normalize_columns(header).columns)
    missing = EXPECTED_COLUMNS - cols
    if missing:
        raise ValueError(f"Faltan columnas en el archivo fuente: {missing}")
    print(f"Archivo OK: {SOURCE_FILE} con {len(cols)} columnas")


def load_raw_batch(ti, run_id, **_):
    eng = engine()
    source_name = os.path.basename(SOURCE_FILE)
    with eng.begin() as conn:
        # Idempotencia: si esta misma ejecución ya cargó su lote, se reutiliza
        prev = conn.execute(text(
            "SELECT batch_id FROM raw.diabetes WHERE dag_run_id = :r LIMIT 1"),
            {"r": run_id}).scalar()
        if prev:
            print(f"La ejecución {run_id} ya cargó el lote {prev}; no se duplica.")
            return prev
        last = conn.execute(text(
            "SELECT COALESCE(MAX(source_row), 0) FROM raw.diabetes "
            "WHERE source_file = :f"), {"f": source_name}).scalar()

    df = read_source()
    batch = df[df["source_row"] > last].head(BATCH_SIZE).copy()
    if batch.empty:
        raise AirflowSkipException("No hay filas nuevas en el archivo fuente.")

    first, end = int(batch.source_row.min()), int(batch.source_row.max())
    batch_id = f"batch_{first:06d}_{end:06d}"
    data_cols = [c for c in df.columns if c != "source_row"]
    batch["row_hash"] = batch[data_cols].astype(str).agg("|".join, axis=1).map(
        lambda s: hashlib.sha256(s.encode()).hexdigest())
    batch["batch_id"] = batch_id
    batch["source_file"] = source_name
    batch["dag_run_id"] = run_id
    batch["status"] = "loaded"
    batch["load_timestamp"] = datetime.utcnow()

    # Carga a tabla temporal y luego INSERT ... ON CONFLICT DO NOTHING (row_hash único)
    with eng.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS raw._staging"))
        batch.to_sql("_staging", conn, schema="raw", index=False)
        cols = ", ".join(f'"{c}"' for c in batch.columns)
        inserted = conn.execute(text(
            f"INSERT INTO raw.diabetes ({cols}) SELECT {cols} FROM raw._staging "
            "ON CONFLICT (row_hash) DO NOTHING")).rowcount
        conn.execute(text("DROP TABLE raw._staging"))
    print(f"{batch_id}: {len(batch)} filas leídas, {inserted} insertadas en raw.diabetes")
    return batch_id


def validate_quality(ti, **_):
    batch_id = ti.xcom_pull(task_ids="load_raw_batch")
    with engine().begin() as conn:
        total = conn.execute(text(
            "SELECT COUNT(*) FROM raw.diabetes WHERE batch_id=:b"), {"b": batch_id}).scalar()
        if total == 0:
            raise ValueError(f"El lote {batch_id} está vacío")
        rejected = conn.execute(text("""
            UPDATE raw.diabetes SET status='rejected'
            WHERE batch_id=:b AND (
                readmitted NOT IN ('NO','>30','<30')
                OR gender NOT IN ('Male','Female')
                OR time_in_hospital !~ '^[0-9]+$'
                OR num_medications !~ '^[0-9]+$')
            RETURNING raw_id"""), {"b": batch_id}).rowcount
        conn.execute(text(
            "UPDATE raw.diabetes SET status='validated' "
            "WHERE batch_id=:b AND status<>'rejected'"), {"b": batch_id})
    ratio = rejected / total
    print(f"{batch_id}: {total} filas, {rejected} rechazadas ({ratio:.2%})")
    if ratio > 0.2:
        raise ValueError("Más del 20% del lote no pasó la validación de calidad")


def preprocess_data(ti, run_id, **_):
    batch_id = ti.xcom_pull(task_ids="load_raw_batch")
    raw = pd.read_sql(text(
        "SELECT * FROM raw.diabetes WHERE batch_id=:b AND status='validated'"),
        engine(), params={"b": batch_id})
    clean = clean_training_frame(raw.set_index("raw_id"))
    clean["raw_id"] = clean.index
    clean["batch_id"] = batch_id
    # Asignación estable de split: un registro siempre cae en el mismo subconjunto
    bucket = raw.set_index("raw_id").loc[clean.index, "row_hash"].map(
        lambda h: int(h[:8], 16) % 100)
    clean["split"] = pd.cut(bucket, [-1, 69, 84, 99],
                            labels=["train", "val", "test"]).astype(str)
    os.makedirs(WORK_DIR, exist_ok=True)
    path = f"{WORK_DIR}/{batch_id}_clean.csv"
    clean.to_csv(path, index=False)
    print(f"{batch_id}: {len(raw)} filas validadas -> {len(clean)} filas limpias")
    return path


def store_clean_data(ti, **_):
    batch_id = ti.xcom_pull(task_ids="load_raw_batch")
    path = ti.xcom_pull(task_ids="preprocess_data")
    clean = pd.read_csv(path, dtype={c: str for c in MODEL_CATEGORICAL})
    clean["processed_at"] = datetime.utcnow()
    with engine().begin() as conn:
        # Borrar y reinsertar el lote -> reejecutar la tarea no duplica filas
        conn.execute(text("DELETE FROM clean.diabetes WHERE batch_id=:b"), {"b": batch_id})
        clean.to_sql("diabetes", conn, schema="clean", if_exists="append", index=False)
        conn.execute(text(
            "UPDATE raw.diabetes SET status='processed' WHERE batch_id=:b "
            "AND status='validated'"), {"b": batch_id})
    print(f"{len(clean)} filas guardadas en clean.diabetes")


def split_data(run_id, **_):
    df = pd.read_sql("SELECT * FROM clean.diabetes", engine())
    out = {}
    safe = run_id.replace(":", "_").replace("+", "_")
    for name in ["train", "val", "test"]:
        part = df[df["split"] == name]
        path = f"{WORK_DIR}/{safe}_{name}.csv"
        part[FEATURES + [TARGET]].to_csv(path, index=False)
        out[name] = path
        print(f"{name}: {len(part)} filas, positivos={part[TARGET].mean():.3f}")
    out["n_rows"] = len(df)
    out["n_batches"] = int(df["batch_id"].nunique())
    return out


def _load_split(path):
    df = pd.read_csv(path, dtype={c: str for c in MODEL_CATEGORICAL})
    return df[FEATURES], df[TARGET]


def _metrics(y, proba, threshold=0.5):
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    pred = (proba >= threshold).astype(int)
    return {
        "f1": f1_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "accuracy": accuracy_score(y, pred),
    }


def train_models(ti, run_id, **_):
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    paths = ti.xcom_pull(task_ids="split_data")
    batch_id = ti.xcom_pull(task_ids="load_raw_batch")
    X_tr, y_tr = _load_split(paths["train"])
    X_va, y_va = _load_split(paths["val"])
    X_te, y_te = _load_split(paths["test"])

    def preprocessor():
        return ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), MODEL_NUMERIC),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="constant",
                                                    fill_value="Missing")),
                              ("oh", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False))]),
             MODEL_CATEGORICAL),
        ])

    candidates = {
        "logistic_regression": (LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=1000), {"C": 0.5}),
        "random_forest": (RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=20,
            class_weight="balanced_subsample", n_jobs=2, random_state=42),
            {"n_estimators": 150, "max_depth": 12, "min_samples_leaf": 20}),
    }

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    results = []
    for name, (clf, params) in candidates.items():
        pipe = Pipeline([("prep", preprocessor()), ("clf", clf)])
        with mlflow.start_run(run_name=f"{name}_{batch_id}") as run:
            pipe.fit(X_tr, y_tr)
            mlflow.log_params({"model_type": name, "class_weight": "balanced",
                               "threshold": 0.5, **params})
            mlflow.log_params({"n_train": len(X_tr), "n_val": len(X_va),
                               "n_test": len(X_te), "last_batch": batch_id,
                               "n_batches": paths["n_batches"],
                               "primary_metric": PRIMARY_METRIC})
            for split, X, y in [("train", X_tr, y_tr), ("val", X_va, y_va),
                                ("test", X_te, y_te)]:
                m = _metrics(y, pipe.predict_proba(X)[:, 1])
                mlflow.log_metrics({f"{split}_{k}": v for k, v in m.items()})
            val_pred = pipe.predict(X_va)
            mlflow.log_text(classification_report(y_va, val_pred, digits=4),
                            "classification_report_val.txt")
            mlflow.log_dict({"confusion_matrix_val": confusion_matrix(
                y_va, val_pred).tolist()}, "confusion_matrix_val.json")
            mlflow.log_dict({"features": FEATURES}, "features.json")
            mlflow.set_tags({"dag_run_id": run_id, "batch_id": batch_id})
            sample = X_va.head(5)
            mlflow.sklearn.log_model(
                pipe, "model", input_example=sample,
                signature=infer_signature(sample, pipe.predict_proba(sample)[:, 1]))
            val_f1 = _metrics(y_va, pipe.predict_proba(X_va)[:, 1])["f1"]
            results.append({"run_id": run.info.run_id, "model": name,
                            "val_f1": val_f1})
            print(f"{name}: val_f1={val_f1:.4f} run={run.info.run_id}")

    best = max(results, key=lambda r: r["val_f1"])
    print(f"Mejor modelo de esta ejecución: {best}")
    return best


def register_best_model(ti, **_):
    import mlflow
    from mlflow import MlflowClient
    best = ti.xcom_pull(task_ids="train_models")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mv = mlflow.register_model(f"runs:/{best['run_id']}/model", MODEL_NAME)
    client = MlflowClient()
    client.set_model_version_tag(MODEL_NAME, mv.version, "model_type", best["model"])
    client.set_model_version_tag(MODEL_NAME, mv.version, "val_f1", f"{best['val_f1']:.4f}")
    print(f"Registrado {MODEL_NAME} versión {mv.version}")
    return {"version": mv.version, **best}


def compare_with_champion(ti, **_):
    """Evalúa el candidato y el champion actual sobre el MISMO conjunto de test."""
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    cand = ti.xcom_pull(task_ids="register_best_model")
    paths = ti.xcom_pull(task_ids="split_data")
    X_te, y_te = _load_split(paths["test"])

    cand_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{cand['version']}")
    cand_f1 = _metrics(y_te, cand_model.predict_proba(X_te)[:, 1])[PRIMARY_METRIC]
    try:
        champ = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        champ_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        champ_f1 = _metrics(y_te, champ_model.predict_proba(X_te)[:, 1])[PRIMARY_METRIC]
        champ_version = champ.version
    except Exception as e:  # no existe champion todavía
        print(f"No hay champion previo ({e})")
        champ_f1, champ_version = -1.0, None
    decision = cand_f1 >= champ_f1
    result = {"candidate_version": cand["version"], "candidate_test_f1": cand_f1,
              "champion_version": champ_version, "champion_test_f1": champ_f1,
              "promote": bool(decision)}
    client.set_model_version_tag(MODEL_NAME, cand["version"],
                                 "comparison", json.dumps(result))
    print(json.dumps(result, indent=2))
    return result


def promote_model(ti, **_):
    import mlflow
    from mlflow import MlflowClient
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    res = ti.xcom_pull(task_ids="compare_with_champion")
    if res["promote"]:
        client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS,
                                          res["candidate_version"])
        client.set_model_version_tag(MODEL_NAME, res["candidate_version"],
                                     "status", "production")
        print(f"Versión {res['candidate_version']} promovida a @{MODEL_ALIAS}")
    else:
        client.set_model_version_tag(MODEL_NAME, res["candidate_version"],
                                     "status", "rejected")
        print(f"Se mantiene el champion v{res['champion_version']}")


# --------------------------------------------------------------------- DAG -
default_args = {"owner": "andrea", "retries": 1, "retry_delay": timedelta(minutes=1)}

with DAG(
    dag_id="diabetes_mlops_pipeline",
    description="Ingesta incremental, limpieza, entrenamiento y promoción en MLflow",
    start_date=datetime(2026, 11, 1),
    schedule="*/10 * * * *",   # cada 10 min llega un nuevo lote
    catchup=False,
    max_active_runs=1,          # nunca dos lotes en paralelo
    default_args=default_args,
    tags=["proyecto2", "mlops"],
) as dag:
    t1 = PythonOperator(task_id="check_source_file", python_callable=check_source_file)
    t2 = PythonOperator(task_id="load_raw_batch", python_callable=load_raw_batch)
    t3 = PythonOperator(task_id="validate_quality", python_callable=validate_quality)
    t4 = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)
    t5 = PythonOperator(task_id="store_clean_data", python_callable=store_clean_data)
    t6 = PythonOperator(task_id="split_data", python_callable=split_data)
    t7 = PythonOperator(task_id="train_models", python_callable=train_models)
    t8 = PythonOperator(task_id="register_best_model", python_callable=register_best_model)
    t9 = PythonOperator(task_id="compare_with_champion", python_callable=compare_with_champion)
    t10 = PythonOperator(task_id="promote_model", python_callable=promote_model)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7 >> t8 >> t9 >> t10
