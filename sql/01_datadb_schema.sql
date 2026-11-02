-- =====================================================================
-- Base de datos de DATOS (datadb). Tres capas separadas por esquema:
--   raw.*        datos tal como llegan del archivo (todo TEXT, sin transformar)
--   clean.*      datos limpios y con ingeniería de características
--   inference.*  registro de cada predicción hecha por la API
-- Este script lo ejecuta PostgreSQL automáticamente la primera vez que arranca
-- (montado en /docker-entrypoint-initdb.d desde un ConfigMap).
-- =====================================================================
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS clean;
CREATE SCHEMA IF NOT EXISTS inference;

-- ------------------------------------------------------------ RAW DATA ----
CREATE TABLE IF NOT EXISTS raw.diabetes (
    raw_id          BIGSERIAL PRIMARY KEY,          -- identificador único
    batch_id        TEXT        NOT NULL,           -- lote de carga
    load_timestamp  TIMESTAMP   NOT NULL DEFAULT now(), -- fecha/hora de carga
    source_file     TEXT        NOT NULL,           -- origen del dato
    source_row      INTEGER     NOT NULL,           -- fila en el archivo origen
    row_hash        TEXT        NOT NULL UNIQUE,    -- control de duplicados
    status          TEXT        NOT NULL DEFAULT 'loaded', -- loaded|validated|rejected|processed
    dag_run_id      TEXT,                           -- ejecución de Airflow que lo cargó
    encounter_id               TEXT,
    patient_nbr                TEXT,
    race                       TEXT,
    gender                     TEXT,
    age                        TEXT,
    weight                     TEXT,
    admission_type_id          TEXT,
    discharge_disposition_id   TEXT,
    admission_source_id        TEXT,
    time_in_hospital           TEXT,
    payer_code                 TEXT,
    medical_specialty          TEXT,
    num_lab_procedures         TEXT,
    num_procedures             TEXT,
    num_medications            TEXT,
    number_outpatient          TEXT,
    number_emergency           TEXT,
    number_inpatient           TEXT,
    diag_1                     TEXT,
    diag_2                     TEXT,
    diag_3                     TEXT,
    number_diagnoses           TEXT,
    max_glu_serum              TEXT,
    a1cresult                  TEXT,
    metformin                  TEXT,
    repaglinide                TEXT,
    nateglinide                TEXT,
    chlorpropamide             TEXT,
    glimepiride                TEXT,
    acetohexamide              TEXT,
    glipizide                  TEXT,
    glyburide                  TEXT,
    tolbutamide                TEXT,
    pioglitazone               TEXT,
    rosiglitazone              TEXT,
    acarbose                   TEXT,
    miglitol                   TEXT,
    troglitazone               TEXT,
    tolazamide                 TEXT,
    examide                    TEXT,
    citoglipton                TEXT,
    insulin                    TEXT,
    glyburide_metformin        TEXT,
    glipizide_metformin        TEXT,
    glimepiride_pioglitazone   TEXT,
    metformin_rosiglitazone    TEXT,
    metformin_pioglitazone     TEXT,
    change                     TEXT,
    diabetesmed                TEXT,
    readmitted                 TEXT
);
CREATE INDEX IF NOT EXISTS ix_raw_batch  ON raw.diabetes (batch_id);
CREATE INDEX IF NOT EXISTS ix_raw_source ON raw.diabetes (source_file, source_row);

-- ---------------------------------------------------------- CLEAN DATA ----
CREATE TABLE IF NOT EXISTS clean.diabetes (
    time_in_hospital          DOUBLE PRECISION,
    num_lab_procedures        DOUBLE PRECISION,
    num_procedures            DOUBLE PRECISION,
    num_medications           DOUBLE PRECISION,
    number_outpatient         DOUBLE PRECISION,
    number_emergency          DOUBLE PRECISION,
    number_inpatient          DOUBLE PRECISION,
    number_diagnoses          DOUBLE PRECISION,
    age_num                   DOUBLE PRECISION,
    total_visits              DOUBLE PRECISION,
    race                      TEXT,
    gender                    TEXT,
    admission_type_id         TEXT,
    discharge_disposition_id  TEXT,
    admission_source_id       TEXT,
    medical_specialty         TEXT,
    diag_1_group              TEXT,
    diag_2_group              TEXT,
    diag_3_group              TEXT,
    max_glu_serum             TEXT,
    a1cresult                 TEXT,
    metformin                 TEXT,
    glipizide                 TEXT,
    glyburide                 TEXT,
    insulin                   TEXT,
    change                    TEXT,
    diabetesmed               TEXT,
    readmitted_30             INTEGER NOT NULL,   -- target: 1 = readmitido < 30 días
    raw_id                    BIGINT  NOT NULL UNIQUE REFERENCES raw.diabetes(raw_id),
    batch_id                  TEXT    NOT NULL,
    split                     TEXT    NOT NULL,   -- train | val | test
    processed_at              TIMESTAMP NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_clean_batch ON clean.diabetes (batch_id);

-- ---------------------------------------------------- INFERENCE LOGS ----
CREATE TABLE IF NOT EXISTS inference.predictions (
    id              BIGSERIAL PRIMARY KEY,
    request_id      TEXT        NOT NULL UNIQUE,  -- identificador de la solicitud
    created_at      TIMESTAMP   NOT NULL DEFAULT now(), -- fecha/hora de la inferencia
    input_data      JSONB       NOT NULL,         -- datos de entrada recibidos
    prediction      INTEGER     NOT NULL,         -- predicción generada
    probability     DOUBLE PRECISION,             -- score del modelo
    model_name      TEXT        NOT NULL,
    model_version   TEXT        NOT NULL,
    model_alias     TEXT,
    latency_ms      DOUBLE PRECISION NOT NULL     -- tiempo de respuesta
);
CREATE INDEX IF NOT EXISTS ix_inf_created ON inference.predictions (created_at);
