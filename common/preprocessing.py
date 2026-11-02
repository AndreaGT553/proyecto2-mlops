"""
Lógica de limpieza e ingeniería de características compartida.

Este mismo archivo se copia en la imagen de Airflow (entrenamiento) y en la
imagen de la API (inferencia) para garantizar que los datos se transforman
EXACTAMENTE igual en ambos lados (evita "training/serving skew").
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Variable objetivo
# readmitted: 'NO', '>30', '<30'  ->  1 si fue readmitido en menos de 30 días
# ---------------------------------------------------------------------------
TARGET_RAW = "readmitted"
TARGET = "readmitted_30"

# Disposiciones de alta que significan fallecimiento u hospicio:
# el paciente no puede ser readmitido, por lo que se excluyen.
EXCLUDED_DISCHARGE = {"11", "13", "14", "19", "20", "21"}

TOP_SPECIALTIES = [
    "InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice",
    "Cardiology", "Surgery-General", "Nephrology", "Orthopedics",
    "Orthopedics-Reconstructive", "Radiologist", "Pulmonology",
]

# Columnas que llegan desde el cliente (API) o desde la tabla raw
INPUT_NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
INPUT_CATEGORICAL = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "medical_specialty", "diag_1", "diag_2", "diag_3",
    "max_glu_serum", "a1cresult", "metformin", "glipizide", "glyburide",
    "insulin", "change", "diabetesmed",
]
INPUT_COLUMNS = INPUT_NUMERIC + INPUT_CATEGORICAL

# Columnas que usa el modelo después de la ingeniería de características
MODEL_NUMERIC = INPUT_NUMERIC + ["age_num", "total_visits"]
MODEL_CATEGORICAL = [
    "race", "gender", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "medical_specialty", "diag_1_group",
    "diag_2_group", "diag_3_group", "max_glu_serum", "a1cresult",
    "metformin", "glipizide", "glyburide", "insulin", "change", "diabetesmed",
]
FEATURES = MODEL_NUMERIC + MODEL_CATEGORICAL


def icd9_group(code) -> str:
    """Agrupa los códigos ICD-9 de diagnóstico en categorías clínicas."""
    if code is None:
        return "Missing"
    code = str(code).strip()
    if code in ("", "?", "nan", "None"):
        return "Missing"
    if code[0] in ("V", "E"):
        return "Other"
    try:
        v = float(code)
    except ValueError:
        return "Other"
    if 390 <= v <= 459 or v == 785:
        return "Circulatory"
    if 460 <= v <= 519 or v == 786:
        return "Respiratory"
    if 520 <= v <= 579 or v == 787:
        return "Digestive"
    if int(v) == 250:
        return "Diabetes"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or v == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


def age_to_num(age) -> float:
    """'[70-80)' -> 75.0 (punto medio del rango)."""
    try:
        s = str(age).strip("[)")
        lo, hi = s.split("-")
        return (float(lo) + float(hi)) / 2
    except Exception:
        return np.nan


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma columnas de entrada (raw) en las columnas del modelo.

    No elimina filas: se usa igual en entrenamiento y en inferencia.
    """
    out = pd.DataFrame(index=df.index)
    for c in INPUT_NUMERIC:
        out[c] = pd.to_numeric(df[c], errors="coerce")

    def clean_cat(s):
        s = s.astype(str).str.strip()
        return s.replace({"?": "Missing", "": "Missing", "nan": "Missing",
                          "None": "None"})

    out["race"] = clean_cat(df["race"])
    out["gender"] = clean_cat(df["gender"])
    out["age_num"] = df["age"].map(age_to_num)
    for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        out[c] = df[c].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    spec = clean_cat(df["medical_specialty"])
    out["medical_specialty"] = spec.where(
        spec.isin(TOP_SPECIALTIES + ["Missing"]), "Other")
    for i in (1, 2, 3):
        out[f"diag_{i}_group"] = df[f"diag_{i}"].map(icd9_group)
    # En el CSV, 'None' significa "no se hizo la prueba" -> categoría válida
    for c in ["max_glu_serum", "a1cresult"]:
        out[c] = df[c].fillna("None").astype(str).replace({"": "None", "nan": "None"})
    for c in ["metformin", "glipizide", "glyburide", "insulin", "change", "diabetesmed"]:
        out[c] = clean_cat(df[c])
    out["total_visits"] = (out["number_outpatient"].fillna(0)
                           + out["number_emergency"].fillna(0)
                           + out["number_inpatient"].fillna(0))
    return out[FEATURES]


def clean_training_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Limpieza para ENTRENAMIENTO: filtra filas inválidas y crea el target."""
    df = raw.copy()
    df = df[df["gender"] != "Unknown/Invalid"]
    df = df[~df["discharge_disposition_id"].astype(str).isin(EXCLUDED_DISCHARGE)]
    df = df[df[TARGET_RAW].isin(["NO", ">30", "<30"])]
    feats = build_features(df)
    feats[TARGET] = (df[TARGET_RAW] == "<30").astype(int)
    return feats
