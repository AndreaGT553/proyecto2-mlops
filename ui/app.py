"""Interfaz gráfica: se comunica ÚNICAMENTE con la API de inferencia."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://inference-api:8000")

st.set_page_config(page_title="Readmisión Diabetes", page_icon="🩺")
st.title("🩺 Predicción de readmisión hospitalaria (< 30 días)")
st.caption(f"API: {API_URL}")

EXAMPLES = {
    "Paciente de bajo riesgo": dict(race="Caucasian", gender="Female", age="[50-60)",
        admission_type_id=3, discharge_disposition_id=1, admission_source_id=1,
        time_in_hospital=2, medical_specialty="Surgery-General", num_lab_procedures=30,
        num_procedures=2, num_medications=10, number_outpatient=0, number_emergency=0,
        number_inpatient=0, diag_1="715", diag_2="250", diag_3="401", number_diagnoses=5,
        max_glu_serum="None", a1cresult="None", metformin="Steady", glipizide="No",
        glyburide="No", insulin="No", change="No", diabetesmed="Yes"),
    "Paciente de alto riesgo": dict(race="AfricanAmerican", gender="Male", age="[80-90)",
        admission_type_id=1, discharge_disposition_id=3, admission_source_id=7,
        time_in_hospital=9, medical_specialty="InternalMedicine", num_lab_procedures=70,
        num_procedures=0, num_medications=25, number_outpatient=2, number_emergency=3,
        number_inpatient=4, diag_1="428", diag_2="403", diag_3="250", number_diagnoses=9,
        max_glu_serum=">300", a1cresult=">8", metformin="No", glipizide="Steady",
        glyburide="No", insulin="Up", change="Ch", diabetesmed="Yes"),
}
DEFAULT = EXAMPLES["Paciente de bajo riesgo"]

# Inicializa valores por defecto la primera vez
for k, v in DEFAULT.items():
    st.session_state.setdefault(k, v)


# Info del modelo productivo (vía API)
with st.sidebar:
    st.subheader("Modelo productivo")
    try:
        info = requests.get(f"{API_URL}/model-info", timeout=5).json()
        st.write(f"**{info['model_name']}** @{info['model_alias']}")
        st.write(f"Versión: **{info['model_version']}** — {info['status']}")
    except Exception as e:
        st.error(f"API no disponible: {e}")
    choice = st.selectbox("Cargar valores de ejemplo", ["—"] + list(EXAMPLES))
    if st.button("Cargar ejemplo") and choice != "—":
        for k, v in EXAMPLES[choice].items():
            st.session_state[k] = v
        st.rerun()

def val(k):
    return st.session_state.get(k, DEFAULT[k])

AGES = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)",
        "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
MED = ["No", "Steady", "Up", "Down"]

with st.form("patient"):
    c1, c2, c3 = st.columns(3)
    race = c1.selectbox("Raza", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"], key="race")
    gender = c2.selectbox("Género", ["Female", "Male"], key="gender")
    age = c3.selectbox("Edad", AGES, key="age")
    admission_type_id = c1.number_input("Tipo de admisión (id)", 1, 8, key="admission_type_id")
    discharge_disposition_id = c2.number_input("Disposición de alta (id)", 1, 30, key="discharge_disposition_id")
    admission_source_id = c3.number_input("Fuente de admisión (id)", 1, 26, key="admission_source_id")
    time_in_hospital = c1.number_input("Días hospitalizado", 1, 14, key="time_in_hospital")
    num_lab_procedures = c2.number_input("Pruebas de laboratorio", 0, 200, key="num_lab_procedures")
    num_procedures = c3.number_input("Procedimientos", 0, 10, key="num_procedures")
    num_medications = c1.number_input("Medicamentos", 0, 100, key="num_medications")
    number_outpatient = c2.number_input("Visitas ambulatorias (año previo)", 0, 100, key="number_outpatient")
    number_emergency = c3.number_input("Visitas de emergencia (año previo)", 0, 100, key="number_emergency")
    number_inpatient = c1.number_input("Hospitalizaciones (año previo)", 0, 100, key="number_inpatient")
    number_diagnoses = c2.number_input("Número de diagnósticos", 1, 20, key="number_diagnoses")
    medical_specialty = c3.text_input("Especialidad médica", key="medical_specialty")
    diag_1 = c1.text_input("Diagnóstico 1 (ICD-9)", key="diag_1")
    diag_2 = c2.text_input("Diagnóstico 2 (ICD-9)", key="diag_2")
    diag_3 = c3.text_input("Diagnóstico 3 (ICD-9)", key="diag_3")
    max_glu_serum = c1.selectbox("Glucosa sérica", ["None", "Norm", ">200", ">300"], key="max_glu_serum")
    a1cresult = c2.selectbox("HbA1c", ["None", "Norm", ">7", ">8"], key="a1cresult")
    insulin = c3.selectbox("Insulina", MED, key="insulin")
    metformin = c1.selectbox("Metformina", MED, key="metformin")
    glipizide = c2.selectbox("Glipizida", MED, key="glipizide")
    glyburide = c3.selectbox("Gliburida", MED, key="glyburide")
    change = c1.selectbox("Cambio de medicación", ["No", "Ch"], key="change")
    diabetesmed = c2.selectbox("Medicación para diabetes", ["Yes", "No"], key="diabetesmed")
    submitted = st.form_submit_button("Predecir", type="primary")

if submitted:
    payload = {k: st.session_state[k] for k in DEFAULT}
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
    except Exception as e:
        st.error(f"No fue posible conectar con la API: {e}")
    else:
        if r.status_code == 200:
            out = r.json()
            (st.error if out["prediction"] == 1 else st.success)(f"**{out['label']}**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Probabilidad", f"{out['probability']:.1%}")
            m2.metric("Versión del modelo", f"v{out['model_version']} (@{out['model_alias']})")
            m3.metric("Tiempo", f"{out['processing_time_ms']:.0f} ms")
            st.caption(f"request_id: {out['request_id']}")
        elif r.status_code == 422:
            st.warning("Datos de entrada no válidos:")
            for err in r.json().get("detail", []):
                st.write(f"- **{err['loc'][-1]}**: {err['msg']}")
        else:
            st.error(f"Error {r.status_code}: {r.text}")
