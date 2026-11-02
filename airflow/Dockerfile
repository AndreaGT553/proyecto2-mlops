# Construir desde la RAÍZ del repo:  docker build -f airflow/Dockerfile .
FROM apache/airflow:2.10.4-python3.11
COPY airflow/requirements.txt /requirements.txt
RUN pip install --no-cache-dir "apache-airflow==2.10.4" -r /requirements.txt
# DAG + módulo compartido de preprocesamiento (ambos quedan en el PYTHONPATH de dags)
COPY airflow/dags/ /opt/airflow/dags/
COPY common/preprocessing.py /opt/airflow/dags/preprocessing.py
# Archivo fuente dentro de la imagen (el DAG valida que exista)
COPY data/diabetic_data.csv /opt/airflow/data/diabetic_data.csv
USER root
RUN mkdir -p /opt/airflow/tmp && chown -R airflow: /opt/airflow/tmp /opt/airflow/data
USER airflow
