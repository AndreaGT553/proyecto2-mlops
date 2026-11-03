# Servidor MLflow con soporte PostgreSQL (backend) y S3/MinIO (artefactos)
# sqlalchemy fijado en 2.0.x: SQLAlchemy 2.1 eliminó FallbackAsyncAdaptedQueuePool,
# que MLflow 2.22 importa al conectarse a PostgreSQL.
FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow==2.22.1 sqlalchemy==2.0.36 psycopg2-binary==2.9.10 boto3==1.35.90
EXPOSE 5000
