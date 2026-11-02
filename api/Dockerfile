# Construir desde la RAÍZ del repo:  docker build -f api/Dockerfile .
FROM python:3.11-slim
WORKDIR /app
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY common/preprocessing.py .
COPY api/main.py .
EXPOSE 8000
# Un solo worker: las métricas de prometheus_client viven en memoria del proceso
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
