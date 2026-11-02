#!/usr/bin/env bash
# Construye las 5 imágenes propias y las publica en DockerHub.
# Uso:  ./build_push.sh <usuario_dockerhub> [tag]
set -euo pipefail
USER_HUB="${1:?Uso: ./build_push.sh <usuario_dockerhub> [tag]}"
TAG="${2:-1.0}"
PLATFORM="linux/amd64"   # la VM es x86_64 (importante si construyes en un Mac M1/M2/M3)

build() {  # nombre  dockerfile
  echo ">>> $USER_HUB/$1:$TAG"
  docker buildx build --platform "$PLATFORM" -f "$2" -t "$USER_HUB/$1:$TAG" --push .
}
build p2-mlflow         mlflow/Dockerfile
build p2-airflow        airflow/Dockerfile
build p2-inference-api  api/Dockerfile
build p2-streamlit-ui   ui/Dockerfile
build p2-locust         locust/Dockerfile

# Apunta los manifiestos a tu usuario de DockerHub
sed -i.bak "s#DOCKERHUB_USER/#$USER_HUB/#g" kustomization.yaml k8s/*.yaml && rm -f kustomization.yaml.bak k8s/*.bak
echo "Listo. Imágenes publicadas y manifiestos actualizados con $USER_HUB"
