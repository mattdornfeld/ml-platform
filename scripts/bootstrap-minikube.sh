#!/usr/bin/env bash
set -euo pipefail
minikube start --driver=docker --cpus=8 --memory=49152
helm repo add kubeflow https://kubeflow.github.io/spark-operator
helm repo update
helm upgrade --install spark-operator kubeflow/spark-operator --namespace spark-operator --create-namespace --set 'spark.jobNamespaces[0]=materialization' --wait
helm upgrade --install kueue oci://registry.k8s.io/kueue/charts/kueue \
  --version 0.20.1 --namespace kueue-system --create-namespace \
  --values deploy/kubernetes/kueue-values.yaml --wait --timeout 300s
minikube image build -t ml-platform:latest .
minikube image build -t ml-platform-minio:latest -f deploy/docker/minio/Dockerfile .
kubectl apply -k deploy/kubernetes
kubectl -n materialization rollout status deployment/postgres --timeout=180s
kubectl -n materialization rollout status deployment/minio --timeout=180s
kubectl -n materialization rollout status deployment/api --timeout=180s
