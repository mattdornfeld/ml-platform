#!/usr/bin/env bash
set -euo pipefail
minikube start --driver=docker --cpus=8 --memory=49152
helm repo add kubeflow https://kubeflow.github.io/spark-operator
helm repo add headlamp https://kubernetes-sigs.github.io/headlamp/
helm repo update
helm upgrade --install spark-operator kubeflow/spark-operator --namespace spark-operator --create-namespace --set 'spark.jobNamespaces[0]=ml-platform' --wait
kubectl create namespace headlamp --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f deploy/kubernetes/headlamp-rbac.yaml
helm upgrade --install headlamp headlamp/headlamp --version 0.46.0 --namespace headlamp --create-namespace --values deploy/kubernetes/headlamp-values.yaml --wait
helm upgrade --install kueue oci://registry.k8s.io/kueue/charts/kueue \
  --version 0.20.1 --namespace kueue-system --create-namespace \
  --values deploy/kubernetes/kueue-values.yaml --wait --timeout 300s
minikube image build -t ml-platform:latest .
minikube image build -t ml-platform-minio:latest -f deploy/docker/minio/Dockerfile .
kubectl apply -k deploy/kubernetes
kubectl -n ml-platform rollout status deployment/postgres --timeout=180s
kubectl -n ml-platform rollout status deployment/minio --timeout=180s
kubectl -n ml-platform rollout status deployment/ml-control-plane --timeout=180s
