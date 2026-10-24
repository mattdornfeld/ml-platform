#!/usr/bin/env bash
# Install every cluster-scoped and namespace-scoped ML Platform service.
set -euo pipefail

KUBECTL="${KUBECTL:-kubectl}"
HELM="${HELM:-helm}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPLOY_DIR="$ROOT/deploy/kubernetes"

"$KUBECTL" apply -f "$DEPLOY_DIR/namespace.yaml"

"$HELM" repo add kubeflow https://kubeflow.github.io/spark-operator
"$HELM" repo add headlamp https://kubernetes-sigs.github.io/headlamp/
"$HELM" repo update

"$HELM" upgrade --install spark-operator kubeflow/spark-operator \
  --version 2.5.2 \
  --namespace spark-operator \
  --create-namespace \
  --values "$DEPLOY_DIR/spark-operator-values.yaml" \
  --wait

"$HELM" upgrade --install kueue oci://registry.k8s.io/kueue/charts/kueue \
  --version 0.20.1 \
  --namespace kueue-system \
  --create-namespace \
  --values "$DEPLOY_DIR/kueue-values.yaml" \
  --wait \
  --timeout 300s

"$KUBECTL" create namespace headlamp --dry-run=client -o yaml | "$KUBECTL" apply -f -
"$KUBECTL" apply -f "$DEPLOY_DIR/headlamp-rbac.yaml"
"$HELM" upgrade --install headlamp headlamp/headlamp \
  --version 0.46.0 \
  --namespace headlamp \
  --create-namespace \
  --values "$DEPLOY_DIR/headlamp-values.yaml" \
  --wait

"$KUBECTL" apply -k "$DEPLOY_DIR"
"$KUBECTL" -n ml-platform rollout status deployment/postgres --timeout=180s
"$KUBECTL" -n ml-platform rollout status deployment/minio --timeout=180s
"$KUBECTL" -n ml-platform rollout status deployment/ml-control-plane --timeout=180s
"$KUBECTL" -n ml-platform wait --for=condition=complete job/create-warehouse-bucket --timeout=180s
"$KUBECTL" -n ml-platform wait --for=condition=complete job/configure-public-warehouse --timeout=180s
