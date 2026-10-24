#!/usr/bin/env bash
# Install every versioned ML Platform Kubernetes resource with kubectl.
set -euo pipefail

KUBECTL="${KUBECTL:-kubectl}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPLOY_DIR="$ROOT/deploy/kubernetes"

"$KUBECTL" apply -f "$DEPLOY_DIR/namespace.yaml"
"$KUBECTL" create namespace spark-operator --dry-run=client -o yaml | "$KUBECTL" apply -f -
"$KUBECTL" --namespace spark-operator apply --server-side --force-conflicts \
  -f "$DEPLOY_DIR/spark-operator.yaml"
"$KUBECTL" apply --server-side --force-conflicts \
  -f "$DEPLOY_DIR/spark-operator-ml-platform.yaml"
"$KUBECTL" -n spark-operator rollout status \
  deployment/spark-operator-controller --timeout=180s

"$KUBECTL" create namespace headlamp --dry-run=client -o yaml | "$KUBECTL" apply -f -
"$KUBECTL" apply -f "$DEPLOY_DIR/headlamp-rbac.yaml"
"$KUBECTL" create namespace kueue-system --dry-run=client -o yaml | "$KUBECTL" apply -f -
"$KUBECTL" apply --server-side --force-conflicts \
  -f "$DEPLOY_DIR/kueue-controller.yaml"
"$KUBECTL" -n kueue-system rollout status \
  deployment/kueue-controller-manager --timeout=180s
"$KUBECTL" apply -f "$DEPLOY_DIR/headlamp.yaml"
"$KUBECTL" -n headlamp rollout status deployment/headlamp --timeout=180s

"$KUBECTL" apply -k "$DEPLOY_DIR"
"$KUBECTL" -n ml-platform rollout status deployment/postgres --timeout=180s
"$KUBECTL" -n ml-platform rollout status deployment/minio --timeout=180s
"$KUBECTL" -n ml-platform rollout status deployment/ml-control-plane --timeout=180s
"$KUBECTL" -n ml-platform wait --for=condition=complete job/create-warehouse-bucket --timeout=180s
"$KUBECTL" -n ml-platform wait --for=condition=complete job/configure-public-warehouse --timeout=180s
