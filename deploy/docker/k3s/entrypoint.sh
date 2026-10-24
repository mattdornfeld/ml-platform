#!/bin/sh
# Start K3s and install ML Platform resources in dependency order.
set -eu

KUBECTL='/bin/kubectl'
MANIFESTS='/opt/ml-platform/kubernetes'

apply_manifest() {
  # Server-side apply avoids client annotation limits for generated CRDs.
  $KUBECTL apply --server-side --force-conflicts -f "$1"
}

apply_namespaced_manifest() {
  # Generated chart resources without a namespace target this explicitly created namespace.
  $KUBECTL --namespace "$1" apply --server-side --force-conflicts -f "$2"
}

create_namespace() {
  $KUBECTL create namespace "$1" --dry-run=client -o yaml | $KUBECTL apply -f -
}

wait_for_deployment() {
  $KUBECTL -n "$1" rollout status "deployment/$2" --timeout=180s
}

bootstrap() {
  until $KUBECTL get --raw='/readyz' >/dev/null 2>&1; do
    sleep 1
  done

  apply_manifest "$MANIFESTS/namespace.yaml"

  create_namespace spark-operator
  apply_namespaced_manifest spark-operator "$MANIFESTS/spark-operator.yaml"
  apply_manifest "$MANIFESTS/spark-operator-ml-platform.yaml"
  wait_for_deployment spark-operator spark-operator-controller

  create_namespace kueue-system
  apply_manifest "$MANIFESTS/kueue-controller.yaml"
  wait_for_deployment kueue-system kueue-controller-manager

  create_namespace headlamp
  apply_manifest "$MANIFESTS/headlamp-rbac.yaml"
  apply_manifest "$MANIFESTS/headlamp.yaml"
  wait_for_deployment headlamp headlamp

  $KUBECTL apply -k "$MANIFESTS"
  wait_for_deployment ml-platform postgres
  wait_for_deployment ml-platform minio
  wait_for_deployment ml-platform ml-control-plane
  $KUBECTL -n ml-platform wait --for=condition=complete \
    job/create-warehouse-bucket --timeout=180s
  $KUBECTL -n ml-platform wait --for=condition=complete \
    job/configure-public-warehouse --timeout=180s
}

bootstrap &
exec /bin/k3s "$@"
