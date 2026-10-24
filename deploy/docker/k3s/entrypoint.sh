#!/bin/sh
# Start K3s and install ML Platform resources in dependency order.
set -eu

KUBECTL='/bin/kubectl'
MANIFESTS='/opt/ml-platform/kubernetes'

apply_layer() {
  # Server-side apply avoids client annotation limits for generated CRDs.
  $KUBECTL apply --server-side --force-conflicts -k "$1"
}

wait_for_deployment() {
  $KUBECTL -n "$1" rollout status "deployment/$2" --timeout=180s
}

bootstrap() {
  until $KUBECTL get --raw='/readyz' >/dev/null 2>&1; do
    sleep 1
  done

  apply_layer "$MANIFESTS/bootstrap"
  wait_for_deployment spark-operator spark-operator-controller

  wait_for_deployment kueue-system kueue-controller-manager

  wait_for_deployment headlamp headlamp

  apply_layer "$MANIFESTS/platform"
  wait_for_deployment ml-platform postgres
  wait_for_deployment ml-platform minio
  wait_for_deployment ml-platform ml-control-plane
  $KUBECTL -n ml-platform wait --for=condition=complete \
    job/create-warehouse-bucket --timeout=180s
  $KUBECTL -n ml-platform wait --for=condition=complete \
    job/configure-public-warehouse --timeout=180s

  mkdir -p /run/ml-platform
  touch /run/ml-platform/bootstrap-complete
}

(
  if ! bootstrap; then
    # K3s becomes PID 1 after this wrapper execs it. Terminate the cluster on
    # bootstrap failure so Docker exposes a startup failure instead of a
    # permanently unhealthy container.
    kill 1
  fi
) &

exec /bin/k3s "$@"
