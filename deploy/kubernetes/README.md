# ML Platform Kubernetes deployment

This directory is the complete source of truth for the local and E2E deployment.

The K3s image applies two pinned Kustomize layers with `kubectl`; Helm is not
required to deploy or operate the platform.

- `bootstrap/` creates every namespace and deploys Spark Operator, Kueue,
  Headlamp, and their required RBAC and custom-resource definitions.
- `platform/` deploys PostgreSQL, MinIO, the control plane, warehouse
  initialization jobs, and the Kueue materialization queue.

[the K3s Dockerfile](../docker/k3s/Dockerfile) creates a self-contained,
self-bootstrapping single-node cluster for E2E tests. It builds the local
control-plane and MinIO images into K3s-importable archives, applies the same
resources in dependency order, and reports Docker health only after all
platform deployments and initialization jobs are ready. E2E tests therefore
only start the image and wait for its health status.

- Spark Operator `2.5.2`, rendered in `bootstrap/spark-operator.yaml`
- Kueue `0.20.1`, rendered in `bootstrap/kueue-controller.yaml`
- Headlamp `0.46.0`, rendered in `bootstrap/headlamp.yaml`, with read-only
  access defined in `bootstrap/headlamp-rbac.yaml`
- PostgreSQL, MinIO, warehouse initialization jobs, the control plane, and
  Kueue queue resources through `platform/kustomization.yaml`

The `bootstrap/*-values.yaml` files record the source chart configuration used
to render the checked-in controller manifests. They are not read during
deployment.

The K3s image builds and imports `ml-platform:latest` and
`ml-platform-minio:latest` before applying these manifests.
