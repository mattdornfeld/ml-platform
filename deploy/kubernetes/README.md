# ML Platform Kubernetes deployment

This directory is the complete source of truth for the local and E2E deployment.

`install.sh` uses only `kubectl` to apply the pinned Kubernetes resources in
this directory to an existing cluster. Helm is not required to deploy or
operate the platform.

[the K3s Dockerfile](../docker/k3s/Dockerfile) creates a self-bootstrapping
single-node cluster for E2E tests. Its entrypoint applies the same resources in
dependency order, so K3s tests do not invoke `install.sh`.

- Spark Operator `2.5.2`, rendered in `spark-operator.yaml`
- Kueue `0.20.1`, rendered in `kueue-controller.yaml`
- Headlamp `0.46.0`, rendered in `headlamp.yaml`, with read-only access defined
  in `headlamp-rbac.yaml`
- PostgreSQL, MinIO, warehouse initialization jobs, the control plane, and
  Kueue queue resources through `kustomization.yaml`

The `*-values.yaml` files record the source chart configuration used to render
the checked-in manifests. They are not read during deployment.

The caller must make `ml-platform:latest` and `ml-platform-minio:latest`
available to the target Kubernetes runtime before invoking `install.sh`.
