# ML Platform Kubernetes deployment

This directory is the complete source of truth for the local and E2E deployment.

`install.sh` installs the cluster services pinned below, then applies the versioned
Kustomize resources in this directory:

- Spark Operator chart `2.5.2`, configured by `spark-operator-values.yaml`
- Kueue chart `0.20.1`, configured by `kueue-values.yaml`
- Headlamp chart `0.46.0`, configured by `headlamp-values.yaml` and
  `headlamp-rbac.yaml`
- PostgreSQL, MinIO, warehouse initialization jobs, the control plane, and Kueue
  queue resources through `kustomization.yaml`

The caller must make `ml-platform:latest` and `ml-platform-minio:latest`
available to the target Kubernetes runtime before invoking `install.sh`.
