"""External read-only dashboard links for control-plane resources."""

from datetime import datetime
from urllib.parse import quote

from kubernetes import client as kubernetes_client
from kubernetes.client.exceptions import ApiException

from ml_platform.api._configuration import Dependencies
from ml_platform.common.types import Environment


def minio_browser_url(dependencies: Dependencies, object_path: str) -> str:
    """Build a MinIO Console object-browser URL for one warehouse object prefix."""
    console_url = dependencies.minio_console_url
    prefix = quote(f"{object_path.rstrip('/')}/", safe="")
    return f"{console_url.rstrip('/')}/browser/warehouse/{prefix}"


def minio_partition_url(
    dependencies: Dependencies,
    table_path: str,
    timestamp: datetime,
) -> str:
    """Build a MinIO Console URL for one Iceberg day-partition prefix."""
    partition_day = timestamp.date().isoformat()
    return minio_browser_url(
        dependencies, f"{table_path}/data/timestamp_day={partition_day}"
    )


def headlamp_resource_url(
    dependencies: Dependencies, resource_path: str, resource_name: str
) -> str:
    """Build a direct Headlamp URL for one resource in the configured namespace."""
    headlamp_url = dependencies.headlamp_url
    namespace = dependencies.kubernetes_namespace
    return f"{headlamp_url.rstrip('/')}/c/main/{resource_path}/{namespace}/{resource_name}"


def job_kubernetes_resources(
    dependencies: Dependencies, application_name: str
) -> list[dict[str, str]]:
    """Return Headlamp links for the application and any extant Spark driver or executors."""
    application_url = headlamp_resource_url(
        dependencies,
        "customresources/sparkapplications.sparkoperator.k8s.io",
        application_name,
    )
    resources = [
        {
            "resource_type": "Spark Application",
            "resource_id": application_name,
            "url": application_url,
        }
    ]
    namespace = dependencies.kubernetes_namespace
    if dependencies.environment is Environment.TEST:
        return resources
    try:
        kubernetes_client.CustomObjectsApi().get_namespaced_custom_object(
            "sparkoperator.k8s.io",
            "v1beta2",
            namespace,
            "sparkapplications",
            application_name,
        )
    except ApiException as error:
        if error.status == 404:
            return []
        return resources
    try:
        pods = (
            kubernetes_client.CoreV1Api()
            .list_namespaced_pod(
                namespace,
                label_selector=f"sparkoperator.k8s.io/app-name={application_name}",
            )
            .items
        )
    except ApiException:
        return resources
    named_pods = [
        (pod.metadata.name, pod)
        for pod in pods
        if pod.metadata is not None and pod.metadata.name is not None
    ]
    for name, pod in sorted(named_pods):
        metadata = pod.metadata
        if metadata is None:
            continue
        role = (metadata.labels or {}).get("spark-role")
        if role not in {"driver", "executor"}:
            continue
        url = headlamp_resource_url(dependencies, "pods", name)
        resources.append(
            {
                "resource_type": f"{role.title()} Pod",
                "resource_id": name,
                "url": url,
            }
        )
    return resources
