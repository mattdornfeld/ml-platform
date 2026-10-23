"""External read-only dashboard links for control-plane resources."""

import os
from datetime import datetime
from typing import Any
from urllib.parse import quote

from kubernetes import client as kubernetes_client
from kubernetes.client.exceptions import ApiException


def minio_browser_url(object_path: str) -> str | None:
    """Build a MinIO Console object-browser URL for one warehouse object prefix."""
    console_url = os.environ.get("MINIO_CONSOLE_URL")
    if not console_url:
        return None
    prefix = quote(f"{object_path.rstrip('/')}/", safe="")
    return f"{console_url.rstrip('/')}/browser/warehouse/{prefix}"


def attach_minio_locations(resource: dict[str, Any], table_path: str) -> None:
    """Attach direct MinIO Console locations for an Iceberg table and its day partitions."""
    resource["minio_table_url"] = minio_browser_url(table_path)
    for partition in resource.get("partitions", []):
        timestamp = partition["timestamp"]
        partition_day = (
            timestamp.date().isoformat()
            if isinstance(timestamp, datetime)
            else str(timestamp)[:10]
        )
        partition["minio_url"] = minio_browser_url(
            f"{table_path}/data/timestamp_day={partition_day}"
        )


def headlamp_resource_url(resource_path: str, resource_name: str) -> str | None:
    """Build a direct Headlamp URL for one resource in the configured namespace."""
    headlamp_url = os.environ.get("HEADLAMP_URL")
    if not headlamp_url:
        return None
    namespace = os.environ.get("KUBERNETES_NAMESPACE", "ml-platform")
    return f"{headlamp_url.rstrip('/')}/c/main/{resource_path}/{namespace}/{resource_name}"


def job_kubernetes_resources(application_name: str) -> list[dict[str, str]]:
    """Return Headlamp links for the application and any extant Spark driver or executors."""
    application_url = headlamp_resource_url(
        "customresources/sparkapplications.sparkoperator.k8s.io",
        application_name,
    )
    resources = (
        [
            {
                "resource_type": "Spark Application",
                "resource_id": application_name,
                "url": application_url,
            }
        ]
        if application_url
        else []
    )
    if not application_url:
        return resources
    namespace = os.environ.get("KUBERNETES_NAMESPACE", "ml-platform")
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
        url = headlamp_resource_url("pods", name)
        if url:
            resources.append(
                {
                    "resource_type": f"{role.title()} Pod",
                    "resource_id": name,
                    "url": url,
                }
            )
    return resources
