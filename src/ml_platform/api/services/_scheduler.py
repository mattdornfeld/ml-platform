"""Kueue-aware scheduler for dataset materialization SparkApplications."""

import asyncio
import json
from typing import Any
from uuid import UUID

from kubernetes import client
from kubernetes.client.exceptions import ApiException
from kubernetes.config.incluster_config import load_incluster_config

from ml_platform.api._configuration import Dependencies
from ml_platform.api._exceptions import UnsupportedScheduledJobTypeError
from ml_platform.api.repositories import SchedulerRepository
from ml_platform.common.types import JobId

ADMISSION_GATE = "api.ml-control-plane.io/blocked"
QUEUE_LABEL = "kueue.x-k8s.io/queue-name"
QUEUE_NAME = "materialization"


class JobScheduler:
    """Create gated SparkApplications, release eligible jobs, and observe outcomes."""

    def __init__(self, dependencies: Dependencies) -> None:
        """Initialize the scheduler with all typed API runtime dependencies."""
        self.dependencies = dependencies
        self._repository = SchedulerRepository(dependencies.database_url)

    async def run(self, stop: asyncio.Event) -> None:
        """Continuously reconcile submitted jobs without duplicating Kueue states."""
        load_incluster_config()
        while not stop.is_set():
            await asyncio.to_thread(self._reconcile_submitted_jobs)
            await asyncio.sleep(1)

    def _reconcile_submitted_jobs(self) -> None:
        """Create, release, and observe every nonterminal materialization job in FIFO order."""
        for job in self._repository.submitted_jobs():
            job_id = JobId(UUID(str(job["id"])))
            try:
                job_type = str(job["type"])
                application_name = self._application_name(job_id, job_type)
                if job_type == "dataset_materialization":
                    self._ensure_dataset_application(job_id, application_name)
                elif job_type == "feature_materialization":
                    self._ensure_feature_application(job_id, application_name)
                elif job_type == "label_materialization":
                    self._ensure_label_application(job_id, application_name)
                else:
                    raise UnsupportedScheduledJobTypeError(job_type)
                if self._is_eligible(job_id, job_type):
                    self._remove_admission_gate(application_name)
                self._observe_application(job_id, application_name, job_type)
            except ApiException:
                # Kubernetes may briefly be unavailable; retain submitted status for retry.
                continue

    def _dataset_job(self, job_id: JobId) -> dict[str, Any]:
        """Load one dataset job's materialization configuration."""
        return self._repository.dataset_job(job_id)

    def _feature_job(self, job_id: JobId) -> dict[str, Any]:
        """Load one feature job's materialization configuration."""
        return self._repository.feature_job(job_id)

    def _label_job(self, job_id: JobId) -> dict[str, Any]:
        """Load one label job's materialization configuration."""
        return self._repository.label_job(job_id)

    def _is_eligible(self, job_id: JobId, job_type: str) -> bool:
        """Determine whether the job owns all requested partition locks."""
        return self._repository.is_eligible(job_id, job_type)

    def _finish(self, job_id: JobId, outcome: str, job_type: str) -> None:
        """Persist a terminal Spark application result."""
        self._repository.finish(job_id, outcome, job_type)

    @staticmethod
    def _application_name(job_id: JobId, job_type: str) -> str:
        """Return the deterministic SparkApplication name for one API job."""
        prefix = job_type.removesuffix("_materialization")
        return f"{prefix}-{job_id}".lower()

    def _ensure_dataset_application(
        self, job_id: JobId, application_name: str
    ) -> None:
        """Create a Kueue-gated SparkApplication, preserving a pre-existing one."""
        dataset = self._dataset_job(job_id)
        namespace = self.dependencies.kubernetes_namespace
        application = {
            "apiVersion": "sparkoperator.k8s.io/v1beta2",
            "kind": "SparkApplication",
            "metadata": {
                "name": application_name,
                "namespace": namespace,
                "labels": {
                    QUEUE_LABEL: QUEUE_NAME,
                    "api.ml-control-plane.io/job-id": str(job_id),
                },
                "annotations": {
                    "kueue.x-k8s.io/admission-gated-by": ADMISSION_GATE
                },
            },
            "spec": {
                "type": "Python",
                "mode": "cluster",
                "suspend": True,
                "image": self.dependencies.materialization_image,
                "imagePullPolicy": "IfNotPresent",
                "mainApplicationFile": "local:///app/src/ml_platform/data_sdk/materializers/_dataset.py",
                "arguments": [
                    "--config-json",
                    json.dumps(
                        {
                            key: dataset["config"][key]
                            for key in (
                                "description",
                                "validations",
                                "keys",
                                "partition_type",
                                "symbols",
                            )
                        }
                        | {
                            "name": dataset["name"],
                            "version": dataset["version"],
                            "dataset_type": dataset["dataset_type"],
                            "environment": self.dependencies.environment,
                            "warehouse_dir": self.dependencies.warehouse_dir,
                            "partitions": dataset["callbacks"],
                        },
                        default=str,
                    ),
                ],
                "sparkVersion": "4.1.3",
                "restartPolicy": {"type": "Never"},
                "sparkConf": {
                    "spark.sql.session.timeZone": "UTC",
                    "spark.hadoop.fs.s3a.endpoint": self.dependencies.s3_endpoint,
                    "spark.hadoop.fs.s3a.path.style.access": "true",
                    "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
                    "spark.hadoop.fs.s3a.access.key": self.dependencies.minio_root_user,
                    "spark.hadoop.fs.s3a.secret.key": self.dependencies.minio_root_password,
                },
                "driver": {
                    "cores": 1,
                    "memory": "2g",
                    "serviceAccount": "ml-control-plane",
                    "env": [
                        {
                            "name": "MATERIALIZATION_API_URL",
                            "value": "http://ml-control-plane:8000",
                        },
                    ],
                },
                "executor": {"instances": 2, "cores": 1, "memory": "2g"},
            },
        }
        api = client.CustomObjectsApi()
        try:
            api.create_namespaced_custom_object(
                "sparkoperator.k8s.io",
                "v1beta2",
                namespace,
                "sparkapplications",
                application,
            )
        except ApiException as error:
            if error.status != 409:
                raise

    def _ensure_feature_application(
        self, job_id: JobId, application_name: str
    ) -> None:
        """Create one Kueue-gated SparkApplication for a feature job."""
        feature = self._feature_job(job_id)
        namespace = self.dependencies.kubernetes_namespace
        application = {
            "apiVersion": "sparkoperator.k8s.io/v1beta2",
            "kind": "SparkApplication",
            "metadata": {
                "name": application_name,
                "namespace": namespace,
                "labels": {
                    QUEUE_LABEL: QUEUE_NAME,
                    "api.ml-control-plane.io/job-id": str(job_id),
                },
                "annotations": {
                    "kueue.x-k8s.io/admission-gated-by": ADMISSION_GATE
                },
            },
            "spec": {
                "type": "Python",
                "mode": "cluster",
                "suspend": True,
                "image": self.dependencies.materialization_image,
                "imagePullPolicy": "IfNotPresent",
                "mainApplicationFile": "local:///app/src/ml_platform/data_sdk/materializers/_feature.py",
                "arguments": [
                    "--config-json",
                    json.dumps(
                        {
                            **feature["config"],
                            "environment": self.dependencies.environment,
                            "warehouse_dir": self.dependencies.warehouse_dir,
                            "partitions": feature["callbacks"],
                        },
                        default=str,
                    ),
                ],
                "sparkVersion": "4.1.3",
                "restartPolicy": {"type": "Never"},
                "sparkConf": {
                    "spark.sql.session.timeZone": "UTC",
                    "spark.hadoop.fs.s3a.endpoint": self.dependencies.s3_endpoint,
                    "spark.hadoop.fs.s3a.path.style.access": "true",
                    "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
                    "spark.hadoop.fs.s3a.access.key": self.dependencies.minio_root_user,
                    "spark.hadoop.fs.s3a.secret.key": self.dependencies.minio_root_password,
                },
                "driver": {
                    "cores": 1,
                    "memory": "2g",
                    "serviceAccount": "ml-control-plane",
                    "env": [
                        {
                            "name": "MATERIALIZATION_API_URL",
                            "value": "http://ml-control-plane:8000",
                        }
                    ],
                },
                "executor": {"instances": 2, "cores": 1, "memory": "2g"},
            },
        }
        try:
            client.CustomObjectsApi().create_namespaced_custom_object(
                "sparkoperator.k8s.io",
                "v1beta2",
                namespace,
                "sparkapplications",
                application,
            )
        except ApiException as error:
            if error.status != 409:
                raise

    def _ensure_label_application(
        self, job_id: JobId, application_name: str
    ) -> None:
        """Create one Kueue-gated SparkApplication for a label job."""
        label = self._label_job(job_id)
        namespace = self.dependencies.kubernetes_namespace
        application = {
            "apiVersion": "sparkoperator.k8s.io/v1beta2",
            "kind": "SparkApplication",
            "metadata": {
                "name": application_name,
                "namespace": namespace,
                "labels": {
                    QUEUE_LABEL: QUEUE_NAME,
                    "api.ml-control-plane.io/job-id": str(job_id),
                },
                "annotations": {
                    "kueue.x-k8s.io/admission-gated-by": ADMISSION_GATE
                },
            },
            "spec": {
                "type": "Python",
                "mode": "cluster",
                "suspend": True,
                "image": self.dependencies.materialization_image,
                "imagePullPolicy": "IfNotPresent",
                "mainApplicationFile": "local:///app/src/ml_platform/data_sdk/materializers/_label.py",
                "arguments": [
                    "--config-json",
                    json.dumps(
                        {
                            **label["config"],
                            "environment": self.dependencies.environment,
                            "warehouse_dir": self.dependencies.warehouse_dir,
                            "partitions": label["callbacks"],
                        },
                        default=str,
                    ),
                ],
                "sparkVersion": "4.1.3",
                "restartPolicy": {"type": "Never"},
                "sparkConf": {
                    "spark.sql.session.timeZone": "UTC",
                    "spark.hadoop.fs.s3a.endpoint": self.dependencies.s3_endpoint,
                    "spark.hadoop.fs.s3a.path.style.access": "true",
                    "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
                    "spark.hadoop.fs.s3a.access.key": self.dependencies.minio_root_user,
                    "spark.hadoop.fs.s3a.secret.key": self.dependencies.minio_root_password,
                },
                "driver": {
                    "cores": 1,
                    "memory": "2g",
                    "serviceAccount": "ml-control-plane",
                    "env": [
                        {
                            "name": "MATERIALIZATION_API_URL",
                            "value": "http://ml-control-plane:8000",
                        }
                    ],
                },
                "executor": {"instances": 2, "cores": 1, "memory": "2g"},
            },
        }
        try:
            client.CustomObjectsApi().create_namespaced_custom_object(
                "sparkoperator.k8s.io",
                "v1beta2",
                namespace,
                "sparkapplications",
                application,
            )
        except ApiException as error:
            if error.status != 409:
                raise

    def _remove_admission_gate(self, application_name: str) -> None:
        """Release an API-eligible SparkApplication into Kueue resource admission."""
        namespace = self.dependencies.kubernetes_namespace
        client.CustomObjectsApi().patch_namespaced_custom_object(
            "sparkoperator.k8s.io",
            "v1beta2",
            namespace,
            "sparkapplications",
            application_name,
            {
                "metadata": {
                    "annotations": {"kueue.x-k8s.io/admission-gated-by": None}
                }
            },
        )

    def _observe_application(
        self, job_id: JobId, application_name: str, job_type: str
    ) -> None:
        """Map terminal Spark states to the API's durable terminal status."""
        namespace = self.dependencies.kubernetes_namespace
        try:
            application = (
                client.CustomObjectsApi().get_namespaced_custom_object(
                    "sparkoperator.k8s.io",
                    "v1beta2",
                    namespace,
                    "sparkapplications",
                    application_name,
                )
            )
        except ApiException as error:
            if error.status == 404:
                return
            raise
        if not isinstance(application, dict):
            return
        status = application.get("status")
        application_state = (
            status.get("applicationState") if isinstance(status, dict) else None
        )
        state = (
            application_state.get("state")
            if isinstance(application_state, dict)
            else None
        )
        if state in {"FAILED", "SUBMISSION_FAILED", "UNKNOWN"}:
            self._finish(job_id, "failed", job_type)
        elif state == "COMPLETED":
            self._finish(job_id, "succeeded", job_type)
        elif state == "CANCELLED":
            self._finish(job_id, "cancelled", job_type)
