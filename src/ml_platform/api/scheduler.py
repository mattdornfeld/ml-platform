"""Kueue-aware scheduler for dataset materialization SparkApplications."""

import asyncio
import json
import os
from typing import Any
from uuid import UUID

import psycopg
from kubernetes import client
from kubernetes.client.exceptions import ApiException
from kubernetes.config.incluster_config import load_incluster_config
from psycopg.rows import dict_row

ADMISSION_GATE = "api.ml-control-plane.io/blocked"
QUEUE_LABEL = "kueue.x-k8s.io/queue-name"
QUEUE_NAME = "materialization"


class JobScheduler:
    """Create gated SparkApplications, release eligible jobs, and observe outcomes."""

    def __init__(self, database_url: str, max_concurrency: int = 2) -> None:
        """Initialize a scheduler with its database connection and concurrency cap."""
        self.database_url = database_url
        self.max_concurrency = max_concurrency

    async def run(self, stop: asyncio.Event) -> None:
        """Continuously reconcile submitted jobs without duplicating Kueue states."""
        load_incluster_config()
        while not stop.is_set():
            await asyncio.to_thread(self._reconcile_submitted_jobs)
            await asyncio.sleep(1)

    def _reconcile_submitted_jobs(self) -> None:
        """Create, release, and observe every nonterminal materialization job in FIFO order."""
        for job in self._submitted_jobs():
            job_id = UUID(str(job["id"]))
            try:
                job_type = job["type"]
                application_name = self._application_name(job_id, job_type)
                if job_type == "dataset_materialization":
                    self._ensure_dataset_application(job_id, application_name)
                else:
                    self._ensure_feature_application(job_id, application_name)
                if self._is_eligible(job_id, job_type):
                    self._remove_admission_gate(application_name)
                self._observe_application(job_id, application_name, job_type)
            except ApiException:
                # Kubernetes may briefly be unavailable; retain submitted status for retry.
                continue

    def _submitted_jobs(self) -> list[dict[str, Any]]:
        """Return nonterminal jobs in deterministic API submission order."""
        with (
            psycopg.connect(
                self.database_url, row_factory=dict_row
            ) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(
                """SELECT jobs.id, jobs.type, jobs.created_at FROM jobs
                WHERE jobs.type IN ('dataset_materialization', 'feature_materialization')
                  AND jobs.status = 'submitted'
                  AND NOT EXISTS (
                    SELECT 1 FROM partition_update_requests AS request
                    JOIN datasets ON datasets.name = request.dataset_name AND datasets.version = request.dataset_version
                    WHERE request.job_id = jobs.id AND datasets.deleted_at IS NOT NULL
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM feature_partition_update_requests AS request
                    JOIN features ON features.name = request.feature_name AND features.version = request.feature_version
                    WHERE request.job_id = jobs.id AND features.deleted_at IS NOT NULL
                  )
                ORDER BY jobs.created_at, jobs.id"""
            )
            return list(cursor.fetchall())

    @staticmethod
    def _application_name(job_id: UUID, job_type: str) -> str:
        """Return the deterministic SparkApplication name for one API job."""
        prefix = (
            "dataset" if job_type == "dataset_materialization" else "feature"
        )
        return f"{prefix}-{job_id}".lower()

    def _ensure_dataset_application(
        self, job_id: UUID, application_name: str
    ) -> None:
        """Create a Kueue-gated SparkApplication, preserving a pre-existing one."""
        dataset = self._dataset_job(job_id)
        namespace = os.environ.get("KUBERNETES_NAMESPACE", "ml-platform")
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
                "image": os.environ.get(
                    "MATERIALIZATION_IMAGE", "ml-platform:latest"
                ),
                "imagePullPolicy": "IfNotPresent",
                "mainApplicationFile": "local:///app/src/ml_platform/data_sdk/materializers/_dataset.py",
                "arguments": [
                    "--config-json",
                    json.dumps(
                        {
                            **dataset["config"],
                            "name": dataset["name"],
                            "version": dataset["version"],
                            "dataset_type": dataset["dataset_type"],
                            "warehouse_dir": os.environ.get(
                                "DATA_WAREHOUSE_DIR", "s3a://warehouse"
                            ),
                            "partitions": dataset["callbacks"],
                        },
                        default=str,
                    ),
                ],
                "sparkVersion": "4.1.3",
                "restartPolicy": {"type": "Never"},
                "sparkConf": {
                    "spark.sql.session.timeZone": "UTC",
                    "spark.hadoop.fs.s3a.endpoint": os.environ.get(
                        "S3_ENDPOINT", "http://minio:9000"
                    ),
                    "spark.hadoop.fs.s3a.path.style.access": "true",
                    "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
                    "spark.hadoop.fs.s3a.access.key": os.environ.get(
                        "MINIO_ROOT_USER", "minio"
                    ),
                    "spark.hadoop.fs.s3a.secret.key": os.environ.get(
                        "MINIO_ROOT_PASSWORD", "miniosecret"
                    ),
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

    def _dataset_job(self, job_id: UUID) -> dict[str, Any]:
        """Load the immutable dataset config and authenticated callbacks for one job."""
        with (
            psycopg.connect(
                self.database_url, row_factory=dict_row
            ) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(
                """SELECT datasets.name, datasets.version, datasets.dataset_type, datasets.config,
                json_agg(json_build_object('partition_id', request.partition_id,
                    'timestamp', request.timestamp, 'callback_token', request.callback_token)
                    ORDER BY request.timestamp) AS callbacks
                FROM partition_update_requests AS request
                JOIN datasets ON datasets.name = request.dataset_name
                  AND datasets.version = request.dataset_version
                WHERE request.job_id = %s
                GROUP BY datasets.name, datasets.version, datasets.dataset_type, datasets.config""",
                (job_id,),
            )
            dataset = cursor.fetchone()
        if dataset is None:
            raise ValueError(
                f"No dataset partitions are associated with job {job_id}."
            )
        return dataset

    def _ensure_feature_application(
        self, job_id: UUID, application_name: str
    ) -> None:
        """Create one Kueue-gated SparkApplication for a feature job."""
        feature = self._feature_job(job_id)
        namespace = os.environ.get("KUBERNETES_NAMESPACE", "ml-platform")
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
                "image": os.environ.get(
                    "MATERIALIZATION_IMAGE", "ml-platform:latest"
                ),
                "imagePullPolicy": "IfNotPresent",
                "mainApplicationFile": "local:///app/src/ml_platform/data_sdk/materializers/_feature.py",
                "arguments": [
                    "--config-json",
                    json.dumps(
                        {
                            **feature["config"],
                            "warehouse_dir": os.environ.get(
                                "DATA_WAREHOUSE_DIR", "s3a://warehouse"
                            ),
                            "partitions": feature["callbacks"],
                        },
                        default=str,
                    ),
                ],
                "sparkVersion": "4.1.3",
                "restartPolicy": {"type": "Never"},
                "sparkConf": {
                    "spark.sql.session.timeZone": "UTC",
                    "spark.hadoop.fs.s3a.endpoint": os.environ.get(
                        "S3_ENDPOINT", "http://minio:9000"
                    ),
                    "spark.hadoop.fs.s3a.path.style.access": "true",
                    "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
                    "spark.hadoop.fs.s3a.access.key": os.environ.get(
                        "MINIO_ROOT_USER", "minio"
                    ),
                    "spark.hadoop.fs.s3a.secret.key": os.environ.get(
                        "MINIO_ROOT_PASSWORD", "miniosecret"
                    ),
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

    def _feature_job(self, job_id: UUID) -> dict[str, Any]:
        """Load immutable feature metadata, dependencies, and authenticated callbacks."""
        with (
            psycopg.connect(
                self.database_url, row_factory=dict_row
            ) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(
                """SELECT features.name, features.version, features.dtype, features.frequency, features.feature_keys, features.validations, features.transform, features.description, features.parameters,
                json_agg(DISTINCT jsonb_build_object('name', dependencies.dataset_name, 'version', dependencies.dataset_version, 'dataset_type', input_datasets.dataset_type, 'argument_name', dependencies.argument_name, 'num_lookback_partitions', dependencies.num_lookback_partitions)) AS datasets,
                (SELECT json_agg(json_build_object(
                    'partition_id', request.partition_id,
                    'timestamp', request.timestamp,
                    'callback_token', request.callback_token,
                    'partition_dependencies', COALESCE((
                        SELECT json_agg(json_build_object(
                            'partition_id', planned.dataset_partition_id,
                            'dataset_name', parent_request.dataset_name,
                            'dataset_version', parent_request.dataset_version,
                            'timestamp', parent_request.timestamp,
                            'snapshot_id', parent.snapshot_id
                        ) ORDER BY parent_request.timestamp, planned.dataset_partition_id)
                        FROM feature_partition_update_request_dataset_dependencies planned
                        JOIN dataset_partitions parent ON parent.partition_id = planned.dataset_partition_id
                        JOIN partition_update_requests parent_request ON parent_request.partition_id = parent.partition_id
                        WHERE planned.feature_partition_request_id = request.partition_id
                    ), '[]'::json)
                ) ORDER BY request.timestamp) FROM feature_partition_update_requests request WHERE request.job_id = %s) AS callbacks
                FROM feature_partition_update_requests target JOIN features ON features.name = target.feature_name AND features.version = target.feature_version
                JOIN feature_dataset_dependencies dependencies ON dependencies.feature_name = features.name AND dependencies.feature_version = features.version
                JOIN datasets AS input_datasets ON input_datasets.name = dependencies.dataset_name
                  AND input_datasets.version = dependencies.dataset_version
                WHERE target.job_id = %s GROUP BY features.name, features.version, features.dtype, features.frequency, features.feature_keys, features.validations, features.transform, features.description, features.parameters""",
                (job_id, job_id),
            )
            row = cursor.fetchone()
        if row is None:
            raise ValueError(
                f"No feature partitions are associated with job {job_id}."
            )
        config = {
            key: row[key]
            for key in (
                "name",
                "version",
                "dtype",
                "frequency",
                "feature_keys",
                "validations",
                "transform",
                "description",
                "parameters",
                "datasets",
            )
        }
        return {"config": config, "callbacks": row["callbacks"]}

    def _is_eligible(self, job_id: UUID, job_type: str) -> bool:
        """Allow only the oldest submitted job for every partition it targets."""
        with (
            psycopg.connect(
                self.database_url, row_factory=dict_row
            ) as connection,
            connection.cursor() as cursor,
        ):
            table = (
                "partition_update_requests"
                if job_type == "dataset_materialization"
                else "feature_partition_update_requests"
            )
            name = (
                "dataset_name"
                if job_type == "dataset_materialization"
                else "feature_name"
            )
            version = (
                "dataset_version"
                if job_type == "dataset_materialization"
                else "feature_version"
            )
            cursor.execute(
                f"SELECT {name}, {version}, timestamp FROM {table} WHERE job_id = %s ORDER BY {name}, {version}, timestamp",
                (job_id,),
            )
            targets = cursor.fetchall()
            for target in targets:
                key = f"{job_type}:{target[name]}:{target[version]}:{target['timestamp'].isoformat()}"
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (key,),
                )
            cursor.execute(
                f"""SELECT EXISTS (
                    SELECT 1 FROM {table} AS current_request
                    JOIN {table} AS older_request
                      ON older_request.{name} = current_request.{name}
                     AND older_request.{version} = current_request.{version}
                     AND older_request.timestamp = current_request.timestamp
                    JOIN jobs AS current_job ON current_job.id = current_request.job_id
                    JOIN jobs AS older_job ON older_job.id = older_request.job_id
                    WHERE current_request.job_id = %s
                      AND older_job.status = 'submitted'
                      AND (older_job.created_at, older_job.id) < (current_job.created_at, current_job.id)
                ) AS has_older_conflict""",
                (job_id,),
            )
            result = cursor.fetchone()
            if result is None:
                raise RuntimeError(
                    "Partition eligibility query did not return a row."
                )
            return not result["has_older_conflict"]

    def _remove_admission_gate(self, application_name: str) -> None:
        """Release an API-eligible SparkApplication into Kueue resource admission."""
        namespace = os.environ.get("KUBERNETES_NAMESPACE", "ml-platform")
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
        self, job_id: UUID, application_name: str, job_type: str
    ) -> None:
        """Map terminal Spark states to the API's durable terminal status."""
        namespace = os.environ.get("KUBERNETES_NAMESPACE", "ml-platform")
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

    def _finish(self, job_id: UUID, outcome: str, job_type: str) -> None:
        """Persist a terminal outcome after confirming all successful receipts exist."""
        with (
            psycopg.connect(self.database_url) as connection,
            connection.cursor() as cursor,
        ):
            table = (
                "partition_update_requests"
                if job_type == "dataset_materialization"
                else "feature_partition_update_requests"
            )
            if outcome == "succeeded":
                cursor.execute(
                    f"SELECT EXISTS (SELECT 1 FROM {table} WHERE job_id = %s AND status <> 'created')",
                    (job_id,),
                )
                result = cursor.fetchone()
                if result is not None and result[0]:
                    outcome = "failed"
            if outcome == "failed":
                cursor.execute(
                    f"UPDATE {table} SET status = 'failed', updated_at = now() WHERE job_id = %s AND status = 'pending'",
                    (job_id,),
                )
            cursor.execute(
                "UPDATE jobs SET status = %s, updated_at = now() WHERE id = %s AND status = 'submitted'",
                (outcome, job_id),
            )
            connection.commit()
