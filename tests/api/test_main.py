"""Validation tests for materialization API routes."""

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID

from fastapi.testclient import TestClient
from pydantic import ValidationError

from ml_platform.api import app
from ml_platform.api.domain import (
    DatasetDefinition,
    DatasetSummary,
    JobPartition,
    JobStatus,
    JobTarget,
    JobType,
    MaterializationJob,
    MaterializationPartition,
    MaterializationRequest,
    PartitionCounts,
    PartitionStatus,
    ResourceType,
)
from ml_platform.api.infrastructure import get_session
from ml_platform.api.routers import create_dataset, get_job, list_datasets
from ml_platform.api.services import minio_browser_url, minio_partition_url
from ml_platform.common.api.dtos import (
    DatasetCreateRequest,
    DatasetStatus,
    FeatureCreateRequest,
)
from ml_platform.common.domain import MarketDataDatasetConfig
from ml_platform.common.types import (
    CreatedAt,
    EntityKey,
    JobId,
    PartitionTimestamp,
    PartitionType,
    ResourceName,
    ResourceVersion,
    Symbol,
    UpdatedAt,
)


class PartitionMaterializationRequestTest(unittest.TestCase):
    """Verify API validation for partition request timestamps."""

    def setUp(self) -> None:
        """Create an isolated in-process API client."""
        app.dependency_overrides[get_session] = lambda: MagicMock()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        """Discard dependency overrides after each API test."""
        app.dependency_overrides.clear()

    def test_create_dataset_registers_without_materialization_job(self) -> None:
        """Dataset registration persists only the immutable resource definition."""
        now = datetime.now(UTC)
        models: list[object] = []
        session = MagicMock()
        session.add.side_effect = models.append

        def refresh(model: object) -> None:
            """Apply the dataset creation timestamp supplied by PostgreSQL."""
            model.created_at = now

        session.refresh.side_effect = refresh
        payload = DatasetCreateRequest.model_validate(
            {
                "name": "sp500",
                "version": 1,
                "dataset_type": "market_data",
                "config": {
                    "keys": ["symbol"],
                    "partition_type": "daily",
                    "symbols": ["AAPL"],
                },
            }
        )

        response = create_dataset(payload, session)

        self.assertEqual(response.name, "sp500")
        self.assertEqual(response.id, "sp500_v1")
        self.assertEqual(models, [models[0]])
        self.assertEqual(type(models[0]).__name__, "Dataset")
        session.commit.assert_called_once()

    def test_dataset_validation_policy_includes_empty_partition_setting(
        self,
    ) -> None:
        """Dataset registration exposes empty-partition handling as a typed validation policy."""
        request = DatasetCreateRequest.model_validate(
            {
                "name": "sp500",
                "version": 1,
                "dataset_type": "market_data",
                "config": {
                    "keys": ["symbol"],
                    "partition_type": "daily",
                    "symbols": ["AAPL"],
                    "validations": [
                        {
                            "type": "minimum_rows",
                            "minimum_rows": 1,
                            "reason": "Every selected market-data partition requires a row.",
                        }
                    ],
                },
            }
        )

        self.assertEqual(request.config.validations[0].minimum_rows, 1)
        self.assertEqual(
            request.config.model_dump(mode="json")["validations"],
            [
                {
                    "type": "minimum_rows",
                    "reason": "Every selected market-data partition requires a row.",
                    "temporal_filter": {"type": "all_partitions"},
                    "minimum_rows": 1,
                }
            ],
        )

    def test_dataset_request_normalizes_symbols_and_rejects_ambiguity(
        self,
    ) -> None:
        """Dataset request validation rejects ambiguous names, keys, and symbols."""
        request = DatasetCreateRequest.model_validate(
            {
                "name": "sp500",
                "version": 1,
                "dataset_type": "market_data",
                "config": {
                    "keys": ["symbol"],
                    "partition_type": "daily",
                    "symbols": [" aapl "],
                },
            }
        )
        self.assertEqual(request.config.symbols, ["AAPL"])

        with self.assertRaises(ValidationError):
            DatasetCreateRequest.model_validate(
                request.model_dump(mode="json") | {"name": "S&P500"}
            )
        with self.assertRaises(ValidationError):
            DatasetCreateRequest.model_validate(
                request.model_dump(mode="json")
                | {
                    "config": request.config.model_dump(mode="json")
                    | {"keys": ["symbol", "symbol"]}
                }
            )
        with self.assertRaises(ValidationError):
            DatasetCreateRequest.model_validate(
                request.model_dump(mode="json")
                | {
                    "config": request.config.model_dump(mode="json")
                    | {"symbols": ["AAPL", "aapl"]}
                }
            )

    def test_transform_request_rejects_ambiguous_validation_overrides(
        self,
    ) -> None:
        """Feature requests reject duplicate validation policies before storage."""
        with self.assertRaises(ValidationError):
            FeatureCreateRequest.model_validate(
                {
                    "name": "daily_return",
                    "version": 1,
                    "dtype": "double",
                    "frequency": "daily",
                    "datasets": [
                        {
                            "name": "prices",
                            "version": 1,
                            "input_window": {"lookback_partitions": 0},
                        }
                    ],
                    "transform": "ml_platform.transforms.DailyReturn",
                    "description": "Daily return.",
                    "parameters": {},
                    "validations": [
                        {
                            "type": "minimum_rows",
                            "minimum_rows": 1,
                            "reason": "Require an output row.",
                        },
                        {
                            "type": "minimum_rows",
                            "minimum_rows": 2,
                            "reason": "Duplicate configuration.",
                        },
                    ],
                }
            )

    def test_completion_endpoint_rejects_duplicate_receipt_identifiers(
        self,
    ) -> None:
        """Duplicate receipt IDs fail request validation before persistence."""
        receipt = {
            "partition_id": "00000000-0000-0000-0000-000000000001",
            "resource_type": "dataset",
            "row_count": 1,
            "iceberg_schema": {"timestamp": "timestamp"},
            "content_hash": "hash",
            "snapshot_id": 1,
            "callback_token": "token",
        }
        response = self.client.post(
            "/partitions/complete", json=[receipt, receipt]
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("unique partition IDs", response.text)

    def test_dataset_partition_request_rejects_non_hour_timestamp(self) -> None:
        """Partition endpoints reject timestamps that are not aligned to the hour."""
        response = self.client.post(
            "/datasets/market_data/versions/1/partitions",
            json={"timestamps": ["2026-10-20T18:15:47.828Z"]},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("aligned to an exact hour", response.text)

    def test_completion_endpoint_requires_top_level_receipt_array(self) -> None:
        """OpenAPI and request validation expose only the batch callback route."""
        self.assertEqual(
            self.client.post("/partitions/complete", json=[]).status_code, 422
        )
        spec = self.client.get("/openapi.json").json()
        self.assertIn("/partitions/complete", spec["paths"])
        schema = spec["paths"]["/partitions/complete"]["post"]["requestBody"][
            "content"
        ]["application/json"]["schema"]
        self.assertEqual(
            spec["components"]["schemas"][
                schema["$ref"].rsplit("/", maxsplit=1)[1]
            ]["type"],
            "array",
        )
        self.assertNotIn(
            "/dataset-partition-update-requests/{partition_id}/complete",
            spec["paths"],
        )
        self.assertNotIn(
            "/feature-partition-update-requests/{partition_id}/complete",
            spec["paths"],
        )

    def test_feature_ensure_current_returns_completed_partition_without_a_new_job(
        self,
    ) -> None:
        """The route adapts the repository's feature domain result."""
        now = datetime.now(UTC)
        completed = MaterializationRequest(
            submitted_job=None,
            partitions=(
                MaterializationPartition(
                    id=UUID("00000000-0000-0000-0000-000000000002"),
                    job_id=UUID("00000000-0000-0000-0000-000000000001"),
                    timestamp=datetime(2026, 10, 20, 18, tzinfo=UTC),
                    status=PartitionStatus.CREATED,
                    created_at=now,
                    updated_at=now,
                ),
            ),
        )
        with patch(
            "ml_platform.api.routers._features.FeatureRepository"
        ) as repository_type:
            repository_type.return_value.materialize.return_value = completed
            response = self.client.post(
                "/features/trailing_volatility_1d/versions/1/partitions",
                json={"timestamps": ["2026-10-20T18:00:00Z"]},
            )
        self.assertEqual(response.status_code, 202)
        self.assertIsNone(response.json()["submitted_job"])
        self.assertEqual(
            response.json()["partitions"][0]["partition_id"],
            "00000000-0000-0000-0000-000000000002",
        )

    def test_get_job_returns_persisted_job(self) -> None:
        """The job endpoint renders repository-owned job domain models."""
        now = datetime.now(UTC)
        job_id = JobId(UUID("00000000-0000-0000-0000-000000000001"))
        job = MaterializationJob(
            id=job_id,
            type=JobType.DATASET_MATERIALIZATION,
            status=JobStatus.SUBMITTED,
            created_at=CreatedAt(now),
            updated_at=UpdatedAt(now),
        )
        target = JobTarget(
            resource_type=ResourceType.DATASET,
            name=ResourceName("sp500"),
            version=ResourceVersion(3),
        )
        partition = JobPartition(
            resource_type=ResourceType.DATASET,
            name=ResourceName("sp500"),
            version=ResourceVersion(3),
            timestamp=PartitionTimestamp(
                datetime(2026, 10, 20, 18, tzinfo=UTC)
            ),
            status=PartitionStatus.CREATED,
        )
        session = MagicMock()

        with (
            patch.dict(
                "os.environ",
                {"HEADLAMP_URL": "http://headlamp.test"},
                clear=False,
            ),
            patch(
                "ml_platform.api.routers._jobs.JobRepository"
            ) as repository_type,
            patch(
                "ml_platform.api.services._resources.kubernetes_client.CoreV1Api"
            ) as core_api,
            patch(
                "ml_platform.api.services._resources.kubernetes_client.CustomObjectsApi"
            ),
        ):
            repository = repository_type.return_value
            repository.get.return_value = job
            repository.targets.return_value = (target,)
            repository.partitions.return_value = (partition,)
            core_api.return_value.list_namespaced_pod.return_value.items = [
                SimpleNamespace(
                    metadata=SimpleNamespace(
                        name="dataset-job-driver",
                        labels={"spark-role": "driver"},
                    )
                ),
                SimpleNamespace(
                    metadata=SimpleNamespace(
                        name="dataset-job-exec-1",
                        labels={"spark-role": "executor"},
                    )
                ),
            ]
            response = get_job(job_id, session)

        self.assertEqual(str(response.id), str(job_id))
        self.assertEqual(response.status, "submitted")
        self.assertEqual(response.targets[0].name, "sp500")
        self.assertEqual(response.targets[0].version, 3)
        self.assertEqual(response.partitions[0].status, "created")
        self.assertEqual(response.spark_application_name, f"dataset-{job_id}")
        self.assertEqual(
            response.kubernetes_resources[0].resource_type, "Spark Application"
        )
        self.assertEqual(
            response.kubernetes_resources[0].resource_id,
            response.spark_application_name,
        )
        self.assertEqual(
            response.kubernetes_resources[1].resource_type, "Driver Pod"
        )
        self.assertEqual(
            response.kubernetes_resources[1].resource_id, "dataset-job-driver"
        )


class MaterializationBehaviorTest(unittest.TestCase):
    """Exercise idempotency and completion safety at the API boundary."""

    def _connection(self, rows: list[object]) -> tuple[MagicMock, MagicMock]:
        cursor = MagicMock()
        cursor.fetchone.side_effect = rows
        cursor.__enter__.return_value = cursor
        connection = MagicMock()
        connection.cursor.return_value = cursor
        return connection, cursor

    def test_dataset_request_returns_existing_completed_partition(self) -> None:
        """A create-missing dataset request does not enqueue another writer."""
        from ml_platform.api.routers import (
            materialize_dataset_partitions,
        )
        from ml_platform.common.api.dtos import PartitionMaterializationRequest

        now = datetime.now(UTC)
        completed = {
            "partition_id": "00000000-0000-0000-0000-000000000002",
            "job_id": "00000000-0000-0000-0000-000000000001",
            "timestamp": datetime(2026, 10, 20, 18, tzinfo=UTC),
            "status": "created",
            "created_at": now,
            "updated_at": now,
        }
        materialization = MaterializationRequest(
            submitted_job=None,
            partitions=(
                MaterializationPartition(
                    id=UUID(completed["partition_id"]),
                    job_id=UUID(completed["job_id"]),
                    timestamp=completed["timestamp"],
                    status="created",
                    created_at=now,
                    updated_at=now,
                ),
            ),
        )
        session = MagicMock()
        with patch(
            "ml_platform.api.routers._datasets.DatasetRepository"
        ) as repository_type:
            repository_type.return_value.materialize.return_value = (
                materialization
            )
            response = materialize_dataset_partitions(
                "sp500",
                1,
                PartitionMaterializationRequest(
                    timestamps=[completed["timestamp"]]
                ),
                session,
            )
        self.assertIsNone(response.submitted_job)
        self.assertEqual(
            str(response.partitions[0].partition_id), completed["partition_id"]
        )
        repository_type.return_value.materialize.assert_called_once()

    def test_feature_force_replaces_current_partition(self) -> None:
        """The route adapts a submitted feature materialization domain result."""
        from ml_platform.api.routers import (
            materialize_feature_partitions,
        )
        from ml_platform.common.api.dtos import (
            FeaturePartitionMaterializationRequest,
        )

        now = datetime.now(UTC)
        materialization = MaterializationRequest(
            submitted_job=MaterializationJob(
                id=JobId(UUID("00000000-0000-0000-0000-000000000003")),
                type=JobType.FEATURE_MATERIALIZATION,
                status=JobStatus.SUBMITTED,
                created_at=now,
                updated_at=now,
            ),
            partitions=(
                MaterializationPartition(
                    id=UUID("00000000-0000-0000-0000-000000000004"),
                    job_id=UUID("00000000-0000-0000-0000-000000000003"),
                    timestamp=datetime(2026, 10, 20, 18, tzinfo=UTC),
                    status=PartitionStatus.PENDING,
                    created_at=now,
                    updated_at=now,
                ),
            ),
        )
        session = MagicMock()
        with patch(
            "ml_platform.api.routers._features.FeatureRepository"
        ) as repository_type:
            repository_type.return_value.materialize.return_value = (
                materialization
            )
            response = materialize_feature_partitions(
                "trailing_volatility_1d",
                1,
                FeaturePartitionMaterializationRequest.model_validate(
                    {
                        "timestamps": [datetime(2026, 10, 20, 18, tzinfo=UTC)],
                        "mode": "force",
                    }
                ),
                session,
            )
        self.assertEqual(
            response.partitions[0].partition_id,
            UUID("00000000-0000-0000-0000-000000000004"),
        )


class DashboardReadApiTest(unittest.TestCase):
    """Verify dashboard data endpoints expose resource and partition summaries."""

    def test_dataset_summary_returns_status_counts(self) -> None:
        """Dataset list is a read-only projection suitable for dashboard cards."""
        summary = DatasetSummary(
            definition=DatasetDefinition(
                name="sp500",
                version=3,
                dataset_type="market_data",
                config=MarketDataDatasetConfig(
                    keys=[EntityKey.SYMBOL],
                    partition_type=PartitionType.DAILY,
                    symbols=[Symbol("AAPL")],
                ),
                created_at=datetime.now(UTC),
                lifecycle=DatasetStatus.ACTIVE,
                deleted_at=None,
                deprecated_at=None,
            ),
            partitions=PartitionCounts(total=2, created=1, pending=1, failed=0),
        )
        session = MagicMock()
        with patch(
            "ml_platform.api.routers._datasets.DatasetRepository"
        ) as repository_type:
            repository_type.return_value.list.return_value = (summary,)
            result = list_datasets(session)

        self.assertEqual(result[0]["name"], "sp500")
        self.assertEqual(result[0]["created_partition_count"], 1)
        repository_type.return_value.list.assert_called_once_with(
            statuses=(DatasetStatus.ACTIVE, DatasetStatus.DEPRECATED)
        )

    def test_static_dashboard_is_served(self) -> None:
        """The browser UI is available without a separate frontend deployment."""
        client = TestClient(app)
        response = client.get("/ui/datasets/sp500/3")
        self.assertEqual(response.status_code, 200)
        self.assertIn("ML Platform", response.text)
        assets = client.get("/ui/assets/app.js")
        self.assertEqual(assets.status_code, 200)
        self.assertIn("renderDetail", assets.text)


class ResourceDeletionTest(unittest.TestCase):
    """Verify soft-deletion prevents future work while preserving history."""

    def _connection(self, rows: list[object]) -> tuple[MagicMock, MagicMock]:
        cursor = MagicMock()
        cursor.fetchone.side_effect = rows
        cursor.__enter__.return_value = cursor
        connection = MagicMock()
        connection.cursor.return_value = cursor
        return connection, cursor

    def test_delete_dataset_soft_deletes_and_cancels_pending_work(self) -> None:
        """Deleting a dataset retires its definition and any unfinished work."""
        from ml_platform.api.routers import delete_dataset

        dataset = SimpleNamespace(deleted_at=None)
        session = MagicMock()
        session.get.return_value = dataset

        delete_dataset("sp500", 3, session)

        self.assertIsNotNone(dataset.deleted_at)
        self.assertEqual(session.execute.call_count, 2)
        session.commit.assert_called_once()

    def test_deleted_dataset_cannot_materialize_partitions(self) -> None:
        """The partition endpoint rejects a resource that has been soft-deleted."""
        from fastapi import HTTPException

        from ml_platform.api.routers import (
            materialize_dataset_partitions,
        )
        from ml_platform.common.api.dtos import PartitionMaterializationRequest

        connection, _ = self._connection([None])
        with self.assertRaises(HTTPException) as raised:
            materialize_dataset_partitions(
                "sp500",
                3,
                PartitionMaterializationRequest(
                    timestamps=[datetime(2026, 10, 20, tzinfo=UTC)]
                ),
                connection,
            )
        self.assertEqual(raised.exception.status_code, 404)

    def test_dataset_list_rejects_an_empty_status_filter(self) -> None:
        """Reject an empty lifecycle-status query before repository access."""
        app.dependency_overrides[get_session] = lambda: MagicMock()
        try:
            client = TestClient(app)
            response = client.get("/datasets?statuses=")
        finally:
            app.dependency_overrides.clear()

        self.assertEqual(response.status_code, 422)

    def test_dataset_list_filters_by_explicit_statuses(self) -> None:
        """Catalog list calls default to non-deleted dataset statuses."""
        session = MagicMock()
        session.execute.return_value.all.return_value = []
        list_datasets(session)
        default_statement = session.execute.call_args.args[0]
        list_datasets(session, statuses=(DatasetStatus.DELETED,))
        deleted_statement = session.execute.call_args.args[0]

        self.assertIn("deleted_at IS NULL", str(default_statement))
        self.assertIn("deprecated_at IS NOT NULL", str(default_statement))
        self.assertIn("deleted_at IS NOT NULL", str(deleted_statement))


class MinioLocationTest(unittest.TestCase):
    """Verify control-plane metadata links resolve to Iceberg table prefixes."""

    def test_table_and_partition_links_use_the_minio_browser(self) -> None:
        with patch.dict(
            "os.environ",
            {"MINIO_CONSOLE_URL": "http://minio.test:9003"},
            clear=False,
        ):
            table_url = minio_browser_url("features/trailing_volatility_1d-v1")
            partition_url = minio_partition_url(
                "features/trailing_volatility_1d-v1",
                datetime(2026, 10, 20, tzinfo=UTC),
            )

        self.assertEqual(
            table_url,
            "http://minio.test:9003/browser/warehouse/features%2Ftrailing_volatility_1d-v1%2F",
        )
        self.assertEqual(
            partition_url,
            "http://minio.test:9003/browser/warehouse/features%2Ftrailing_volatility_1d-v1%2Fdata%2Ftimestamp_day%3D2026-10-20%2F",
        )
