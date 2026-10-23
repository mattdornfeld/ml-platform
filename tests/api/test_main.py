"""Validation tests for materialization API routes."""

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from ml_platform.api import main as api_main
from ml_platform.api.database import get_connection
from ml_platform.api.main import app, create_dataset, get_job
from ml_platform.common.api.dtos import DatasetCreateRequest


class PartitionMaterializationRequestTest(unittest.TestCase):
    """Verify API validation for partition request timestamps."""

    def setUp(self) -> None:
        """Create an isolated in-process API client."""
        app.dependency_overrides[get_connection] = lambda: MagicMock()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        """Discard dependency overrides after each API test."""
        app.dependency_overrides.clear()

    def test_create_dataset_enqueues_materialization_job(self) -> None:
        """Registration creates the authoritative submitted materialization job."""
        row = {
            "name": "sp500",
            "version": 1,
            "dataset_type": "market_data",
            "config": {
                "keys": ["symbol"],
                "partition_type": "daily",
                "symbols": ["AAPL"],
                "time_range": {
                    "start_date": "2026-01-01",
                    "end_date": "2026-01-02",
                },
            },
            "created_at": datetime.now(UTC),
        }
        cursor = MagicMock()
        cursor.fetchone.side_effect = [
            row,
            {"id": "00000000-0000-0000-0000-000000000001"},
        ]
        cursor.__enter__.return_value = cursor
        connection = MagicMock()
        connection.cursor.return_value = cursor
        payload = DatasetCreateRequest.model_validate(
            {key: value for key, value in row.items() if key != "created_at"}
        )

        response = create_dataset(payload, connection)

        self.assertEqual(response.name, "sp500")
        self.assertEqual(response.id, "sp500_v1")
        connection.commit.assert_called_once()
        self.assertTrue(
            any(
                "submitted" in str(call)
                for call in cursor.execute.call_args_list
            )
        )

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
                    "time_range": {
                        "start_date": "2026-01-01",
                        "end_date": "2026-01-02",
                    },
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
        self.assertEqual(
            spec["paths"]["/partitions/complete"]["post"]["requestBody"][
                "content"
            ]["application/json"]["schema"]["type"],
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

    def test_feature_partition_request_accepts_hour_aligned_utc_timestamp(
        self,
    ) -> None:
        """Hour-aligned UTC timestamps create a feature materialization job."""
        now = datetime.now(UTC)
        job = {
            "id": "00000000-0000-0000-0000-000000000001",
            "type": "feature_materialization",
            "status": "submitted",
            "created_at": now,
            "updated_at": now,
        }
        partition = {
            "partition_id": "00000000-0000-0000-0000-000000000002",
            "job_id": job["id"],
            "timestamp": datetime(2026, 10, 20, 18, tzinfo=UTC),
            "status": "pending",
            "created_at": now,
            "updated_at": now,
        }
        cursor = MagicMock()
        cursor.fetchone.side_effect = [
            {"exists": True},
            None,
            None,
            None,
            job,
            partition,
        ]
        cursor.fetchall.return_value = []
        cursor.__enter__.return_value = cursor
        connection = MagicMock()
        connection.cursor.return_value = cursor
        app.dependency_overrides[get_connection] = lambda: connection

        response = self.client.post(
            "/features/trailing_volatility_1d/versions/1/partitions",
            json={"timestamps": ["2026-10-20T18:00:00Z"]},
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["job"]["id"], job["id"])
        self.assertEqual(response.json()["partitions"][0]["status"], "pending")

    def test_feature_ensure_current_returns_completed_partition_without_a_new_job(
        self,
    ) -> None:
        """The default feature mode joins a matching materialized revision."""
        now = datetime.now(UTC)
        completed = {
            "partition_id": "00000000-0000-0000-0000-000000000002",
            "job_id": "00000000-0000-0000-0000-000000000001",
            "timestamp": datetime(2026, 10, 20, 18, tzinfo=UTC),
            "status": "created",
            "created_at": now,
            "updated_at": now,
        }
        cursor = MagicMock()
        cursor.fetchone.side_effect = [{"exists": True}, None]
        cursor.__enter__.return_value = cursor
        connection = MagicMock()
        connection.cursor.return_value = cursor
        app.dependency_overrides[get_connection] = lambda: connection
        with (
            patch.object(
                api_main,
                "_plan_feature_inputs",
                return_value=["input-revision"],
            ),
            patch.object(
                api_main, "_completed_feature_request", return_value=completed
            ),
            patch.object(
                api_main, "_feature_plan_ids", return_value={"input-revision"}
            ),
        ):
            response = self.client.post(
                "/features/trailing_volatility_1d/versions/1/partitions",
                json={"timestamps": ["2026-10-20T18:00:00Z"]},
            )
        self.assertEqual(response.status_code, 202)
        self.assertIsNone(response.json()["job"])
        self.assertEqual(
            response.json()["partitions"][0]["partition_id"],
            completed["partition_id"],
        )
        self.assertFalse(
            any(
                "INSERT INTO jobs" in str(call)
                for call in cursor.execute.call_args_list
            )
        )

    def test_get_job_returns_persisted_job(self) -> None:
        """The job endpoint returns the database-backed aggregate status."""
        now = datetime.now(UTC)
        row = {
            "id": "00000000-0000-0000-0000-000000000001",
            "type": "dataset_materialization",
            "status": "submitted",
            "created_at": now,
            "updated_at": now,
        }
        cursor = MagicMock()
        cursor.fetchone.return_value = row
        cursor.fetchall.side_effect = [
            [{"resource_type": "dataset", "name": "sp500", "version": 3}],
            [
                {
                    "resource_type": "dataset",
                    "name": "sp500",
                    "version": 3,
                    "timestamp": datetime(2026, 10, 20, 18, tzinfo=UTC),
                    "status": "created",
                }
            ],
        ]
        cursor.__enter__.return_value = cursor
        connection = MagicMock()
        connection.cursor.return_value = cursor

        with (
            patch.dict(
                "os.environ",
                {"HEADLAMP_URL": "http://headlamp.test"},
                clear=False,
            ),
            patch(
                "ml_platform.api.main.kubernetes_client.CoreV1Api"
            ) as core_api,
            patch("ml_platform.api.main.kubernetes_client.CustomObjectsApi"),
        ):
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
            response = get_job(row["id"], connection)

        self.assertEqual(str(response.id), row["id"])
        self.assertEqual(response.status, "submitted")
        self.assertEqual(response.targets[0].name, "sp500")
        self.assertEqual(response.targets[0].version, 3)
        self.assertEqual(response.partitions[0].status, "created")
        self.assertEqual(
            response.spark_application_name,
            "dataset-00000000-0000-0000-0000-000000000001",
        )
        self.assertEqual(
            response.headlamp_url,
            "http://headlamp.test/c/main/customresources/"
            "sparkapplications.sparkoperator.k8s.io/ml-platform/"
            "dataset-00000000-0000-0000-0000-000000000001",
        )
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
        self.assertEqual(
            response.kubernetes_resources[1].url,
            "http://headlamp.test/c/main/pods/ml-platform/dataset-job-driver",
        )
        self.assertEqual(
            response.kubernetes_resources[2].resource_type, "Executor Pod"
        )
        self.assertIn(
            "partition_update_requests", str(cursor.execute.call_args_list)
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
        from ml_platform.api.main import materialize_dataset_partitions
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
        connection, cursor = self._connection([{"exists": True}, completed])
        response = materialize_dataset_partitions(
            "sp500",
            1,
            PartitionMaterializationRequest(
                timestamps=[completed["timestamp"]]
            ),
            connection,
        )
        self.assertIsNone(response.job)
        self.assertEqual(
            str(response.partitions[0].partition_id), completed["partition_id"]
        )
        self.assertFalse(
            any(
                "INSERT INTO jobs" in str(call)
                for call in cursor.execute.call_args_list
            )
        )

    def test_feature_force_replaces_current_partition(self) -> None:
        """Force creates a new request even when recorded inputs still match."""
        from ml_platform.api.main import materialize_feature_partitions
        from ml_platform.common.api.dtos import (
            FeaturePartitionMaterializationRequest,
        )

        now = datetime.now(UTC)
        completed = {
            "partition_id": "00000000-0000-0000-0000-000000000002",
            "job_id": "00000000-0000-0000-0000-000000000001",
            "timestamp": datetime(2026, 10, 20, 18, tzinfo=UTC),
            "status": "created",
            "created_at": now,
            "updated_at": now,
        }
        job = {
            "id": "00000000-0000-0000-0000-000000000003",
            "type": "feature_materialization",
            "status": "submitted",
            "created_at": now,
            "updated_at": now,
        }
        created = {
            **completed,
            "partition_id": "00000000-0000-0000-0000-000000000004",
            "job_id": job["id"],
            "status": "pending",
        }
        connection, _ = self._connection(
            [{"exists": True}, None, None, job, created]
        )
        with (
            patch.object(
                api_main, "_plan_feature_inputs", return_value=["input"]
            ),
            patch.object(
                api_main, "_completed_feature_request", return_value=completed
            ),
            patch.object(api_main, "_feature_plan_ids", return_value={"input"}),
        ):
            response = materialize_feature_partitions(
                "returns",
                1,
                FeaturePartitionMaterializationRequest(
                    timestamps=[completed["timestamp"]], mode="force"
                ),
                connection,
            )
        self.assertEqual(str(response.job.id), job["id"])
        self.assertEqual(response.partitions[0].status, "pending")

    def test_feature_completion_rejects_unplanned_parent_revisions(
        self,
    ) -> None:
        """A worker cannot report a receipt for a different input plan."""
        from fastapi import HTTPException

        from ml_platform.api.main import complete_partitions
        from ml_platform.common.api.dtos import PartitionCompletionRequest

        partition_id = "00000000-0000-0000-0000-000000000002"
        connection, cursor = self._connection(
            [
                {"name": "returns", "version": 1},
                {
                    "callback_token": "token",
                    "name": "returns",
                    "version": 1,
                    "deleted_at": None,
                    "status": "pending",
                    "job_status": "submitted",
                    "job_id": "00000000-0000-0000-0000-000000000001",
                },
            ]
        )
        with (
            patch.object(
                api_main, "_completion_plan_ids", return_value={"planned"}
            ),
            self.assertRaises(HTTPException) as raised,
        ):
            complete_partitions(
                [
                    PartitionCompletionRequest.model_validate(
                        {
                            "partition_id": partition_id,
                            "resource_type": "feature",
                            "row_count": 1,
                            "schema": {
                                "timestamp": "timestamp",
                                "symbol": "string",
                                "feature_value": "double",
                            },
                            "content_hash": "hash",
                            "snapshot_id": 1,
                            "callback_token": "token",
                            "dataset_partition_ids": [
                                "00000000-0000-0000-0000-000000000003"
                            ],
                        }
                    )
                ],
                connection,
            )
        self.assertEqual(raised.exception.status_code, 409)
        self.assertFalse(
            any(
                "INSERT INTO feature_partitions" in str(call)
                for call in cursor.execute.call_args_list
            )
        )


class DashboardReadApiTest(unittest.TestCase):
    """Verify dashboard data endpoints expose resource and partition summaries."""

    def test_dataset_summary_returns_status_counts(self) -> None:
        """Dataset list is a read-only projection suitable for the dashboard cards."""
        from ml_platform.api.main import list_datasets

        cursor = MagicMock()
        cursor.fetchall.return_value = [
            {
                "name": "sp500",
                "version": 3,
                "dataset_type": "market_data",
                "config": {},
                "created_at": datetime.now(UTC),
                "partition_count": 2,
                "created_partition_count": 1,
                "pending_partition_count": 1,
                "failed_partition_count": 0,
            }
        ]
        cursor.__enter__.return_value = cursor
        connection = MagicMock()
        connection.cursor.return_value = cursor

        result = list_datasets(connection)

        self.assertEqual(result[0]["name"], "sp500")
        self.assertEqual(result[0]["created_partition_count"], 1)
        self.assertIn(
            "FILTER (WHERE request.status = 'created')",
            str(cursor.execute.call_args),
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
        from ml_platform.api.main import delete_dataset

        connection, cursor = self._connection([{"name": "sp500"}])
        delete_dataset("sp500", 3, connection)
        sql = "\n".join(str(call) for call in cursor.execute.call_args_list)
        self.assertIn("deleted_at", sql)
        self.assertIn("cancelled", sql)
        connection.commit.assert_called_once()

    def test_deleted_dataset_cannot_materialize_partitions(self) -> None:
        """The partition endpoint rejects a resource that has been soft-deleted."""
        from fastapi import HTTPException

        from ml_platform.api.main import materialize_dataset_partitions
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

    def test_dataset_list_excludes_deleted_unless_requested(self) -> None:
        """Catalog list calls keep soft-deleted versions hidden by default."""
        from ml_platform.api.main import list_datasets

        connection, cursor = self._connection([])
        cursor.fetchall.return_value = []
        list_datasets(connection)
        self.assertEqual(cursor.execute.call_args.args[1], (False,))
        list_datasets(connection, include_deleted=True)
        self.assertEqual(cursor.execute.call_args.args[1], (True,))


class MinioLocationTest(unittest.TestCase):
    """Verify control-plane metadata links resolve to Iceberg table prefixes."""

    def test_table_and_partition_links_use_the_minio_browser(self) -> None:
        resource = {
            "partitions": [{"timestamp": datetime(2026, 10, 20, tzinfo=UTC)}]
        }
        with patch.dict(
            "os.environ",
            {"MINIO_CONSOLE_URL": "http://minio.test:9003"},
            clear=False,
        ):
            api_main._attach_minio_locations(
                resource, "features/trailing_volatility_1d-v1"
            )

        self.assertEqual(
            resource["minio_table_url"],
            "http://minio.test:9003/browser/warehouse/features%2Ftrailing_volatility_1d-v1%2F",
        )
        self.assertEqual(
            resource["partitions"][0]["minio_url"],
            "http://minio.test:9003/browser/warehouse/features%2Ftrailing_volatility_1d-v1%2Fdata%2Ftimestamp_day%3D2026-10-20%2F",
        )
