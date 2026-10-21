"""Validation tests for materialization API routes."""

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from ml_platform.api.database import get_connection
from ml_platform.api.dtos import DatasetCreateRequest
from ml_platform.api.main import app, create_dataset, get_job
from ml_platform.api import main as api_main


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
            "name": "sp500", "version": 1, "dataset_type": "market_data",
            "config": {"symbols": ["AAPL"], "time_range": {"start_date": "2026-01-01", "end_date": "2026-01-02"}},
            "created_at": datetime.now(timezone.utc),
        }
        cursor = MagicMock()
        cursor.fetchone.side_effect = [row, {"id": "00000000-0000-0000-0000-000000000001"}]
        cursor.__enter__.return_value = cursor
        connection = MagicMock()
        connection.cursor.return_value = cursor
        payload = DatasetCreateRequest.model_validate({key: value for key, value in row.items() if key != "created_at"})

        response = create_dataset(payload, connection)

        self.assertEqual(response.name, "sp500")
        connection.commit.assert_called_once()
        self.assertTrue(any("submitted" in str(call) for call in cursor.execute.call_args_list))

    def test_dataset_partition_request_rejects_non_hour_timestamp(self) -> None:
        """Partition endpoints reject timestamps that are not aligned to the hour."""
        response = self.client.post(
            "/datasets/market_data/versions/1/partitions",
            json={"timestamps": ["2026-10-20T18:15:47.828Z"]},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("aligned to an exact hour", response.text)

    def test_feature_partition_request_accepts_hour_aligned_utc_timestamp(self) -> None:
        """Hour-aligned UTC timestamps create a feature materialization job."""
        now = datetime.now(timezone.utc)
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
            "timestamp": datetime(2026, 10, 20, 18, tzinfo=timezone.utc),
            "status": "pending",
            "created_at": now,
            "updated_at": now,
        }
        cursor = MagicMock()
        cursor.fetchone.side_effect = [{"exists": True}, None, None, job, partition]
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

    def test_feature_ensure_current_returns_completed_partition_without_a_new_job(self) -> None:
        """The default feature mode joins a matching materialized revision."""
        now = datetime.now(timezone.utc)
        completed = {
            "partition_id": "00000000-0000-0000-0000-000000000002",
            "job_id": "00000000-0000-0000-0000-000000000001",
            "timestamp": datetime(2026, 10, 20, 18, tzinfo=timezone.utc),
            "status": "created", "created_at": now, "updated_at": now,
        }
        cursor = MagicMock()
        cursor.fetchone.side_effect = [{"exists": True}]
        cursor.__enter__.return_value = cursor
        connection = MagicMock()
        connection.cursor.return_value = cursor
        app.dependency_overrides[get_connection] = lambda: connection
        with patch.object(api_main, "_plan_feature_inputs", return_value=["input-revision"]), \
             patch.object(api_main, "_completed_feature_request", return_value=completed), \
             patch.object(api_main, "_feature_plan_ids", return_value={"input-revision"}):
            response = self.client.post(
                "/features/trailing_volatility_1d/versions/1/partitions",
                json={"timestamps": ["2026-10-20T18:00:00Z"]},
            )
        self.assertEqual(response.status_code, 202)
        self.assertIsNone(response.json()["job"])
        self.assertEqual(response.json()["partitions"][0]["partition_id"], completed["partition_id"])
        self.assertFalse(any("INSERT INTO jobs" in str(call) for call in cursor.execute.call_args_list))

    def test_get_job_returns_persisted_job(self) -> None:
        """The job endpoint returns the database-backed aggregate status."""
        now = datetime.now(timezone.utc)
        row = {
            "id": "00000000-0000-0000-0000-000000000001",
            "type": "dataset_materialization",
            "status": "submitted",
            "created_at": now,
            "updated_at": now,
        }
        cursor = MagicMock()
        cursor.fetchone.return_value = row
        cursor.__enter__.return_value = cursor
        connection = MagicMock()
        connection.cursor.return_value = cursor

        response = get_job(row["id"], connection)

        self.assertEqual(str(response.id), row["id"])
        self.assertEqual(response.status, "submitted")

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
        from ml_platform.api.dtos import PartitionMaterializationRequest
        from ml_platform.api.main import materialize_dataset_partitions

        now = datetime.now(timezone.utc)
        completed = {"partition_id": "00000000-0000-0000-0000-000000000002", "job_id": "00000000-0000-0000-0000-000000000001", "timestamp": datetime(2026, 10, 20, 18, tzinfo=timezone.utc), "status": "created", "created_at": now, "updated_at": now}
        connection, cursor = self._connection([{"exists": True}, completed])
        response = materialize_dataset_partitions("sp500", 1, PartitionMaterializationRequest(timestamps=[completed["timestamp"]]), connection)
        self.assertIsNone(response.job)
        self.assertEqual(str(response.partitions[0].partition_id), completed["partition_id"])
        self.assertFalse(any("INSERT INTO jobs" in str(call) for call in cursor.execute.call_args_list))

    def test_feature_force_replaces_current_partition(self) -> None:
        """Force creates a new request even when recorded inputs still match."""
        from ml_platform.api.dtos import FeaturePartitionMaterializationRequest
        from ml_platform.api.main import materialize_feature_partitions

        now = datetime.now(timezone.utc)
        completed = {"partition_id": "00000000-0000-0000-0000-000000000002", "job_id": "00000000-0000-0000-0000-000000000001", "timestamp": datetime(2026, 10, 20, 18, tzinfo=timezone.utc), "status": "created", "created_at": now, "updated_at": now}
        job = {"id": "00000000-0000-0000-0000-000000000003", "type": "feature_materialization", "status": "submitted", "created_at": now, "updated_at": now}
        created = {**completed, "partition_id": "00000000-0000-0000-0000-000000000004", "job_id": job["id"], "status": "pending"}
        connection, _ = self._connection([{"exists": True}, None, job, created])
        with patch.object(api_main, "_plan_feature_inputs", return_value=["input"]), \
             patch.object(api_main, "_completed_feature_request", return_value=completed), \
             patch.object(api_main, "_feature_plan_ids", return_value={"input"}):
            response = materialize_feature_partitions("returns", 1, FeaturePartitionMaterializationRequest(timestamps=[completed["timestamp"]], mode="force"), connection)
        self.assertEqual(str(response.job.id), job["id"])
        self.assertEqual(response.partitions[0].status, "pending")

    def test_feature_completion_rejects_unplanned_parent_revisions(self) -> None:
        """A worker cannot report a receipt for a different input plan."""
        from fastapi import HTTPException
        from ml_platform.api.dtos import FeaturePartitionCompletionRequest
        from ml_platform.api.main import complete_feature_partition

        partition_id = "00000000-0000-0000-0000-000000000002"
        connection, cursor = self._connection([{"callback_token": "token"}])
        with patch.object(api_main, "_feature_plan_ids", return_value={"planned"}):
            with self.assertRaises(HTTPException) as raised:
                complete_feature_partition(
                    partition_id,
                    FeaturePartitionCompletionRequest(content_hash="hash", snapshot_id=1, dataset_partition_ids=["00000000-0000-0000-0000-000000000003"]),
                    "token",
                    connection,
                )
        self.assertEqual(raised.exception.status_code, 409)
        self.assertFalse(any("INSERT INTO feature_partitions" in str(call) for call in cursor.execute.call_args_list))
