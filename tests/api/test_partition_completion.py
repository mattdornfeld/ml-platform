"""PostgreSQL integration coverage for atomic partition completion receipts."""

import os
import unittest
from uuid import UUID, uuid4

import psycopg
from fastapi import HTTPException
from psycopg.rows import dict_row

from ml_platform.api.dtos import PartitionCompletionRequest
from ml_platform.api.main import complete_partitions
from ml_platform.api.migrations import migrate_database


@unittest.skipUnless(
    os.environ.get("ML_PLATFORM_TEST_DATABASE_URL"),
    "Requires a disposable PostgreSQL test database.",
)
class PartitionCompletionDatabaseTest(unittest.TestCase):
    """Verify all-or-nothing writes against real PostgreSQL constraints."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = os.environ["ML_PLATFORM_TEST_DATABASE_URL"]
        previous = os.environ.get("DATABASE_URL")
        os.environ["DATABASE_URL"] = cls.database_url
        try:
            migrate_database()
        finally:
            if previous is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = previous

    def setUp(self) -> None:
        with psycopg.connect(self.database_url) as connection:
            connection.execute("TRUNCATE datasets, features, labels, jobs CASCADE")
            connection.execute(
                "INSERT INTO datasets (name, version, dataset_type, config) VALUES ('prices', 1, 'market_data', '{}'::jsonb)"
            )
            job_id = uuid4()
            connection.execute(
                "INSERT INTO jobs (id, type) VALUES (%s, 'dataset_materialization')",
                (job_id,),
            )
            self.partition_ids = [uuid4(), uuid4()]
            for index, partition_id in enumerate(self.partition_ids):
                connection.execute(
                    """INSERT INTO partition_update_requests
                    (partition_id, job_id, dataset_name, dataset_version, timestamp, callback_token)
                    VALUES (%s, %s, 'prices', 1, %s, %s)""",
                    (
                        partition_id,
                        job_id,
                        f"2026-01-0{index + 1}T00:00:00Z",
                        f"token-{index}",
                    ),
                )
        self.job_id = job_id

    def _receipt(
        self,
        index: int,
        *,
        schema: dict[str, str] | None = None,
        token: str | None = None,
        content_hash: str | None = None,
    ) -> PartitionCompletionRequest:
        return PartitionCompletionRequest.model_validate(
            {
                "partition_id": self.partition_ids[index],
                "resource_type": "dataset",
                "row_count": index + 1,
                "schema": schema or {"timestamp": "timestamp", "symbol": "string"},
                "content_hash": content_hash or f"hash-{index}",
                "snapshot_id": 42,
                "callback_token": token or f"token-{index}",
            }
        )

    def _complete(self, receipts: list[PartitionCompletionRequest]):
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            return complete_partitions(receipts, connection)

    def _state(self) -> tuple[int, list[str], str]:
        with psycopg.connect(self.database_url) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM dataset_partitions"
            ).fetchone()[0]
            statuses = [
                row[0]
                for row in connection.execute(
                    "SELECT status::text FROM partition_update_requests ORDER BY timestamp"
                ).fetchall()
            ]
            job_status = connection.execute(
                "SELECT status::text FROM jobs WHERE id = %s", (self.job_id,)
            ).fetchone()[0]
        return count, statuses, job_status

    def test_batch_commits_all_receipts_and_identical_retry_succeeds(self) -> None:
        receipts = [self._receipt(0), self._receipt(1)]
        created = self._complete(receipts)
        retried = self._complete(receipts)

        self.assertEqual([item.partition_id for item in created], self.partition_ids)
        self.assertEqual([item.row_count for item in retried], [1, 2])
        self.assertEqual(self._state(), (2, ["created", "created"], "succeeded"))

    def test_invalid_token_rolls_back_every_item(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            self._complete([self._receipt(0), self._receipt(1, token="wrong")])
        self.assertEqual(raised.exception.status_code, 403)
        self.assertEqual(self._state(), (0, ["pending", "pending"], "submitted"))

    def test_mixed_schema_batch_rolls_back_every_item(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            self._complete(
                [self._receipt(0), self._receipt(1, schema={"timestamp": "string"})]
            )
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(self._state(), (0, ["pending", "pending"], "submitted"))

    def test_database_error_after_first_insert_rolls_back_batch(self) -> None:
        """A late SQL failure cannot leave a partial completion receipt."""
        overflowing = self._receipt(1).model_copy(update={"row_count": 2**63})
        with self.assertRaises(psycopg.errors.NumericValueOutOfRange):
            self._complete([self._receipt(0), overflowing])
        self.assertEqual(self._state(), (0, ["pending", "pending"], "submitted"))

    def test_conflicting_retry_is_rejected(self) -> None:
        self._complete([self._receipt(0)])
        with self.assertRaises(HTTPException) as raised:
            self._complete(
                [self._receipt(0, content_hash="different"), self._receipt(1)]
            )
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(self._state(), (1, ["created", "pending"], "submitted"))

    def test_existing_table_schema_conflict_is_rejected(self) -> None:
        self._complete([self._receipt(0)])
        with self.assertRaises(HTTPException) as raised:
            self._complete([self._receipt(1, schema={"timestamp": "string"})])
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(self._state(), (1, ["created", "pending"], "submitted"))

    def test_duplicate_partition_id_is_rejected(self) -> None:
        receipt = self._receipt(0)
        with self.assertRaises(HTTPException) as raised:
            self._complete([receipt, receipt])
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(self._state(), (0, ["pending", "pending"], "submitted"))

    def _derived_request(self, resource_type: str) -> tuple[UUID, UUID]:
        """Create one feature or label request pinned to a completed dataset revision."""
        self._complete([self._receipt(0)])
        partition_id, job_id = uuid4(), uuid4()
        with psycopg.connect(self.database_url) as connection:
            if resource_type == "feature":
                connection.execute("""INSERT INTO features
                    (name, version, dtype, frequency, feature_keys, transform, description)
                    VALUES ('derived', 1, 'double', 'daily', '["symbol"]'::jsonb, 'example.Transform', 'Derived data')""")
            else:
                connection.execute("""INSERT INTO labels (name, version, transform, description)
                    VALUES ('derived', 1, 'example.Transform', 'Derived labels')""")
            connection.execute(
                "INSERT INTO jobs (id, type) VALUES (%s, %s::job_type)",
                (job_id, f"{resource_type}_materialization"),
            )
            connection.execute(
                f"""INSERT INTO {resource_type}_partition_update_requests
                (partition_id, job_id, {resource_type}_name, {resource_type}_version, timestamp, callback_token)
                VALUES (%s, %s, 'derived', 1, '2026-01-01T00:00:00Z', 'derived-token')""",
                (partition_id, job_id),
            )
            connection.execute(
                f"""INSERT INTO {resource_type}_partition_update_request_dataset_dependencies
                ({resource_type}_partition_request_id, dataset_partition_id) VALUES (%s, %s)""",
                (partition_id, self.partition_ids[0]),
            )
        return partition_id, job_id

    def _derived_receipt(
        self, resource_type: str, partition_id: UUID, parent_id: UUID
    ) -> PartitionCompletionRequest:
        return PartitionCompletionRequest.model_validate(
            {
                "partition_id": partition_id,
                "resource_type": resource_type,
                "row_count": 3,
                "schema": {
                    "timestamp": "timestamp",
                    "symbol": "string",
                    "feature_value": "double",
                },
                "content_hash": "derived-hash",
                "snapshot_id": 51,
                "callback_token": "derived-token",
                "dataset_partition_ids": [parent_id],
            }
        )

    def test_feature_receipt_preserves_pinned_dependency(self) -> None:
        partition_id, job_id = self._derived_request("feature")
        receipt = self._derived_receipt("feature", partition_id, self.partition_ids[0])
        self._complete([receipt])
        self._complete([receipt])
        with psycopg.connect(self.database_url) as connection:
            parent_id = connection.execute(
                """SELECT dataset_partition_id
                FROM feature_partition_dataset_dependencies WHERE feature_partition_id = %s""",
                (partition_id,),
            ).fetchone()[0]
            status = connection.execute(
                "SELECT status::text FROM jobs WHERE id = %s", (job_id,)
            ).fetchone()[0]
        self.assertEqual(parent_id, self.partition_ids[0])
        self.assertEqual(status, "succeeded")

    def test_label_receipt_rejects_wrong_plan_then_preserves_correct_plan(self) -> None:
        partition_id, job_id = self._derived_request("label")
        wrong = self._derived_receipt("label", partition_id, self.partition_ids[1])
        with self.assertRaises(HTTPException) as raised:
            self._complete([wrong])
        self.assertEqual(raised.exception.status_code, 409)
        correct = self._derived_receipt("label", partition_id, self.partition_ids[0])
        self._complete([correct])
        with psycopg.connect(self.database_url) as connection:
            parent_id = connection.execute(
                """SELECT dataset_partition_id
                FROM label_partition_dataset_dependencies WHERE label_partition_id = %s""",
                (partition_id,),
            ).fetchone()[0]
            status = connection.execute(
                "SELECT status::text FROM jobs WHERE id = %s", (job_id,)
            ).fetchone()[0]
        self.assertEqual(parent_id, self.partition_ids[0])
        self.assertEqual(status, "succeeded")


if __name__ == "__main__":
    unittest.main()
