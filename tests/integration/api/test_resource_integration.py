"""PostgreSQL-backed API tests for resource and materialization workflows."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Barrier
from uuid import UUID, uuid4

import psycopg
from fastapi.testclient import TestClient
from psycopg import sql
from sqlalchemy.orm import Session

from ml_platform.api.infrastructure import engine_for
from ml_platform.api.repositories import DatasetRepository, SchedulerRepository
from ml_platform.common.api.dtos import ResourceType
from ml_platform.common.types import (
    JobId,
    PartitionTimestamp,
    ResourceName,
    ResourceVersion,
)


def _dataset_payload() -> dict[str, object]:
    """Return a minimal valid market-data dataset definition."""
    return {
        "name": "prices",
        "version": 1,
        "dataset_type": "market_data",
        "config": {
            "description": "Daily market prices.",
            "keys": ["symbol"],
            "partition_type": "daily",
            "symbols": ["AAPL"],
        },
    }


def _feature_payload() -> dict[str, object]:
    """Return a minimal feature that depends on the prices dataset."""
    return {
        "name": "daily_return",
        "version": 1,
        "dtype": "double",
        "frequency": "daily",
        "keys": ["symbol"],
        "datasets": [
            {
                "name": "prices",
                "version": 1,
                "input_window": {"lookback_partitions": 0},
            }
        ],
        "transform": "ml_platform.data_sdk.feature_transforms.ReturnTransform",
        "description": "Daily return.",
        "parameters": {},
        "validations": [],
    }


def _label_payload() -> dict[str, object]:
    """Return a minimal label that depends on the prices dataset."""
    return {
        "name": "direction",
        "version": 1,
        "dtype": "double",
        "frequency": "daily",
        "keys": ["symbol"],
        "datasets": [
            {
                "name": "prices",
                "version": 1,
                "input_window": {
                    "lookback_partitions": 0,
                    "lookahead_partitions": 0,
                },
            }
        ],
        "transform": "ml_platform.data_sdk.label_transforms.DirectionTransform",
        "description": "Next-period direction.",
        "parameters": {},
        "validations": [],
    }


def _seed_created_dataset_partition(database_url: str) -> UUID:
    """Insert a completed parent partition for feature and label planning tests."""
    partition_id = uuid4()
    job_id = uuid4()
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    with psycopg.connect(database_url) as connection:
        connection.execute(
            "INSERT INTO jobs (id, type, status) VALUES (%s, 'dataset_materialization', 'succeeded')",
            (job_id,),
        )
        connection.execute(
            """INSERT INTO partition_update_requests
            (partition_id, job_id, dataset_name, dataset_version, timestamp, status, callback_token)
            VALUES (%s, %s, 'prices', 1, %s, 'created', 'parent-token')""",
            (partition_id, job_id, timestamp),
        )
        connection.execute(
            """INSERT INTO dataset_partitions
            (partition_id, content_hash, row_count, schema, snapshot_id)
            VALUES (%s, 'parent-hash', 1, '{"timestamp": "timestamp"}'::jsonb, 1)""",
            (partition_id,),
        )
    return partition_id


def _complete_transform_partition(
    api_client: TestClient,
    database_url: str,
    *,
    resource_type: ResourceType,
    partition_id: str,
    dataset_partition_id: UUID,
) -> None:
    """Complete one submitted derived partition through its authenticated API receipt."""
    match resource_type:
        case ResourceType.FEATURE:
            request_table = "feature_partition_update_requests"
            value_column = "feature_value"
        case ResourceType.LABEL:
            request_table = "label_partition_update_requests"
            value_column = "label_value"
        case _ as unsupported_resource_type:
            raise AssertionError(
                f"Expected a derived resource type, got {unsupported_resource_type}."
            )
    with psycopg.connect(database_url) as connection:
        callback_token = connection.execute(
            sql.SQL(
                "SELECT callback_token FROM {} WHERE partition_id = %s"
            ).format(sql.Identifier(request_table)),
            (UUID(partition_id),),
        ).fetchone()[0]
    response = api_client.post(
        "/partitions/complete",
        json=[
            {
                "partition_id": partition_id,
                "resource_type": resource_type.value,
                "row_count": 1,
                "iceberg_schema": {
                    "timestamp": "timestamp",
                    "symbol": "string",
                    value_column: "double",
                },
                "content_hash": f"{resource_type.value}-hash-{partition_id}",
                "snapshot_id": 1,
                "callback_token": callback_token,
                "dataset_partition_ids": [str(dataset_partition_id)],
            }
        ],
    )
    assert response.status_code == 200


def _seed_newer_dataset_partition(database_url: str) -> UUID:
    """Insert a later completed revision of the shared source partition."""
    partition_id = uuid4()
    job_id = uuid4()
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    revision_time = datetime(2030, 1, 1, tzinfo=UTC)
    with psycopg.connect(database_url) as connection:
        connection.execute(
            """INSERT INTO jobs (id, type, status, created_at, updated_at)
            VALUES (%s, 'dataset_materialization', 'succeeded', %s, %s)""",
            (job_id, revision_time, revision_time),
        )
        connection.execute(
            """INSERT INTO partition_update_requests
            (partition_id, job_id, dataset_name, dataset_version, timestamp, status,
             callback_token, created_at, updated_at)
            VALUES (%s, %s, 'prices', 1, %s, 'created', 'new-parent-token', %s, %s)""",
            (partition_id, job_id, timestamp, revision_time, revision_time),
        )
        connection.execute(
            """INSERT INTO dataset_partitions
            (partition_id, content_hash, row_count, schema, snapshot_id, created_at)
            VALUES (%s, 'new-parent-hash', 1, '{"timestamp": "timestamp"}'::jsonb,
                    2, %s)""",
            (partition_id, revision_time),
        )
    return partition_id


def test_dataset_lifecycle_materialization_and_job_inspection(
    api_client: TestClient,
) -> None:
    """Persist a dataset, create a partition job, inspect it, then soft-delete it."""
    created = api_client.post("/datasets", json=_dataset_payload())
    assert created.status_code == 201
    assert created.json()["id"] == "prices_v1"

    listed = api_client.get("/datasets")
    assert listed.status_code == 200
    assert listed.json()[0]["partition_count"] == 0

    materialized = api_client.post(
        "/datasets/prices/versions/1/partitions",
        json={"timestamps": ["2026-01-01T00:00:00Z"]},
    )
    assert materialized.status_code == 202
    body = materialized.json()
    job_id = body["submitted_job"]["id"]
    assert body["partitions"][0]["status"] == "pending"

    job = api_client.get(f"/jobs/{job_id}")
    assert job.status_code == 200
    assert job.json()["targets"] == [
        {"resource_type": "dataset", "name": "prices", "version": 1}
    ]

    updated = api_client.patch(
        "/datasets/prices/versions/1",
        json={"description": "Updated market prices."},
    )
    assert updated.status_code == 200
    assert updated.json()["config"]["description"] == "Updated market prices."

    active_payload = _dataset_payload()
    active_payload["name"] = "current_prices"
    assert api_client.post("/datasets", json=active_payload).status_code == 201

    deleted = api_client.delete("/datasets/prices/versions/1")
    assert deleted.status_code == 204
    assert [item["name"] for item in api_client.get("/datasets").json()] == [
        "current_prices"
    ]
    included = api_client.get("/datasets?statuses=deleted")
    assert included.json()[0]["deleted_at"] is not None
    multiple_statuses = api_client.get(
        "/datasets?statuses=active&statuses=deleted"
    )
    assert [item["name"] for item in multiple_statuses.json()] == [
        "current_prices",
        "prices",
    ]
    assert (
        api_client.post(
            "/datasets/prices/versions/1/partitions",
            json={"timestamps": ["2026-01-02T00:00:00Z"]},
        ).status_code
        == 404
    )


def test_feature_and_label_lifecycle_and_materialization(
    api_client: TestClient, clean_postgres: str
) -> None:
    """Register dependent resources and submit their partition materializations."""
    assert (
        api_client.post("/datasets", json=_dataset_payload()).status_code == 201
    )
    _seed_created_dataset_partition(clean_postgres)

    feature = api_client.post("/features", json=_feature_payload())
    label = api_client.post("/labels", json=_label_payload())
    assert feature.status_code == 201
    assert label.status_code == 201

    feature_detail = api_client.get("/features/daily_return/versions/1")
    assert feature_detail.status_code == 200
    assert feature_detail.json()["datasets"][0]["name"] == "prices"
    label_detail = api_client.get("/labels/direction/versions/1")
    assert label_detail.status_code == 200
    assert label_detail.json()["datasets"][0]["input_window"] == {
        "lookback_partitions": 0,
        "lookahead_partitions": 0,
    }

    partition_payload = {"timestamps": ["2026-01-01T00:00:00Z"]}
    feature_job = api_client.post(
        "/features/daily_return/versions/1/partitions",
        json=partition_payload,
    )
    label_job = api_client.post(
        "/labels/direction/versions/1/partitions",
        json=partition_payload,
    )
    assert feature_job.status_code == 202
    assert label_job.status_code == 202
    assert (
        feature_job.json()["submitted_job"]["type"] == "feature_materialization"
    )
    assert label_job.json()["submitted_job"]["type"] == "label_materialization"

    assert (
        api_client.delete("/features/daily_return/versions/1").status_code
        == 204
    )
    assert api_client.delete("/labels/direction/versions/1").status_code == 204
    assert api_client.get("/features").json() == []
    assert api_client.get("/labels").json() == []
    assert (
        api_client.get("/features?include_deleted=true").json()[0]["deleted_at"]
        is not None
    )
    assert (
        api_client.get("/labels?include_deleted=true").json()[0]["deleted_at"]
        is not None
    )


def test_dependent_resources_reject_deleted_or_missing_datasets(
    api_client: TestClient,
) -> None:
    """Reject feature and label definitions that cannot bind active input datasets."""
    missing_feature = api_client.post("/features", json=_feature_payload())
    missing_label = api_client.post("/labels", json=_label_payload())
    assert missing_feature.status_code == 422
    assert missing_label.status_code == 422

    assert (
        api_client.post("/datasets", json=_dataset_payload()).status_code == 201
    )
    assert api_client.delete("/datasets/prices/versions/1").status_code == 204
    assert (
        api_client.post("/features", json=_feature_payload()).status_code == 422
    )
    assert api_client.post("/labels", json=_label_payload()).status_code == 422


def test_scheduler_repository_builds_job_inputs_and_enforces_fifo_eligibility(
    api_client: TestClient, clean_postgres: str
) -> None:
    """Exercise scheduler persistence against real materialization requests."""
    assert (
        api_client.post("/datasets", json=_dataset_payload()).status_code == 201
    )
    dataset_request = api_client.post(
        "/datasets/prices/versions/1/partitions",
        json={"timestamps": ["2026-01-01T00:00:00Z"]},
    )
    dataset_job_id = UUID(dataset_request.json()["submitted_job"]["id"])
    scheduler = SchedulerRepository(clean_postgres)

    submitted = scheduler.submitted_jobs()
    assert [item["id"] for item in submitted] == [dataset_job_id]
    dataset_job = scheduler.dataset_job(JobId(dataset_job_id))
    assert dataset_job["name"] == "prices"
    assert (
        dataset_job["callbacks"][0]["timestamp"] == "2026-01-01T00:00:00+00:00"
    )
    assert scheduler.is_eligible(
        JobId(dataset_job_id), "dataset_materialization"
    )

    scheduler.finish(
        JobId(dataset_job_id), "succeeded", "dataset_materialization"
    )
    with psycopg.connect(clean_postgres) as connection:
        status = connection.execute(
            "SELECT status::text FROM jobs WHERE id = %s", (dataset_job_id,)
        ).fetchone()
    assert status == ("failed",)


def test_scheduler_repository_builds_feature_and_label_job_inputs(
    api_client: TestClient, clean_postgres: str
) -> None:
    """Build transform-worker inputs from persisted feature and label plans."""
    assert (
        api_client.post("/datasets", json=_dataset_payload()).status_code == 201
    )
    _seed_created_dataset_partition(clean_postgres)
    assert (
        api_client.post("/features", json=_feature_payload()).status_code == 201
    )
    assert api_client.post("/labels", json=_label_payload()).status_code == 201
    partition_payload = {"timestamps": ["2026-01-01T00:00:00Z"]}
    feature_response = api_client.post(
        "/features/daily_return/versions/1/partitions", json=partition_payload
    )
    label_response = api_client.post(
        "/labels/direction/versions/1/partitions", json=partition_payload
    )
    feature_job_id = JobId(feature_response.json()["submitted_job"]["id"])
    label_job_id = JobId(label_response.json()["submitted_job"]["id"])
    scheduler = SchedulerRepository(clean_postgres)

    feature_job = scheduler.feature_job(feature_job_id)
    assert feature_job["config"]["name"] == "daily_return"
    assert (
        feature_job["callbacks"][0]["partition_dependencies"][0]["dataset_name"]
        == "prices"
    )
    label_job = scheduler.label_job(label_job_id)
    assert label_job["config"]["name"] == "direction"
    assert (
        label_job["callbacks"][0]["partition_dependencies"][0]["dataset_name"]
        == "prices"
    )

    scheduler.finish(feature_job_id, "cancelled", "feature_materialization")
    scheduler.finish(label_job_id, "failed", "label_materialization")
    with psycopg.connect(clean_postgres) as connection:
        statuses = connection.execute(
            "SELECT type::text, status::text FROM jobs WHERE id IN (%s, %s) ORDER BY type",
            (feature_job_id, label_job_id),
        ).fetchall()
    assert statuses == [
        ("feature_materialization", "cancelled"),
        ("label_materialization", "failed"),
    ]
    with psycopg.connect(clean_postgres) as connection:
        partition_statuses = connection.execute(
            """SELECT type::text, status::text
            FROM (
                SELECT jobs.type, feature_partition_update_requests.status
                FROM feature_partition_update_requests
                JOIN jobs ON jobs.id = feature_partition_update_requests.job_id
                WHERE feature_partition_update_requests.job_id = %s
                UNION ALL
                SELECT jobs.type, label_partition_update_requests.status
                FROM label_partition_update_requests
                JOIN jobs ON jobs.id = label_partition_update_requests.job_id
                WHERE label_partition_update_requests.job_id = %s
            ) AS partition_statuses
            ORDER BY type""",
            (feature_job_id, label_job_id),
        ).fetchall()
    assert partition_statuses == [
        ("feature_materialization", "cancelled"),
        ("label_materialization", "failed"),
    ]


def _seed_created_hourly_dataset_partitions(database_url: str) -> None:
    """Insert three contiguous hourly source partitions for dependency planning."""
    timestamps = (
        datetime(2026, 1, 1, 0, tzinfo=UTC),
        datetime(2026, 1, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 1, 2, tzinfo=UTC),
    )
    with psycopg.connect(database_url) as connection:
        for timestamp in timestamps:
            partition_id = uuid4()
            job_id = uuid4()
            connection.execute(
                "INSERT INTO jobs (id, type, status) VALUES (%s, 'dataset_materialization', 'succeeded')",
                (job_id,),
            )
            connection.execute(
                """INSERT INTO partition_update_requests
                (partition_id, job_id, dataset_name, dataset_version, timestamp, status, callback_token)
                VALUES (%s, %s, 'prices', 1, %s, 'created', %s)""",
                (
                    partition_id,
                    job_id,
                    timestamp,
                    f"parent-token-{partition_id}",
                ),
            )
            connection.execute(
                """INSERT INTO dataset_partitions
                (partition_id, content_hash, row_count, schema, snapshot_id)
                VALUES (%s, 'parent-hash', 1, '{\"timestamp\": \"timestamp\"}'::jsonb, 1)""",
                (partition_id,),
            )


def test_resource_partition_requests_require_the_registered_cadence(
    api_client: TestClient,
) -> None:
    """Reject daily requests outside midnight while accepting hourly partitions."""
    assert (
        api_client.post("/datasets", json=_dataset_payload()).status_code == 201
    )
    daily_response = api_client.post(
        "/datasets/prices/versions/1/partitions",
        json={"timestamps": ["2026-01-01T18:00:00Z"]},
    )
    assert daily_response.status_code == 422
    assert "daily partitions" in daily_response.json()["detail"]

    assert (
        api_client.post("/features", json=_feature_payload()).status_code == 201
    )
    assert api_client.post("/labels", json=_label_payload()).status_code == 201
    feature_response = api_client.post(
        "/features/daily_return/versions/1/partitions",
        json={"timestamps": ["2026-01-01T18:00:00Z"]},
    )
    label_response = api_client.post(
        "/labels/direction/versions/1/partitions",
        json={"timestamps": ["2026-01-01T18:00:00Z"]},
    )
    assert feature_response.status_code == 422
    assert label_response.status_code == 422


def test_hourly_dependencies_use_hourly_input_windows(
    api_client: TestClient, clean_postgres: str
) -> None:
    """Plan feature and label inputs by each upstream dataset's cadence."""
    dataset_payload = {
        "name": "prices",
        "version": 1,
        "dataset_type": "market_data",
        "config": {
            "description": "Hourly market prices.",
            "keys": ["symbol"],
            "partition_type": "hourly",
            "symbols": ["AAPL"],
        },
    }
    assert api_client.post("/datasets", json=dataset_payload).status_code == 201
    _seed_created_hourly_dataset_partitions(clean_postgres)

    feature_payload = {
        "name": "daily_return",
        "version": 1,
        "dtype": "double",
        "frequency": "hourly",
        "keys": ["symbol"],
        "datasets": [
            {
                "name": "prices",
                "version": 1,
                "input_window": {"lookback_partitions": 2},
            }
        ],
        "transform": "ml_platform.data_sdk.feature_transforms.ReturnTransform",
        "description": "Hourly return.",
        "parameters": {},
        "validations": [],
    }
    assert api_client.post("/features", json=feature_payload).status_code == 201

    label_payload = {
        "name": "direction",
        "version": 1,
        "dtype": "double",
        "frequency": "hourly",
        "keys": ["symbol"],
        "datasets": [
            {
                "name": "prices",
                "version": 1,
                "input_window": {
                    "lookback_partitions": 1,
                    "lookahead_partitions": 1,
                },
            }
        ],
        "transform": "ml_platform.data_sdk.label_transforms.DirectionTransform",
        "description": "Hourly direction.",
        "parameters": {},
        "validations": [],
    }
    assert api_client.post("/labels", json=label_payload).status_code == 201

    feature_response = api_client.post(
        "/features/daily_return/versions/1/partitions",
        json={"timestamps": ["2026-01-01T02:00:00Z"]},
    )
    label_response = api_client.post(
        "/labels/direction/versions/1/partitions",
        json={"timestamps": ["2026-01-01T01:00:00Z"]},
    )
    assert feature_response.status_code == 202
    assert label_response.status_code == 202


def test_concurrent_dataset_requests_create_one_active_partition_job(
    api_client: TestClient,
    clean_postgres: str,
) -> None:
    """Transactional advisory locks preserve one pending job per dataset partition."""
    assert (
        api_client.post("/datasets", json=_dataset_payload()).status_code == 201
    )
    synchronization = Barrier(2)
    engine = engine_for(clean_postgres)

    def materialize_once():
        synchronization.wait()
        with Session(engine) as session:
            request = DatasetRepository(session).materialize(
                ResourceName("prices"),
                ResourceVersion(1),
                (PartitionTimestamp(datetime(2026, 1, 1, tzinfo=UTC)),),
            )
        assert request is not None
        return request

    with ThreadPoolExecutor(max_workers=2) as executor:
        requests = tuple(
            future.result()
            for future in (
                executor.submit(materialize_once),
                executor.submit(materialize_once),
            )
        )

    assert sum(request.submitted_job is not None for request in requests) == 1
    assert {
        partition.status.value
        for request in requests
        for partition in request.partitions
    } == {"pending"}
    with psycopg.connect(clean_postgres) as connection:
        active_requests = connection.execute(
            """SELECT COUNT(*) FROM partition_update_requests
            WHERE dataset_name = 'prices' AND dataset_version = 1
            AND timestamp = '2026-01-01T00:00:00Z' AND status = 'pending'"""
        ).fetchone()
    assert active_requests == (1,)


def test_feature_and_label_materialization_requests_are_idempotent(
    api_client: TestClient, clean_postgres: str
) -> None:
    """An identical in-flight request reuses the existing partition plan."""
    assert (
        api_client.post("/datasets", json=_dataset_payload()).status_code == 201
    )
    _seed_created_dataset_partition(clean_postgres)
    assert (
        api_client.post("/features", json=_feature_payload()).status_code == 201
    )
    assert api_client.post("/labels", json=_label_payload()).status_code == 201

    payload = {"timestamps": ["2026-01-01T00:00:00Z"]}
    first_feature = api_client.post(
        "/features/daily_return/versions/1/partitions", json=payload
    )
    repeat_feature = api_client.post(
        "/features/daily_return/versions/1/partitions", json=payload
    )
    first_label = api_client.post(
        "/labels/direction/versions/1/partitions", json=payload
    )
    repeat_label = api_client.post(
        "/labels/direction/versions/1/partitions", json=payload
    )

    assert first_feature.status_code == 202
    assert first_label.status_code == 202
    assert repeat_feature.status_code == 202
    assert repeat_label.status_code == 202
    assert repeat_feature.json()["submitted_job"] is None
    assert repeat_label.json()["submitted_job"] is None
    with psycopg.connect(clean_postgres) as connection:
        counts = connection.execute(
            """SELECT type::text, COUNT(*) FROM jobs
            WHERE type IN ('feature_materialization', 'label_materialization')
            GROUP BY type ORDER BY type"""
        ).fetchall()
    assert counts == [
        ("feature_materialization", 1),
        ("label_materialization", 1),
    ]


def test_ensure_current_replaces_derived_partitions_with_new_input_revisions(
    api_client: TestClient, clean_postgres: str
) -> None:
    """Ensure-current replaces outputs when a source partition revision changes."""
    assert (
        api_client.post("/datasets", json=_dataset_payload()).status_code == 201
    )
    initial_dataset_partition_id = _seed_created_dataset_partition(
        clean_postgres
    )
    assert (
        api_client.post("/features", json=_feature_payload()).status_code == 201
    )
    assert api_client.post("/labels", json=_label_payload()).status_code == 201

    request_payload = {"timestamps": ["2026-01-01T00:00:00Z"]}
    initial_feature = api_client.post(
        "/features/daily_return/versions/1/partitions",
        json=request_payload,
    )
    initial_label = api_client.post(
        "/labels/direction/versions/1/partitions",
        json=request_payload,
    )
    assert initial_feature.status_code == 202
    assert initial_label.status_code == 202
    initial_feature_partition_id = initial_feature.json()["partitions"][0][
        "partition_id"
    ]
    initial_label_partition_id = initial_label.json()["partitions"][0][
        "partition_id"
    ]
    _complete_transform_partition(
        api_client,
        clean_postgres,
        resource_type=ResourceType.FEATURE,
        partition_id=initial_feature_partition_id,
        dataset_partition_id=initial_dataset_partition_id,
    )
    _complete_transform_partition(
        api_client,
        clean_postgres,
        resource_type=ResourceType.LABEL,
        partition_id=initial_label_partition_id,
        dataset_partition_id=initial_dataset_partition_id,
    )

    new_dataset_partition_id = _seed_newer_dataset_partition(clean_postgres)
    refreshed_feature = api_client.post(
        "/features/daily_return/versions/1/partitions",
        json=request_payload,
    )
    refreshed_label = api_client.post(
        "/labels/direction/versions/1/partitions",
        json=request_payload,
    )
    assert refreshed_feature.status_code == 202
    assert refreshed_label.status_code == 202
    assert refreshed_feature.json()["submitted_job"] is not None
    assert refreshed_label.json()["submitted_job"] is not None
    assert (
        refreshed_feature.json()["partitions"][0]["partition_id"]
        != initial_feature_partition_id
    )
    assert (
        refreshed_label.json()["partitions"][0]["partition_id"]
        != initial_label_partition_id
    )

    with psycopg.connect(clean_postgres) as connection:
        feature_plan = connection.execute(
            """SELECT dataset_partition_id
            FROM feature_partition_update_request_dataset_dependencies
            WHERE feature_partition_request_id = %s""",
            (UUID(refreshed_feature.json()["partitions"][0]["partition_id"]),),
        ).fetchone()[0]
        label_plan = connection.execute(
            """SELECT dataset_partition_id
            FROM label_partition_update_request_dataset_dependencies
            WHERE label_partition_request_id = %s""",
            (UUID(refreshed_label.json()["partitions"][0]["partition_id"]),),
        ).fetchone()[0]
    assert feature_plan == new_dataset_partition_id
    assert label_plan == new_dataset_partition_id
