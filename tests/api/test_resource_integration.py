"""PostgreSQL-backed API tests for resource and materialization workflows."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import psycopg
from fastapi.testclient import TestClient

from ml_platform.api.repositories import SchedulerRepository
from ml_platform.common.types import JobId


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
                "dataset_type": "market_data",
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
                "dataset_type": "market_data",
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


def _seed_created_dataset_partition(database_url: str) -> None:
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
            VALUES (%s, 'parent-hash', 1, '{\"timestamp\": \"timestamp\"}'::jsonb, 1)""",
            (partition_id,),
        )


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
