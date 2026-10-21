"""FastAPI application for the dataset and feature materialization control plane."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, time, timedelta, timezone
import os
import secrets
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Path, status
from psycopg import Connection, errors
from psycopg.rows import dict_row
from psycopg.types.json import Json

from ml_platform.api.database import get_connection
from ml_platform.api.scheduler import JobScheduler
from ml_platform.api.migrations import migrate_database
from ml_platform.api.dtos import (
    DatasetCreateRequest,
    DatasetResponse,
    ErrorResponse,
    FeatureCreateRequest,
    FeatureResponse,
    FeatureMaterializationMode,
    FeaturePartitionCompletionRequest,
    FeaturePartitionMaterializationRequest,
    JobResponse,
    MaterializationJobResponse,
    PartitionCompletionRequest,
    PartitionMaterializationRequest,
    PartitionResultResponse,
    ValidationErrorResponse,
)

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Populate the PostgreSQL schema before accepting API traffic."""
    migrate_database()
    stop = asyncio.Event()
    scheduler = JobScheduler(os.environ["DATABASE_URL"])
    scheduler_task = asyncio.create_task(scheduler.run(stop))
    try:
        yield
    finally:
        stop.set()
        await scheduler_task


app = FastAPI(
    title="ML Platform Materialization API",
    version="0.1.0",
    summary="Control plane for Iceberg-backed dataset and feature materialization.",
    lifespan=lifespan,
)

ResourceName = Annotated[str, Path(min_length=1)]
ResourceVersion = Annotated[int, Path(gt=0)]
_NOT_IMPLEMENTED_DETAIL = "Database-backed materialization services have not been implemented."
_NOT_IMPLEMENTED = HTTPException(status_code=501, detail=_NOT_IMPLEMENTED_DETAIL)
DatabaseConnection = Annotated[Connection, Depends(get_connection)]


def validation_example(
    location: list[str | int],
    message: str,
    error_type: str,
    invalid_input: Any,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a complete FastAPI validation-error example."""
    detail: dict[str, Any] = {
        "loc": location,
        "msg": message,
        "type": error_type,
        "input": invalid_input,
    }
    if context is not None:
        detail["ctx"] = context
    return {"detail": [detail]}


def error_responses(examples: dict[str, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Document every currently possible error response for one route."""
    return {
        422: {
            "model": ValidationErrorResponse,
            "description": "Request validation failed.",
            "content": {"application/json": {"examples": examples}},
        },
        501: {
            "model": ErrorResponse,
            "description": "The database-backed service is not implemented.",
            "content": {
                "application/json": {
                    "examples": {
                        "service_not_implemented": {
                            "summary": "Database service not implemented",
                            "value": {"detail": _NOT_IMPLEMENTED_DETAIL},
                        }
                    }
                }
            },
        },
    }


DATASET_VALIDATION = {
    "empty_name": {
        "summary": "Dataset name must not be empty",
        "value": validation_example(
            ["body", "name"], "String should have at least 1 character", "string_too_short", "", {"min_length": 1}
        ),
    },
    "non_positive_version": {
        "summary": "Dataset version must be greater than zero",
        "value": validation_example(
            ["body", "version"], "Input should be greater than 0", "greater_than", 0, {"gt": 0}
        ),
    },
    "missing_config": {
        "summary": "Dataset configuration is required",
        "value": validation_example(["body", "config"], "Field required", "missing", {"name": "market_data", "version": 1}),
    },
    "unexpected_property": {
        "summary": "Unknown properties are rejected",
        "value": validation_example(["body", "unknown"], "Extra inputs are not permitted", "extra_forbidden", "value"),
    },
}

FEATURE_VALIDATION = {
    **DATASET_VALIDATION,
    "invalid_dtype": {
        "summary": "Feature dtype must be double, float, int, or bigint",
        "value": validation_example(
            ["body", "dtype"], "Input should be 'double', 'float', 'int' or 'bigint'", "literal_error", "decimal"
        ),
    },
    "invalid_frequency": {
        "summary": "Feature frequency must be hourly or daily",
        "value": validation_example(
            ["body", "frequency"], "Input should be 'hourly' or 'daily'", "literal_error", "minute"
        ),
    },
    "empty_datasets": {
        "summary": "A feature requires at least one dataset dependency",
        "value": validation_example(
            ["body", "datasets"], "List should have at least 1 item after validation, not 0", "too_short", [],
            {"field_type": "List", "min_length": 1, "actual_length": 0},
        ),
    },
    "invalid_dependency": {
        "summary": "Dependency versions must be positive and lookbacks non-negative",
        "value": validation_example(
            ["body", "datasets", 0, "num_lookback_partitions"], "Input should be greater than or equal to 0", "greater_than_equal", -1, {"ge": 0}
        ),
    },
    "empty_transform_or_description": {
        "summary": "Transform and description must not be empty",
        "value": validation_example(
            ["body", "transform"], "String should have at least 1 character", "string_too_short", "", {"min_length": 1}
        ),
    },
}

PARTITION_VALIDATION = {
    "non_positive_resource_version": {
        "summary": "Resource version path parameter must be greater than zero",
        "value": validation_example(["path", "version"], "Input should be greater than 0", "greater_than", "0", {"gt": 0}),
    },
    "empty_timestamps": {
        "summary": "At least one timestamp is required",
        "value": validation_example(
            ["body", "timestamps"], "List should have at least 1 item after validation, not 0", "too_short", [],
            {"field_type": "List", "min_length": 1, "actual_length": 0},
        ),
    },
    "missing_utc_offset": {
        "summary": "Timestamp must include a UTC offset",
        "value": validation_example(
            ["body", "timestamps"], "Value error, Partition timestamps must include a UTC offset.", "value_error", ["2026-10-20T18:00:00"], {"error": {}}
        ),
    },
    "non_utc_offset": {
        "summary": "Timestamp must be UTC",
        "value": validation_example(
            ["body", "timestamps"], "Value error, Partition timestamps must be in UTC.", "value_error", ["2026-10-20T18:00:00+01:00"], {"error": {}}
        ),
    },
    "not_hour_aligned": {
        "summary": "Timestamp must be aligned to the start of an hour",
        "value": validation_example(
            ["body", "timestamps"], "Value error, Partition timestamps must be aligned to an exact hour.", "value_error", ["2026-10-20T18:15:47.828Z"], {"error": {}}
        ),
    },
    "unexpected_property": DATASET_VALIDATION["unexpected_property"],
}

COMPLETION_VALIDATION = {
    "invalid_partition_id": {
        "summary": "Partition request ID must be a UUID",
        "value": validation_example(
            ["path", "partition_id"], "Input should be a valid UUID, unable to parse string as a UUID", "uuid_parsing", "not-a-uuid"
        ),
    },
    "empty_content_hash": {
        "summary": "Content hash must not be empty",
        "value": validation_example(
            ["body", "content_hash"], "String should have at least 1 character", "string_too_short", "", {"min_length": 1}
        ),
    },
    "unexpected_property": DATASET_VALIDATION["unexpected_property"],
}

JOB_VALIDATION = {
    "invalid_job_id": {
        "summary": "Job ID must be a UUID",
        "value": validation_example(
            ["path", "job_id"], "Input should be a valid UUID, unable to parse string as a UUID", "uuid_parsing", "not-a-uuid"
        ),
    }
}


@app.post("/datasets", response_model=DatasetResponse, status_code=201, responses=error_responses(DATASET_VALIDATION), tags=["datasets"])
def create_dataset(payload: DatasetCreateRequest, connection: DatabaseConnection) -> DatasetResponse:
    """Register an immutable dataset definition."""
    try:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """INSERT INTO datasets (name, version, dataset_type, config)
                VALUES (%s, %s, %s, %s)
                RETURNING name, version, dataset_type, config, created_at""",
                (payload.name, payload.version, payload.dataset_type.value, Json(payload.config)),
            )
            row = cursor.fetchone()
            job_id = _enqueue_dataset_job(cursor, payload.name, payload.version, payload.config)
        connection.commit()
    except errors.UniqueViolation as error:
        raise HTTPException(status_code=409, detail="Dataset name and version already exist.") from error
    assert row is not None
    return DatasetResponse.model_validate({**row, "job_id": job_id})


def _enqueue_dataset_job(cursor: Any, name: str, version: int, config: dict[str, Any]) -> UUID:
    """Create the authoritative queue record and daily partition requests."""
    start = datetime.combine(datetime.fromisoformat(config["time_range"]["start_date"]).date(), time.min, timezone.utc)
    end = datetime.combine(datetime.fromisoformat(config["time_range"]["end_date"]).date(), time.min, timezone.utc)
    cursor.execute("INSERT INTO jobs (type, status) VALUES ('dataset_materialization', 'submitted') RETURNING id")
    job_id = cursor.fetchone()["id"]
    timestamp = start
    while timestamp < end:
        cursor.execute("""INSERT INTO partition_update_requests (job_id, dataset_name, dataset_version, timestamp, status, callback_token)
            VALUES (%s, %s, %s, %s, 'pending', %s)""", (job_id, name, version, timestamp, secrets.token_urlsafe(32)))
        timestamp += timedelta(days=1)
    return job_id


@app.post("/features", response_model=FeatureResponse, status_code=201, responses=error_responses(FEATURE_VALIDATION), tags=["features"])
def create_feature(payload: FeatureCreateRequest, connection: DatabaseConnection) -> FeatureResponse:
    """Register an immutable feature definition."""
    try:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("""INSERT INTO features
                (name, version, dtype, frequency, feature_key, validations, transform, description, parameters)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING name, version, dtype, frequency, feature_key, validations, transform, description, parameters, created_at""",
                (payload.name, payload.version, payload.dtype, payload.frequency, payload.feature_key.value,
                 Json(payload.validations.model_dump()), payload.transform, payload.description, Json(payload.parameters)))
            row = cursor.fetchone()
            for dependency in payload.datasets:
                cursor.execute("""INSERT INTO feature_dataset_dependencies
                    (feature_name, feature_version, dataset_name, dataset_version, num_lookback_partitions)
                    VALUES (%s, %s, %s, %s, %s)""",
                    (payload.name, payload.version, dependency.name, dependency.version, dependency.num_lookback_partitions))
        connection.commit()
    except errors.ForeignKeyViolation as error:
        raise HTTPException(status_code=422, detail="Feature datasets must reference registered dataset versions.") from error
    except errors.UniqueViolation as error:
        raise HTTPException(status_code=409, detail="Feature name and version already exist.") from error
    assert row is not None
    return FeatureResponse.model_validate({**row, "datasets": [item.model_dump() for item in payload.datasets]})


@app.post("/datasets/{name}/versions/{version}/partitions", response_model=MaterializationJobResponse, status_code=202, responses=error_responses(PARTITION_VALIDATION), tags=["dataset materialization"])
def materialize_dataset_partitions(name: ResourceName, version: ResourceVersion, payload: PartitionMaterializationRequest, connection: DatabaseConnection) -> MaterializationJobResponse:
    """Create missing dataset partitions and join any completed or pending request."""
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT 1 FROM datasets WHERE name = %s AND version = %s", (name, version))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Dataset version does not exist.")
        existing: list[dict[str, Any]] = []
        missing: list[datetime] = []
        for timestamp in sorted(payload.timestamps):
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"dataset:{name}:{version}:{timestamp.isoformat()}",),
            )
        for timestamp in payload.timestamps:
            cursor.execute(
                """SELECT partition_id, job_id, timestamp, status, created_at, updated_at
                   FROM partition_update_requests
                   WHERE dataset_name = %s AND dataset_version = %s AND timestamp = %s
                     AND status = 'created'
                   ORDER BY created_at DESC LIMIT 1""",
                (name, version, timestamp),
            )
            completed = cursor.fetchone()
            if completed is not None:
                existing.append(completed)
                continue
            cursor.execute(
                """SELECT partition_id, job_id, timestamp, status, created_at, updated_at
                   FROM partition_update_requests
                   WHERE dataset_name = %s AND dataset_version = %s AND timestamp = %s
                     AND status = 'pending'
                   ORDER BY created_at DESC LIMIT 1""",
                (name, version, timestamp),
            )
            active = cursor.fetchone()
            if active is not None:
                existing.append(active)
                continue
            missing.append(timestamp)
        job: dict[str, Any] | None = None
        created: list[dict[str, Any]] = []
        if missing:
            cursor.execute("""INSERT INTO jobs (type, status) VALUES ('dataset_materialization', 'submitted')
                RETURNING id, type, status, created_at, updated_at""")
            job = cursor.fetchone()
            assert job is not None
            for timestamp in missing:
                cursor.execute(
                    """INSERT INTO partition_update_requests
                    (job_id, dataset_name, dataset_version, timestamp, status, callback_token)
                    VALUES (%s, %s, %s, %s, 'pending', %s)
                    RETURNING partition_id, job_id, timestamp, status, created_at, updated_at""",
                    (job["id"], name, version, timestamp, secrets.token_urlsafe(32)),
                )
                request = cursor.fetchone()
                assert request is not None
                created.append(request)
    connection.commit()
    return MaterializationJobResponse.model_validate({"job": job, "partitions": [*existing, *created]})


def _plan_feature_inputs(cursor: Any, name: str, version: int, timestamp: datetime) -> list[UUID]:
    """Resolve the exact successful dataset revisions required by one output partition."""
    cursor.execute(
        """SELECT dataset_name, dataset_version, num_lookback_partitions
        FROM feature_dataset_dependencies
        WHERE feature_name = %s AND feature_version = %s
        ORDER BY dataset_name, dataset_version""",
        (name, version),
    )
    dependencies = cursor.fetchall()
    planned: list[UUID] = []
    for dependency in dependencies:
        start = timestamp - timedelta(days=dependency["num_lookback_partitions"])
        expected = {start + timedelta(days=offset) for offset in range(dependency["num_lookback_partitions"] + 1)}
        cursor.execute(
            """SELECT DISTINCT ON (request.timestamp)
                request.timestamp, partition.partition_id
            FROM partition_update_requests AS request
            JOIN dataset_partitions AS partition ON partition.partition_id = request.partition_id
            WHERE request.dataset_name = %s
              AND request.dataset_version = %s
              AND request.status = 'created'
              AND request.timestamp >= %s AND request.timestamp <= %s
            ORDER BY request.timestamp, partition.created_at DESC, request.created_at DESC""",
            (dependency["dataset_name"], dependency["dataset_version"], start, timestamp),
        )
        revisions = {row["timestamp"]: row["partition_id"] for row in cursor.fetchall()}
        missing = sorted(item.isoformat() for item in expected.difference(revisions))
        if missing:
            raise HTTPException(
                status_code=409,
                detail=("Feature partition cannot be materialized because its required "
                        f"{dependency['dataset_name']}:{dependency['dataset_version']} input partitions are missing: {missing}"),
            )
        planned.extend(revisions[item] for item in sorted(expected))
    return planned


def _completed_feature_request(cursor: Any, name: str, version: int, timestamp: datetime) -> dict[str, Any] | None:
    """Fetch the most recent completed revision of one logical feature partition."""
    cursor.execute(
        """SELECT request.partition_id, request.job_id, request.timestamp, request.status,
                  request.created_at, request.updated_at
           FROM feature_partition_update_requests AS request
           JOIN feature_partitions AS partition ON partition.partition_id = request.partition_id
           WHERE request.feature_name = %s AND request.feature_version = %s
             AND request.timestamp = %s AND request.status = 'created'
           ORDER BY partition.created_at DESC, request.created_at DESC
           LIMIT 1""",
        (name, version, timestamp),
    )
    return cursor.fetchone()


def _feature_plan_ids(cursor: Any, request_id: UUID, *, completed: bool) -> set[UUID]:
    """Read an immutable planned or recorded parent-revision set."""
    table = "feature_partition_dataset_dependencies" if completed else "feature_partition_update_request_dataset_dependencies"
    column = "feature_partition_id" if completed else "feature_partition_request_id"
    cursor.execute(f"SELECT dataset_partition_id FROM {table} WHERE {column} = %s", (request_id,))
    return {row["dataset_partition_id"] for row in cursor.fetchall()}


@app.post("/features/{name}/versions/{version}/partitions", response_model=MaterializationJobResponse, status_code=202, responses=error_responses(PARTITION_VALIDATION), tags=["feature materialization"])
def materialize_feature_partitions(name: ResourceName, version: ResourceVersion, payload: FeaturePartitionMaterializationRequest, connection: DatabaseConnection) -> MaterializationJobResponse:
    """Create only stale/missing feature partitions after pinning their input revisions."""
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT 1 FROM features WHERE name = %s AND version = %s", (name, version))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Feature version does not exist.")

        existing: list[dict[str, Any]] = []
        planned_by_timestamp: dict[datetime, list[UUID]] = {}
        for timestamp in sorted(payload.timestamps):
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"feature:{name}:{version}:{timestamp.isoformat()}",),
            )
        for timestamp in payload.timestamps:
            plan = _plan_feature_inputs(cursor, name, version, timestamp)
            completed = _completed_feature_request(cursor, name, version, timestamp)
            if completed is not None:
                recorded = _feature_plan_ids(cursor, completed["partition_id"], completed=True)
                is_current = recorded == set(plan) and bool(recorded)
                if payload.mode is FeatureMaterializationMode.CREATE_ONLY or (
                    payload.mode is FeatureMaterializationMode.ENSURE_CURRENT and is_current
                ):
                    existing.append(completed)
                    continue
            cursor.execute(
                """SELECT partition_id, job_id, timestamp, status, created_at, updated_at
                   FROM feature_partition_update_requests
                   WHERE feature_name = %s AND feature_version = %s AND timestamp = %s
                     AND status = 'pending'
                   ORDER BY created_at DESC LIMIT 1""",
                (name, version, timestamp),
            )
            active = cursor.fetchone()
            if active is not None:
                active_plan = _feature_plan_ids(cursor, active["partition_id"], completed=False)
                if active_plan == set(plan):
                    existing.append(active)
                    continue
                raise HTTPException(status_code=409, detail="Feature partition is already being materialized against a different input revision plan.")
            planned_by_timestamp[timestamp] = plan

        job: dict[str, Any] | None = None
        created: list[dict[str, Any]] = []
        if planned_by_timestamp:
            cursor.execute("""INSERT INTO jobs (type, status) VALUES ('feature_materialization', 'submitted')
                RETURNING id, type, status, created_at, updated_at""")
            job = cursor.fetchone()
            assert job is not None
            for timestamp, plan in planned_by_timestamp.items():
                cursor.execute("""INSERT INTO feature_partition_update_requests
                    (job_id, feature_name, feature_version, timestamp, status, callback_token)
                    VALUES (%s, %s, %s, %s, 'pending', %s)
                    RETURNING partition_id, job_id, timestamp, status, created_at, updated_at""",
                    (job["id"], name, version, timestamp, secrets.token_urlsafe(32)))
                request = cursor.fetchone()
                assert request is not None
                for dataset_partition_id in plan:
                    cursor.execute("""INSERT INTO feature_partition_update_request_dataset_dependencies
                        (feature_partition_request_id, dataset_partition_id) VALUES (%s, %s)""",
                        (request["partition_id"], dataset_partition_id))
                created.append(request)
    connection.commit()
    return MaterializationJobResponse.model_validate({"job": job, "partitions": [*existing, *created]})


@app.post("/dataset-partition-update-requests/{partition_id}/complete", response_model=PartitionResultResponse, responses=error_responses(COMPLETION_VALIDATION), tags=["dataset materialization"])
def complete_dataset_partition(partition_id: UUID, payload: PartitionCompletionRequest, callback_token: Annotated[str, Header()], connection: DatabaseConnection) -> PartitionResultResponse:
    """Record a successful dataset Iceberg write authenticated by its callback token."""
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT callback_token FROM partition_update_requests WHERE partition_id = %s", (partition_id,))
        request = cursor.fetchone()
        if request is None or not secrets.compare_digest(request["callback_token"], callback_token):
            raise HTTPException(status_code=403, detail="Invalid partition callback token.")
        cursor.execute("""INSERT INTO dataset_partitions (partition_id, content_hash, snapshot_id)
            VALUES (%s, %s, %s) ON CONFLICT (partition_id) DO NOTHING
            RETURNING partition_id, content_hash, snapshot_id, created_at""", (partition_id, payload.content_hash, payload.snapshot_id))
        result = cursor.fetchone()
        if result is None:
            cursor.execute("SELECT partition_id, content_hash, snapshot_id, created_at FROM dataset_partitions WHERE partition_id = %s", (partition_id,))
            result = cursor.fetchone()
        cursor.execute("UPDATE partition_update_requests SET status = 'created', updated_at = now() WHERE partition_id = %s RETURNING job_id", (partition_id,))
        job = cursor.fetchone()
        assert job is not None
        cursor.execute("""UPDATE jobs SET status = 'succeeded', updated_at = now()
            WHERE id = %s AND status = 'submitted'
              AND NOT EXISTS (SELECT 1 FROM partition_update_requests WHERE job_id = %s AND status <> 'created')""", (job["job_id"], job["job_id"]))
    connection.commit()
    return PartitionResultResponse.model_validate(result)


@app.post("/feature-partition-update-requests/{partition_id}/complete", response_model=PartitionResultResponse, responses=error_responses(COMPLETION_VALIDATION), tags=["feature materialization"])
def complete_feature_partition(partition_id: UUID, payload: FeaturePartitionCompletionRequest, callback_token: Annotated[str, Header()], connection: DatabaseConnection) -> PartitionResultResponse:
    """Record a successful feature Iceberg write and its partition receipt."""
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT callback_token FROM feature_partition_update_requests WHERE partition_id = %s", (partition_id,))
        request = cursor.fetchone()
        if request is None or not secrets.compare_digest(request["callback_token"], callback_token):
            raise HTTPException(status_code=403, detail="Invalid partition callback token.")
        planned = _feature_plan_ids(cursor, partition_id, completed=False)
        if planned != set(payload.dataset_partition_ids):
            raise HTTPException(status_code=409, detail="Feature callback dataset revisions do not match the immutable input plan.")
        cursor.execute("""INSERT INTO feature_partitions (partition_id, content_hash, snapshot_id)
            VALUES (%s, %s, %s) ON CONFLICT (partition_id) DO NOTHING
            RETURNING partition_id, content_hash, snapshot_id, created_at""", (partition_id, payload.content_hash, payload.snapshot_id))
        result = cursor.fetchone()
        if result is None:
            cursor.execute("SELECT partition_id, content_hash, snapshot_id, created_at FROM feature_partitions WHERE partition_id = %s", (partition_id,))
            result = cursor.fetchone()
        cursor.execute("""INSERT INTO feature_partition_dataset_dependencies (feature_partition_id, dataset_partition_id)
            SELECT %s, dataset_partition_id
            FROM feature_partition_update_request_dataset_dependencies
            WHERE feature_partition_request_id = %s
            ON CONFLICT DO NOTHING""", (partition_id, partition_id))
        cursor.execute("UPDATE feature_partition_update_requests SET status = 'created', updated_at = now() WHERE partition_id = %s RETURNING job_id", (partition_id,))
        job_id = cursor.fetchone()["job_id"]
        cursor.execute("""UPDATE jobs SET status = 'succeeded', updated_at = now() WHERE id = %s
            AND status = 'submitted' AND NOT EXISTS (SELECT 1 FROM feature_partition_update_requests WHERE job_id = %s AND status <> 'created')""", (job_id, job_id))
    connection.commit()
    return PartitionResultResponse.model_validate(result)


@app.get("/jobs/{job_id}", response_model=JobResponse, responses=error_responses(JOB_VALIDATION), tags=["jobs"])
def get_job(job_id: UUID, connection: DatabaseConnection) -> JobResponse:
    """Return aggregate status for one materialization job."""
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "SELECT id, type, status, created_at, updated_at FROM jobs WHERE id = %s",
            (job_id,),
        )
        job = cursor.fetchone()
    if job is None:
        raise HTTPException(status_code=404, detail="Materialization job does not exist.")
    return JobResponse.model_validate(job)


def main() -> None:
    """Run the application with Uvicorn for local development."""
    import uvicorn

    uvicorn.run("ml_platform.api.main:app", host="127.0.0.1", port=8000, reload=True)
