"""FastAPI application for the dataset and feature materialization control plane."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path as FilePath
from datetime import datetime, time, timedelta, timezone
import os
import secrets
from typing import Annotated, Any
from urllib.parse import quote
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Path, status
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from psycopg import Connection, errors
from psycopg.rows import dict_row
from psycopg.types.json import Json
from kubernetes import client as kubernetes_client
from kubernetes.client.exceptions import ApiException

from ml_platform.api.database import get_connection
from ml_platform.api.scheduler import JobScheduler
from ml_platform.api.migrations import migrate_database
from ml_platform.api.dtos import (
    DatasetCreateRequest,
    DatasetDescriptionUpdate,
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

UI_DIRECTORY = FilePath(__file__).with_name("ui")
app.mount("/ui/assets", StaticFiles(directory=UI_DIRECTORY), name="ui-assets")


@app.get("/", include_in_schema=False)
def dashboard() -> RedirectResponse:
    """Open the control-plane dashboard."""
    return RedirectResponse(url="/ui")


@app.get("/ui", include_in_schema=False)
@app.get("/ui/{dashboard_path:path}", include_in_schema=False)
def dashboard_ui(dashboard_path: str = "") -> FileResponse:
    """Serve the single-page dashboard at every resource browser path."""
    return FileResponse(UI_DIRECTORY / "index.html")


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


def _job_targets(cursor: Any, job_id: UUID) -> list[dict[str, Any]]:
    """Return each versioned resource whose partitions a job targets."""
    cursor.execute(
        """SELECT 'dataset' AS resource_type, dataset_name AS name, dataset_version AS version
            FROM partition_update_requests WHERE job_id = %s
            GROUP BY dataset_name, dataset_version
            UNION
            SELECT 'feature' AS resource_type, feature_name AS name, feature_version AS version
            FROM feature_partition_update_requests WHERE job_id = %s
            GROUP BY feature_name, feature_version
            ORDER BY resource_type, name, version""",
        (job_id, job_id),
    )
    return list(cursor.fetchall())


def _job_partitions(cursor: Any, job_id: UUID) -> list[dict[str, Any]]:
    """Return the partition requests targeted by one materialization job."""
    cursor.execute(
        """SELECT 'dataset' AS resource_type, dataset_name AS name, dataset_version AS version, timestamp, status
            FROM partition_update_requests WHERE job_id = %s
            UNION ALL
            SELECT 'feature' AS resource_type, feature_name AS name, feature_version AS version, timestamp, status
            FROM feature_partition_update_requests WHERE job_id = %s
            ORDER BY timestamp, resource_type, name, version""",
        (job_id, job_id),
    )
    return list(cursor.fetchall())


def _resource_jobs(cursor: Any, resource_type: str, name: str, version: int) -> list[dict[str, Any]]:
    """Return jobs which targeted one immutable resource version."""
    if resource_type == "dataset":
        cursor.execute(
            """SELECT DISTINCT jobs.id, jobs.type, jobs.status, jobs.created_at, jobs.updated_at
                FROM jobs JOIN partition_update_requests AS request ON request.job_id = jobs.id
                WHERE request.dataset_name = %s AND request.dataset_version = %s
                ORDER BY jobs.created_at DESC""",
            (name, version),
        )
    else:
        cursor.execute(
            """SELECT DISTINCT jobs.id, jobs.type, jobs.status, jobs.created_at, jobs.updated_at
                FROM jobs JOIN feature_partition_update_requests AS request ON request.job_id = jobs.id
                WHERE request.feature_name = %s AND request.feature_version = %s
                ORDER BY jobs.created_at DESC""",
            (name, version),
        )
    return list(cursor.fetchall())


@app.get("/datasets", tags=["datasets"])
def list_datasets(connection: DatabaseConnection, include_deleted: bool = False) -> list[dict[str, Any]]:
    """List dataset definitions with logical partition status counts for the dashboard."""
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute("""SELECT datasets.name, datasets.version, datasets.dataset_type, datasets.config, datasets.created_at, datasets.deleted_at, datasets.deprecated_at,
            COUNT(request.partition_id) AS partition_count,
            COUNT(request.partition_id) FILTER (WHERE request.status = 'created') AS created_partition_count,
            COUNT(request.partition_id) FILTER (WHERE request.status = 'pending') AS pending_partition_count,
            COUNT(request.partition_id) FILTER (WHERE request.status IN ('failed', 'cancelled')) AS failed_partition_count
            FROM datasets LEFT JOIN partition_update_requests request
              ON request.dataset_name = datasets.name AND request.dataset_version = datasets.version
            WHERE (%s OR datasets.deleted_at IS NULL)
            GROUP BY datasets.name, datasets.version, datasets.dataset_type, datasets.config, datasets.created_at, datasets.deleted_at, datasets.deprecated_at
            ORDER BY datasets.name, datasets.version DESC""", (include_deleted,))
        datasets = list(cursor.fetchall())
    for dataset in datasets:
        dataset["id"] = f"{dataset['name']}_v{dataset['version']}"
    return datasets


@app.get("/datasets/{name}/versions/{version}", tags=["datasets"])
def get_dataset(name: ResourceName, version: ResourceVersion, connection: DatabaseConnection) -> dict[str, Any]:
    """Return one dataset and the latest receipt for each logical partition."""
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT name, version, dataset_type, config, created_at, deleted_at, deprecated_at FROM datasets WHERE name = %s AND version = %s", (name, version))
        dataset = cursor.fetchone()
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset version does not exist.")
        cursor.execute("""SELECT DISTINCT ON (request.timestamp) request.partition_id, request.timestamp, request.status,
                request.created_at, request.updated_at, receipt.content_hash, receipt.snapshot_id
            FROM partition_update_requests request LEFT JOIN dataset_partitions receipt ON receipt.partition_id = request.partition_id
            WHERE request.dataset_name = %s AND request.dataset_version = %s
            ORDER BY request.timestamp DESC, request.created_at DESC""", (name, version))
        dataset["partitions"] = list(cursor.fetchall())
        dataset["jobs"] = _resource_jobs(cursor, "dataset", name, version)
    dataset["id"] = f"{dataset['name']}_v{dataset['version']}"
    _attach_minio_locations(dataset, f"market_data/{name}-v{version}")
    return dataset


@app.patch("/datasets/{name}/versions/{version}", tags=["datasets"])
def update_dataset_description(
    name: ResourceName,
    version: ResourceVersion,
    payload: DatasetDescriptionUpdate,
    connection: DatabaseConnection,
) -> dict[str, Any]:
    """Update display-only dataset metadata without changing its materialized definition."""
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """UPDATE datasets SET config = jsonb_set(config, '{description}', to_jsonb(%s::text))
            WHERE name = %s AND version = %s
            RETURNING name, version, dataset_type, config, created_at""",
            (payload.description, name, version),
        )
        dataset = cursor.fetchone()
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset version does not exist.")
    connection.commit()
    dataset["id"] = f"{dataset['name']}_v{dataset['version']}"
    return dataset


@app.delete("/datasets/{name}/versions/{version}", status_code=status.HTTP_204_NO_CONTENT, tags=["datasets"])
def delete_dataset(name: ResourceName, version: ResourceVersion, connection: DatabaseConnection) -> None:
    """Soft-delete a dataset version and cancel its unstarted partition requests."""
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute("""UPDATE datasets SET deleted_at = COALESCE(deleted_at, now())
            WHERE name = %s AND version = %s RETURNING name""", (name, version))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Dataset version does not exist.")
        cursor.execute("""UPDATE partition_update_requests SET status = 'cancelled', updated_at = now()
            WHERE dataset_name = %s AND dataset_version = %s AND status = 'pending'""", (name, version))
        cursor.execute("""UPDATE jobs SET status = 'cancelled', updated_at = now()
            WHERE status = 'submitted' AND id IN (
              SELECT DISTINCT job_id FROM partition_update_requests
              WHERE dataset_name = %s AND dataset_version = %s
            )""", (name, version))
    connection.commit()


@app.post("/datasets", response_model=DatasetResponse, status_code=201, responses=error_responses(DATASET_VALIDATION), tags=["datasets"])
def create_dataset(payload: DatasetCreateRequest, connection: DatabaseConnection) -> DatasetResponse:
    """Register an immutable dataset definition."""
    try:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """INSERT INTO datasets (name, version, dataset_type, config)
                VALUES (%s, %s, %s, %s)
                RETURNING name, version, dataset_type, config, created_at""",
                (payload.name, payload.version, payload.dataset_type.value, Json(payload.config.model_dump(mode="json"))),
            )
            row = cursor.fetchone()
            job_id = _enqueue_dataset_job(cursor, payload.name, payload.version, payload.config.model_dump(mode="json"))
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


@app.get("/features", tags=["features"])
def list_features(connection: DatabaseConnection, include_deleted: bool = False) -> list[dict[str, Any]]:
    """List feature definitions with logical partition status counts for the dashboard."""
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute("""SELECT features.name, features.version, features.dtype, features.frequency, features.feature_key,
            features.description, features.created_at, features.deleted_at, features.deprecated_at, COUNT(request.partition_id) AS partition_count,
            COUNT(request.partition_id) FILTER (WHERE request.status = 'created') AS created_partition_count,
            COUNT(request.partition_id) FILTER (WHERE request.status = 'pending') AS pending_partition_count,
            COUNT(request.partition_id) FILTER (WHERE request.status IN ('failed', 'cancelled')) AS failed_partition_count
            FROM features LEFT JOIN feature_partition_update_requests request
              ON request.feature_name = features.name AND request.feature_version = features.version
            WHERE (%s OR features.deleted_at IS NULL)
            GROUP BY features.name, features.version, features.dtype, features.frequency, features.feature_key,
                features.description, features.created_at, features.deleted_at, features.deprecated_at
            ORDER BY features.name, features.version DESC""", (include_deleted,))
        return list(cursor.fetchall())


@app.get("/features/{name}/versions/{version}", tags=["features"])
def get_feature(name: ResourceName, version: ResourceVersion, connection: DatabaseConnection) -> dict[str, Any]:
    """Return one feature, its dependencies, and latest logical partition receipts."""
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute("""SELECT name, version, dtype, frequency, feature_key, validations, transform, description,
            parameters, created_at, deleted_at, deprecated_at FROM features WHERE name = %s AND version = %s""", (name, version))
        feature = cursor.fetchone()
        if feature is None:
            raise HTTPException(status_code=404, detail="Feature version does not exist.")
        cursor.execute("""SELECT dataset_name AS name, dataset_version AS version, num_lookback_partitions
            FROM feature_dataset_dependencies WHERE feature_name = %s AND feature_version = %s
            ORDER BY dataset_name, dataset_version""", (name, version))
        feature["datasets"] = list(cursor.fetchall())
        cursor.execute("""SELECT DISTINCT ON (request.timestamp) request.partition_id, request.timestamp, request.status,
                request.created_at, request.updated_at, receipt.content_hash, receipt.snapshot_id
            FROM feature_partition_update_requests request LEFT JOIN feature_partitions receipt ON receipt.partition_id = request.partition_id
            WHERE request.feature_name = %s AND request.feature_version = %s
            ORDER BY request.timestamp DESC, request.created_at DESC""", (name, version))
        feature["partitions"] = list(cursor.fetchall())
        feature["jobs"] = _resource_jobs(cursor, "feature", name, version)
    _attach_minio_locations(feature, f"features/{name}-v{version}")
    return feature


@app.delete("/features/{name}/versions/{version}", status_code=status.HTTP_204_NO_CONTENT, tags=["features"])
def delete_feature(name: ResourceName, version: ResourceVersion, connection: DatabaseConnection) -> None:
    """Soft-delete a feature version and cancel its unstarted partition requests."""
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute("""UPDATE features SET deleted_at = COALESCE(deleted_at, now())
            WHERE name = %s AND version = %s RETURNING name""", (name, version))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Feature version does not exist.")
        cursor.execute("""UPDATE feature_partition_update_requests SET status = 'cancelled', updated_at = now()
            WHERE feature_name = %s AND feature_version = %s AND status = 'pending'""", (name, version))
        cursor.execute("""UPDATE jobs SET status = 'cancelled', updated_at = now()
            WHERE status = 'submitted' AND id IN (
              SELECT DISTINCT job_id FROM feature_partition_update_requests
              WHERE feature_name = %s AND feature_version = %s
            )""", (name, version))
    connection.commit()


@app.post("/features", response_model=FeatureResponse, status_code=201, responses=error_responses(FEATURE_VALIDATION), tags=["features"])
def create_feature(payload: FeatureCreateRequest, connection: DatabaseConnection) -> FeatureResponse:
    """Register an immutable feature definition."""
    try:
        with connection.cursor(row_factory=dict_row) as cursor:
            for dependency in payload.datasets:
                cursor.execute("""SELECT deleted_at FROM datasets WHERE name = %s AND version = %s""",
                    (dependency.name, dependency.version))
                dataset = cursor.fetchone()
                if dataset is None or dataset["deleted_at"] is not None:
                    raise HTTPException(status_code=422, detail="Feature datasets must reference active dataset versions.")
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
        cursor.execute("SELECT 1 FROM datasets WHERE name = %s AND version = %s AND deleted_at IS NULL", (name, version))
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
        cursor.execute("SELECT 1 FROM features WHERE name = %s AND version = %s AND deleted_at IS NULL", (name, version))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Feature version does not exist.")
        cursor.execute("""SELECT 1 FROM feature_dataset_dependencies AS dependency
            JOIN datasets ON datasets.name = dependency.dataset_name AND datasets.version = dependency.dataset_version
            WHERE dependency.feature_name = %s AND dependency.feature_version = %s AND datasets.deleted_at IS NOT NULL""", (name, version))
        if cursor.fetchone() is not None:
            raise HTTPException(status_code=409, detail="Feature partition cannot be materialized because an input dataset version is deleted.")

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
        cursor.execute("""SELECT request.callback_token FROM partition_update_requests AS request
            JOIN datasets ON datasets.name = request.dataset_name AND datasets.version = request.dataset_version
            WHERE request.partition_id = %s AND datasets.deleted_at IS NULL""", (partition_id,))
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
        cursor.execute("""SELECT request.callback_token FROM feature_partition_update_requests AS request
            JOIN features ON features.name = request.feature_name AND features.version = request.feature_version
            WHERE request.partition_id = %s AND features.deleted_at IS NULL""", (partition_id,))
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


@app.get("/jobs", tags=["jobs"])
def list_jobs(connection: DatabaseConnection) -> list[dict[str, Any]]:
    """List materialization jobs and their versioned resource targets."""
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT id, type, status, created_at, updated_at FROM jobs ORDER BY created_at DESC")
        jobs = list(cursor.fetchall())
        for job in jobs:
            job["targets"] = _job_targets(cursor, job["id"])
        return jobs


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
    with connection.cursor(row_factory=dict_row) as cursor:
        job["targets"] = _job_targets(cursor, job_id)
        job["partitions"] = _job_partitions(cursor, job_id)
    prefix = "dataset" if job["type"] == "dataset_materialization" else "feature"
    job["spark_application_name"] = f"{prefix}-{job_id}".lower()
    job["headlamp_url"] = _headlamp_resource_url("customresources/sparkapplications.sparkoperator.k8s.io", job["spark_application_name"])
    job["kubernetes_resources"] = _job_kubernetes_resources(job["spark_application_name"])
    return JobResponse.model_validate(job)


def _minio_browser_url(object_path: str) -> str | None:
    """Build a MinIO Console object-browser URL for one warehouse object prefix."""
    console_url = os.environ.get("MINIO_CONSOLE_URL")
    if not console_url:
        return None
    prefix = quote(f"{object_path.rstrip('/')}/", safe="")
    return f"{console_url.rstrip('/')}/browser/warehouse/{prefix}"


def _attach_minio_locations(resource: dict[str, Any], table_path: str) -> None:
    """Attach direct MinIO Console locations for an Iceberg table and its day partitions."""
    resource["minio_table_url"] = _minio_browser_url(table_path)
    for partition in resource.get("partitions", []):
        timestamp = partition["timestamp"]
        partition_day = timestamp.date().isoformat() if isinstance(timestamp, datetime) else str(timestamp)[:10]
        partition["minio_url"] = _minio_browser_url(f"{table_path}/data/timestamp_day={partition_day}")


def _headlamp_resource_url(resource_path: str, resource_name: str) -> str | None:
    """Build a direct Headlamp URL for one resource in the configured namespace."""
    headlamp_url = os.environ.get("HEADLAMP_URL")
    if not headlamp_url:
        return None
    namespace = os.environ.get("KUBERNETES_NAMESPACE", "ml-platform")
    return f"{headlamp_url.rstrip('/')}/c/main/{resource_path}/{namespace}/{resource_name}"


def _job_kubernetes_resources(application_name: str) -> list[dict[str, str]]:
    """Return Headlamp links for the application and any extant Spark driver or executors."""
    application_url = _headlamp_resource_url("customresources/sparkapplications.sparkoperator.k8s.io", application_name)
    resources = [{"resource_type": "Spark Application", "resource_id": application_name, "url": application_url}] if application_url else []
    if not application_url:
        return resources
    namespace = os.environ.get("KUBERNETES_NAMESPACE", "ml-platform")
    try:
        kubernetes_client.CustomObjectsApi().get_namespaced_custom_object(
            "sparkoperator.k8s.io", "v1beta2", namespace, "sparkapplications", application_name
        )
    except ApiException as error:
        if error.status == 404:
            return []
    except Exception:
        return resources
    try:
        pods = kubernetes_client.CoreV1Api().list_namespaced_pod(
            namespace,
            label_selector=f"sparkoperator.k8s.io/app-name={application_name}",
        ).items
    except Exception:
        return resources
    for pod in sorted(pods, key=lambda value: value.metadata.name):
        role = (pod.metadata.labels or {}).get("spark-role")
        if role not in {"driver", "executor"}:
            continue
        url = _headlamp_resource_url("pods", pod.metadata.name)
        if url:
            resources.append({"resource_type": f"{role.title()} Pod", "resource_id": pod.metadata.name, "url": url})
    return resources


def main() -> None:
    """Run the application with Uvicorn for local development."""
    import uvicorn

    uvicorn.run("ml_platform.api.main:app", host="127.0.0.1", port=8000, reload=True)
