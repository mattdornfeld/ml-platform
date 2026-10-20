"""FastAPI application for the dataset and feature materialization control plane."""

from typing import Annotated, Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Path, status

from api.dtos import (
    DatasetCreateRequest,
    DatasetResponse,
    ErrorResponse,
    FeatureCreateRequest,
    FeatureResponse,
    JobResponse,
    MaterializationJobResponse,
    PartitionCompletionRequest,
    PartitionMaterializationRequest,
    PartitionResultResponse,
    ValidationErrorResponse,
)

app = FastAPI(
    title="PyTorch Tutorial Materialization API",
    version="0.1.0",
    summary="Control plane for Iceberg-backed dataset and feature materialization.",
)

ResourceName = Annotated[str, Path(min_length=1)]
ResourceVersion = Annotated[int, Path(gt=0)]
_NOT_IMPLEMENTED_DETAIL = "Database-backed materialization services have not been implemented."
_NOT_IMPLEMENTED = HTTPException(status_code=501, detail=_NOT_IMPLEMENTED_DETAIL)


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
def create_dataset(payload: DatasetCreateRequest) -> DatasetResponse:
    """Register an immutable dataset definition."""
    raise _NOT_IMPLEMENTED


@app.post("/features", response_model=FeatureResponse, status_code=201, responses=error_responses(FEATURE_VALIDATION), tags=["features"])
def create_feature(payload: FeatureCreateRequest) -> FeatureResponse:
    """Register an immutable feature definition."""
    raise _NOT_IMPLEMENTED


@app.post("/datasets/{name}/versions/{version}/partitions", response_model=MaterializationJobResponse, status_code=202, responses=error_responses(PARTITION_VALIDATION), tags=["dataset materialization"])
def materialize_dataset_partitions(name: ResourceName, version: ResourceVersion, payload: PartitionMaterializationRequest) -> MaterializationJobResponse:
    """Create a dataset materialization job and its partition update requests."""
    raise _NOT_IMPLEMENTED


@app.post("/features/{name}/versions/{version}/partitions", response_model=MaterializationJobResponse, status_code=202, responses=error_responses(PARTITION_VALIDATION), tags=["feature materialization"])
def materialize_feature_partitions(name: ResourceName, version: ResourceVersion, payload: PartitionMaterializationRequest) -> MaterializationJobResponse:
    """Create a feature materialization job and its partition update requests."""
    raise _NOT_IMPLEMENTED


@app.post("/dataset-partition-update-requests/{partition_id}/complete", response_model=PartitionResultResponse, responses=error_responses(COMPLETION_VALIDATION), tags=["dataset materialization"])
def complete_dataset_partition(partition_id: UUID, payload: PartitionCompletionRequest) -> PartitionResultResponse:
    """Record a successful dataset Iceberg write and its partition receipt."""
    raise _NOT_IMPLEMENTED


@app.post("/feature-partition-update-requests/{partition_id}/complete", response_model=PartitionResultResponse, responses=error_responses(COMPLETION_VALIDATION), tags=["feature materialization"])
def complete_feature_partition(partition_id: UUID, payload: PartitionCompletionRequest) -> PartitionResultResponse:
    """Record a successful feature Iceberg write and its partition receipt."""
    raise _NOT_IMPLEMENTED


@app.get("/jobs/{job_id}", response_model=JobResponse, responses=error_responses(JOB_VALIDATION), tags=["jobs"])
def get_job(job_id: UUID) -> JobResponse:
    """Return aggregate status for one materialization job."""
    raise _NOT_IMPLEMENTED


def main() -> None:
    """Run the application with Uvicorn for local development."""
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
