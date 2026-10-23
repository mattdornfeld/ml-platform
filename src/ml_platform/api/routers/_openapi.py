"""OpenAPI error examples shared by control-plane route modules."""

from typing import Any

from ml_platform.common.api.dtos import ErrorResponse, ValidationErrorResponse


def validation_example(
    location: list[str | int],
    message: str,
    error_type: str,
    invalid_input: object,
    context: dict[str, object] | None = None,
) -> dict[str, object]:
    """Create a complete FastAPI validation-error example."""
    detail: dict[str, object] = {
        "loc": location,
        "msg": message,
        "type": error_type,
        "input": invalid_input,
    }
    if context is not None:
        detail["ctx"] = context
    return {"detail": [detail]}


def error_responses(
    examples: dict[str, dict[str, Any]],
) -> dict[int | str, dict[str, Any]]:
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
                    "example": {
                        "detail": "Database-backed materialization services have not been implemented."
                    }
                }
            },
        },
    }


DATASET_VALIDATION = {
    "empty_name": {
        "summary": "Dataset name must not be empty",
        "value": validation_example(
            ["body", "name"],
            "String should have at least 1 character",
            "string_too_short",
            "",
            {"min_length": 1},
        ),
    },
    "non_positive_version": {
        "summary": "Dataset version must be greater than zero",
        "value": validation_example(
            ["body", "version"],
            "Input should be greater than 0",
            "greater_than",
            0,
            {"gt": 0},
        ),
    },
    "missing_config": {
        "summary": "Dataset configuration is required",
        "value": validation_example(
            ["body", "config"],
            "Field required",
            "missing",
            {"name": "market_data", "version": 1},
        ),
    },
    "unexpected_property": {
        "summary": "Unknown properties are rejected",
        "value": validation_example(
            ["body", "unknown"],
            "Extra inputs are not permitted",
            "extra_forbidden",
            "value",
        ),
    },
}

FEATURE_VALIDATION = {
    **DATASET_VALIDATION,
    "invalid_dtype": {
        "summary": "Feature dtype must be double, float, int, or bigint",
        "value": validation_example(
            ["body", "dtype"],
            "Input should be 'double', 'float', 'int' or 'bigint'",
            "literal_error",
            "decimal",
        ),
    },
    "invalid_frequency": {
        "summary": "Feature frequency must be hourly or daily",
        "value": validation_example(
            ["body", "frequency"],
            "Input should be 'hourly' or 'daily'",
            "literal_error",
            "minute",
        ),
    },
    "empty_datasets": {
        "summary": "A feature requires at least one dataset dependency",
        "value": validation_example(
            ["body", "datasets"],
            "List should have at least 1 item after validation, not 0",
            "too_short",
            [],
            {"field_type": "List", "min_length": 1, "actual_length": 0},
        ),
    },
    "invalid_dependency": {
        "summary": "Dependency versions must be positive and lookbacks non-negative",
        "value": validation_example(
            ["body", "datasets", 0, "input_window", "lookback_partitions"],
            "Input should be greater than or equal to 0",
            "greater_than_equal",
            -1,
            {"ge": 0},
        ),
    },
    "empty_transform_or_description": {
        "summary": "Transform and description must not be empty",
        "value": validation_example(
            ["body", "transform"],
            "String should have at least 1 character",
            "string_too_short",
            "",
            {"min_length": 1},
        ),
    },
}

PARTITION_VALIDATION = {
    "non_positive_resource_version": {
        "summary": "Resource version path parameter must be greater than zero",
        "value": validation_example(
            ["path", "version"],
            "Input should be greater than 0",
            "greater_than",
            "0",
            {"gt": 0},
        ),
    },
    "empty_timestamps": {
        "summary": "At least one timestamp is required",
        "value": validation_example(
            ["body", "timestamps"],
            "List should have at least 1 item after validation, not 0",
            "too_short",
            [],
            {"field_type": "List", "min_length": 1, "actual_length": 0},
        ),
    },
    "missing_utc_offset": {
        "summary": "Timestamp must include a UTC offset",
        "value": validation_example(
            ["body", "timestamps"],
            "Value error, Partition timestamps must include a UTC offset.",
            "value_error",
            ["2026-10-20T18:00:00"],
            {"error": {}},
        ),
    },
    "non_utc_offset": {
        "summary": "Timestamp must be UTC",
        "value": validation_example(
            ["body", "timestamps"],
            "Value error, Partition timestamps must be in UTC.",
            "value_error",
            ["2026-10-20T18:00:00+01:00"],
            {"error": {}},
        ),
    },
    "not_hour_aligned": {
        "summary": "Timestamp must be aligned to the start of an hour",
        "value": validation_example(
            ["body", "timestamps"],
            "Value error, Partition timestamps must be aligned to an exact hour.",
            "value_error",
            ["2026-10-20T18:15:47.828Z"],
            {"error": {}},
        ),
    },
    "unexpected_property": DATASET_VALIDATION["unexpected_property"],
}

COMPLETION_VALIDATION = {
    "invalid_partition_id": {
        "summary": "Every partition request ID must be a UUID",
        "value": validation_example(
            ["body", 0, "partition_id"],
            "Input should be a valid UUID, unable to parse string as a UUID",
            "uuid_parsing",
            "not-a-uuid",
        ),
    },
    "empty_batch": {
        "summary": "At least one receipt is required",
        "value": validation_example(
            ["body"], "List should have at least 1 item", "too_short", []
        ),
    },
    "invalid_row_count": {
        "summary": "Row count cannot be negative",
        "value": validation_example(
            ["body", 0, "row_count"],
            "Input should be greater than or equal to 0",
            "greater_than_equal",
            -1,
            {"ge": 0},
        ),
    },
    "missing_schema": {
        "summary": "Committed schema is required",
        "value": validation_example(
            ["body", 0, "iceberg_schema"], "Field required", "missing", {}
        ),
    },
    "empty_content_hash": {
        "summary": "Content hash must not be empty",
        "value": validation_example(
            ["body", 0, "content_hash"],
            "String should have at least 1 character",
            "string_too_short",
            "",
            {"min_length": 1},
        ),
    },
    "unexpected_property": {
        "summary": "Unknown receipt properties are rejected",
        "value": validation_example(
            ["body", 0, "unknown"],
            "Extra inputs are not permitted",
            "extra_forbidden",
            "value",
        ),
    },
}

JOB_VALIDATION = {
    "invalid_job_id": {
        "summary": "Job ID must be a UUID",
        "value": validation_example(
            ["path", "job_id"],
            "Input should be a valid UUID, unable to parse string as a UUID",
            "uuid_parsing",
            "not-a-uuid",
        ),
    }
}
