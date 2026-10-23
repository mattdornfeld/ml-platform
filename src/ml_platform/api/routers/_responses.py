"""Adapters from API domain models to public response DTOs."""

from ml_platform.api.domain import MaterializationRequest
from ml_platform.common.api.dtos import MaterializationJobResponse


def materialization_response(
    request: MaterializationRequest,
) -> MaterializationJobResponse:
    """Adapt a materialization domain result to the public response DTO."""
    submitted_job = (
        None
        if request.submitted_job is None
        else {
            "id": request.submitted_job.id,
            "type": request.submitted_job.type,
            "status": request.submitted_job.status,
            "created_at": request.submitted_job.created_at,
            "updated_at": request.submitted_job.updated_at,
        }
    )
    return MaterializationJobResponse.model_validate(
        {
            "submitted_job": submitted_job,
            "partitions": [
                {
                    "partition_id": partition.id,
                    "job_id": partition.job_id,
                    "timestamp": partition.timestamp,
                    "status": partition.status,
                    "created_at": partition.created_at,
                    "updated_at": partition.updated_at,
                }
                for partition in request.partitions
            ],
        }
    )
