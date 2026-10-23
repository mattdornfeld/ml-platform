"""Materialization job inspection routes."""

from fastapi import APIRouter, HTTPException

from ml_platform.api._dependencies import DatabaseSession, JobPathId
from ml_platform.api.repositories import JobRepository
from ml_platform.api.routers._openapi import JOB_VALIDATION, error_responses
from ml_platform.api.services import (
    headlamp_resource_url,
    job_kubernetes_resources,
)
from ml_platform.common.api.dtos import JobResponse
from ml_platform.common.api.routes import JobRoutes
from ml_platform.common.types import JobId

router = APIRouter(tags=["jobs"])


def _target_response(target: object) -> dict[str, object]:
    """Adapt one repository domain target to its HTTP representation."""
    from ml_platform.api.domain import JobTarget

    match target:
        case JobTarget():
            return {
                "resource_type": target.resource_type,
                "name": target.name,
                "version": target.version,
            }
        case _ as unsupported_target:
            raise TypeError(f"Unsupported job target {unsupported_target!r}.")


def _partition_response(partition: object) -> dict[str, object]:
    """Adapt one repository domain partition to its HTTP representation."""
    from ml_platform.api.domain import JobPartition

    match partition:
        case JobPartition():
            return {
                "resource_type": partition.resource_type,
                "name": partition.name,
                "version": partition.version,
                "timestamp": partition.timestamp,
                "status": partition.status,
            }
        case _ as unsupported_partition:
            raise TypeError(
                f"Unsupported job partition {unsupported_partition!r}."
            )


@router.get(JobRoutes.LIST.template, tags=["jobs"])
def list_jobs(session: DatabaseSession) -> list[dict[str, object]]:
    """List materialization jobs and their versioned resource targets."""
    repository = JobRepository(session)
    return [
        {
            "id": job.id,
            "type": job.type,
            "status": job.status,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "targets": [
                _target_response(target)
                for target in repository.targets(JobId(job.id))
            ],
        }
        for job in repository.list_jobs()
    ]


@router.get(
    JobRoutes.DETAIL.template,
    response_model=JobResponse,
    responses=error_responses(JOB_VALIDATION),
    tags=["jobs"],
)
def get_job(job_id: JobPathId, session: DatabaseSession) -> JobResponse:
    """Return aggregate status for one materialization job."""
    repository = JobRepository(session)
    job = repository.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail="Materialization job does not exist."
        )
    prefix = job.type.removesuffix("_materialization")
    application_name = f"{prefix}-{job_id}".lower()
    return JobResponse.model_validate(
        {
            "id": job.id,
            "type": job.type,
            "status": job.status,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "targets": [
                _target_response(target)
                for target in repository.targets(job_id)
            ],
            "partitions": [
                _partition_response(partition)
                for partition in repository.partitions(job_id)
            ],
            "spark_application_name": application_name,
            "headlamp_url": headlamp_resource_url(
                "customresources/sparkapplications.sparkoperator.k8s.io",
                application_name,
            ),
            "kubernetes_resources": job_kubernetes_resources(application_name),
        }
    )
