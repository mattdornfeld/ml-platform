"""Dataset registration, inspection, deletion, and materialization routes."""

from typing import Annotated

from fastapi import APIRouter, Query, status

from ml_platform.api._dependencies import (
    ApiDependencies,
    DatabaseSession,
    ResourcePathName,
    ResourcePathVersion,
)
from ml_platform.api.domain import (
    DatasetDefinition,
    DatasetSummary,
    MaterializationRequest,
    ResourcePartition,
    ResourceType,
)
from ml_platform.api.repositories import DatasetRepository, JobRepository
from ml_platform.api.routers._openapi import (
    DATASET_VALIDATION,
    PARTITION_VALIDATION,
    error_responses,
)
from ml_platform.api.routers.exceptions import (
    DatasetVersionNotFoundError,
)
from ml_platform.api.services import minio_browser_url, minio_partition_url
from ml_platform.common.api.dtos import (
    DatasetCreateRequest,
    DatasetDescriptionUpdate,
    DatasetResponse,
    DatasetStatus,
    MaterializationJobResponse,
    PartitionMaterializationRequest,
)
from ml_platform.common.api.routes import DatasetRoutes
from ml_platform.common.types import ResourceName, ResourceVersion

router = APIRouter(tags=["datasets"])


def _definition_response(definition: DatasetDefinition) -> dict[str, object]:
    """Adapt a dataset domain model to the existing dashboard response shape."""
    return {
        "name": definition.name,
        "version": definition.version,
        "dataset_type": definition.dataset_type,
        "config": definition.config.model_dump(mode="json"),
        "created_at": definition.created_at,
        "deleted_at": definition.deleted_at,
        "deprecated_at": definition.deprecated_at,
        "id": f"{definition.name}_v{definition.version}",
    }


def _partition_response(
    dependencies: ApiDependencies,
    partition: ResourcePartition,
    table_path: str,
) -> dict[str, object]:
    """Adapt one latest logical partition to the existing HTTP shape."""
    return {
        "partition_id": partition.id,
        "timestamp": partition.timestamp,
        "status": partition.status,
        "created_at": partition.created_at,
        "updated_at": partition.updated_at,
        "content_hash": partition.content_hash,
        "snapshot_id": partition.snapshot_id,
        "minio_url": minio_partition_url(
            dependencies, table_path, partition.timestamp
        ),
    }


def _summary_response(summary: DatasetSummary) -> dict[str, object]:
    """Adapt a dataset summary to the existing dashboard response shape."""
    response = _definition_response(summary.definition)
    response.update(
        {
            "partition_count": summary.partitions.total,
            "created_partition_count": summary.partitions.created,
            "pending_partition_count": summary.partitions.pending,
            "failed_partition_count": summary.partitions.failed,
        }
    )
    return response


def _materialization_response(
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


DEFAULT_DATASET_STATUSES = (
    DatasetStatus.ACTIVE,
    DatasetStatus.DEPRECATED,
)
DatasetStatuses = Annotated[
    tuple[DatasetStatus, ...],
    Query(
        min_length=1,
        description="Lifecycle states included in the dataset catalog response.",
    ),
]


@router.get(DatasetRoutes.LIST.template, tags=["datasets"])
def list_datasets(
    session: DatabaseSession,
    *,
    statuses: DatasetStatuses = DEFAULT_DATASET_STATUSES,
) -> list[dict[str, object]]:
    """List dataset definitions in the requested lifecycle states."""
    return [
        _summary_response(summary)
        for summary in DatasetRepository(session).list(statuses=statuses)
    ]


@router.get(DatasetRoutes.DETAIL.template, tags=["datasets"])
def get_dataset(
    name: ResourcePathName,
    version: ResourcePathVersion,
    dependencies: ApiDependencies,
    session: DatabaseSession,
) -> dict[str, object]:
    """Return one dataset and its latest logical partition receipts."""
    repository = DatasetRepository(session)
    definition = repository.get(name, version)
    if definition is None:
        raise DatasetVersionNotFoundError
    table_path = f"market_data/{name}-v{version}"
    return {
        **_definition_response(definition),
        "minio_table_url": minio_browser_url(dependencies, table_path),
        "partitions": [
            _partition_response(dependencies, partition, table_path)
            for partition in repository.partitions(name, version)
        ],
        "jobs": [
            {
                "id": job.id,
                "type": job.type,
                "status": job.status,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
            }
            for job in JobRepository(session).for_resource(
                ResourceType.DATASET, name, version
            )
        ],
    }


@router.patch(DatasetRoutes.UPDATE.template, tags=["datasets"])
def update_dataset_description(
    name: ResourcePathName,
    version: ResourcePathVersion,
    payload: DatasetDescriptionUpdate,
    session: DatabaseSession,
) -> dict[str, object]:
    """Update display-only dataset metadata without changing its definition."""
    definition = DatasetRepository(session).update_description(
        name, version, payload.description
    )
    if definition is None:
        raise DatasetVersionNotFoundError
    return _definition_response(definition)


@router.delete(
    DatasetRoutes.DELETE.template,
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["datasets"],
)
def delete_dataset(
    name: ResourcePathName,
    version: ResourcePathVersion,
    session: DatabaseSession,
) -> None:
    """Soft-delete a dataset and cancel each unfinished partition job."""
    if not DatasetRepository(session).delete(name, version):
        raise DatasetVersionNotFoundError


@router.post(
    DatasetRoutes.CREATE.template,
    response_model=DatasetResponse,
    status_code=201,
    responses=error_responses(DATASET_VALIDATION),
    tags=["datasets"],
)
def create_dataset(
    payload: DatasetCreateRequest, session: DatabaseSession
) -> DatasetResponse:
    """Register an immutable dataset definition without scheduling work."""
    definition = DatasetRepository(session).create(
        name=ResourceName(payload.name),
        version=ResourceVersion(payload.version),
        config=payload.config,
    )
    return DatasetResponse.model_validate(
        {
            "name": definition.name,
            "version": definition.version,
            "id": f"{definition.name}_v{definition.version}",
            "dataset_type": definition.dataset_type,
            "config": definition.config.model_dump(mode="json"),
            "created_at": definition.created_at,
        }
    )


@router.post(
    DatasetRoutes.MATERIALIZE.template,
    response_model=MaterializationJobResponse,
    status_code=202,
    responses=error_responses(PARTITION_VALIDATION),
    tags=["dataset materialization"],
)
def materialize_dataset_partitions(
    name: ResourcePathName,
    version: ResourcePathVersion,
    payload: PartitionMaterializationRequest,
    session: DatabaseSession,
) -> MaterializationJobResponse:
    """Create missing dataset partitions and join completed or pending requests."""
    materialization = DatasetRepository(session).materialize(
        name,
        version,
        tuple(payload.timestamps),
    )
    if materialization is None:
        raise DatasetVersionNotFoundError
    return _materialization_response(materialization)
