"""Label registration, inspection, deletion, and materialization routes."""

from fastapi import APIRouter, status

from ml_platform.api._dependencies import (
    ApiDependencies,
    DatabaseSession,
    ResourcePathName,
    ResourcePathVersion,
)
from ml_platform.api.domain import TransformDefinition
from ml_platform.api.repositories import LabelRepository
from ml_platform.api.routers._openapi import (
    FEATURE_VALIDATION,
    PARTITION_VALIDATION,
    error_responses,
)
from ml_platform.api.routers._responses import materialization_response
from ml_platform.api.routers.exceptions import LabelVersionNotFoundError
from ml_platform.api.services import minio_browser_url, minio_partition_url
from ml_platform.common.api.dtos import (
    LabelCreateRequest,
    LabelPartitionMaterializationRequest,
    LabelResponse,
    MaterializationJobResponse,
)
from ml_platform.common.api.routes import LabelRoutes

router = APIRouter(tags=["labels"])


def _definition_response(definition: TransformDefinition) -> dict[str, object]:
    """Present persisted transform definition fields shared by each route."""
    return {
        "name": definition.name,
        "version": definition.version,
        "dtype": definition.dtype,
        "frequency": definition.frequency,
        "keys": definition.keys,
        "validations": definition.validations,
        "transform": definition.transform,
        "description": definition.description,
        "parameters": definition.parameters,
        "created_at": definition.created_at,
        "deleted_at": definition.deleted_at,
        "deprecated_at": definition.deprecated_at,
    }


@router.get(LabelRoutes.LIST.template, tags=["labels"])
def list_labels(
    session: DatabaseSession, *, include_deleted: bool = False
) -> list[dict[str, object]]:
    """List label definitions with logical partition status counts."""
    return [
        {
            "name": summary.definition.name,
            "version": summary.definition.version,
            "dtype": summary.definition.dtype,
            "frequency": summary.definition.frequency,
            "keys": summary.definition.keys,
            "description": summary.definition.description,
            "created_at": summary.definition.created_at,
            "deleted_at": summary.definition.deleted_at,
            "deprecated_at": summary.definition.deprecated_at,
            "partition_count": summary.partitions.total,
            "created_partition_count": summary.partitions.created,
            "pending_partition_count": summary.partitions.pending,
            "failed_partition_count": summary.partitions.failed,
        }
        for summary in LabelRepository(session).summaries(
            include_deleted=include_deleted
        )
    ]


@router.get(LabelRoutes.DETAIL.template, tags=["labels"])
def get_label(
    name: ResourcePathName,
    version: ResourcePathVersion,
    dependencies: ApiDependencies,
    session: DatabaseSession,
) -> dict[str, object]:
    """Return one label, its dependencies, and logical partitions."""
    detail = LabelRepository(session).detail(name, version)
    if detail is None:
        raise LabelVersionNotFoundError
    table_path = f"labels/{name}-v{version}"
    return {
        **_definition_response(detail.definition),
        "minio_table_url": minio_browser_url(dependencies, table_path),
        "datasets": [
            {
                "name": dependency.name,
                "version": dependency.version,
                "dataset_type": dependency.dataset_type,
                "input_window": {
                    "lookback_partitions": dependency.lookback_partitions,
                    "lookahead_partitions": dependency.lookahead_partitions,
                },
            }
            for dependency in detail.datasets
        ],
        "partitions": [
            {
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
            for partition in detail.partitions
        ],
        "jobs": [
            {
                "id": job.id,
                "type": job.type,
                "status": job.status,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
            }
            for job in detail.jobs
        ],
    }


@router.post(
    LabelRoutes.CREATE.template,
    response_model=LabelResponse,
    status_code=201,
    responses=error_responses(FEATURE_VALIDATION),
    tags=["labels"],
)
def create_label(
    payload: LabelCreateRequest, session: DatabaseSession
) -> LabelResponse:
    """Register an immutable label definition and active dataset dependencies."""
    definition = LabelRepository(session).create(payload)
    return LabelResponse.model_validate(
        {
            "name": definition.name,
            "version": definition.version,
            "dtype": definition.dtype,
            "frequency": definition.frequency,
            "keys": definition.keys,
            "validations": definition.validations,
            "transform": definition.transform,
            "description": definition.description,
            "parameters": definition.parameters,
            "created_at": definition.created_at,
            "datasets": [
                dependency.model_dump() for dependency in payload.datasets
            ],
        }
    )


@router.delete(
    LabelRoutes.DELETE.template,
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["labels"],
)
def delete_label(
    name: ResourcePathName,
    version: ResourcePathVersion,
    session: DatabaseSession,
) -> None:
    """Soft-delete a label and cancel each unfinished partition job."""
    if not LabelRepository(session).delete(name, version):
        raise LabelVersionNotFoundError


@router.post(
    LabelRoutes.MATERIALIZE.template,
    response_model=MaterializationJobResponse,
    status_code=202,
    responses=error_responses(PARTITION_VALIDATION),
    tags=["label materialization"],
)
def materialize_label_partitions(
    name: ResourcePathName,
    version: ResourcePathVersion,
    payload: LabelPartitionMaterializationRequest,
    session: DatabaseSession,
) -> MaterializationJobResponse:
    """Request label partitions from the persistence repository."""
    return materialization_response(
        LabelRepository(session).materialize(name, version, payload)
    )
