"""Feature registration, inspection, deletion, and materialization routes."""

from fastapi import APIRouter, status

from ml_platform.api._dependencies import (
    ApiDependencies,
    DatabaseSession,
    ResourcePathName,
    ResourcePathVersion,
)
from ml_platform.api.domain import TransformDefinition
from ml_platform.api.repositories import FeatureRepository
from ml_platform.api.routers._openapi import (
    FEATURE_VALIDATION,
    PARTITION_VALIDATION,
    error_responses,
)
from ml_platform.api.routers._responses import materialization_response
from ml_platform.api.routers.exceptions import FeatureVersionNotFoundError
from ml_platform.api.services import minio_browser_url, minio_partition_url
from ml_platform.common.api.dtos import (
    FeatureCreateRequest,
    FeaturePartitionMaterializationRequest,
    FeatureResponse,
    MaterializationJobResponse,
)
from ml_platform.common.api.routes import FeatureRoutes

router = APIRouter(tags=["features"])


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


@router.get(FeatureRoutes.LIST.template, tags=["features"])
def list_features(
    session: DatabaseSession, *, include_deleted: bool = False
) -> list[dict[str, object]]:
    """List feature definitions with logical partition status counts."""
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
        for summary in FeatureRepository(session).summaries(
            include_deleted=include_deleted
        )
    ]


@router.get(FeatureRoutes.DETAIL.template, tags=["features"])
def get_feature(
    name: ResourcePathName,
    version: ResourcePathVersion,
    dependencies: ApiDependencies,
    session: DatabaseSession,
) -> dict[str, object]:
    """Return one feature, its dependencies, and logical partitions."""
    detail = FeatureRepository(session).detail(name, version)
    if detail is None:
        raise FeatureVersionNotFoundError
    table_path = f"features/{name}-v{version}"
    return {
        **_definition_response(detail.definition),
        "minio_table_url": minio_browser_url(dependencies, table_path),
        "datasets": [
            {
                "name": dependency.name,
                "version": dependency.version,
                "dataset_type": dependency.dataset_type,
                "input_window": {
                    "lookback_partitions": dependency.lookback_partitions
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


@router.delete(
    FeatureRoutes.DELETE.template,
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["features"],
)
def delete_feature(
    name: ResourcePathName,
    version: ResourcePathVersion,
    session: DatabaseSession,
) -> None:
    """Soft-delete a feature and cancel each unfinished partition job."""
    if not FeatureRepository(session).delete(name, version):
        raise FeatureVersionNotFoundError


@router.post(
    FeatureRoutes.CREATE.template,
    response_model=FeatureResponse,
    status_code=201,
    responses=error_responses(FEATURE_VALIDATION),
    tags=["features"],
)
def create_feature(
    payload: FeatureCreateRequest, session: DatabaseSession
) -> FeatureResponse:
    """Register an immutable feature definition and active dataset dependencies."""
    definition = FeatureRepository(session).create(payload)
    return FeatureResponse.model_validate(
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


@router.post(
    FeatureRoutes.MATERIALIZE.template,
    response_model=MaterializationJobResponse,
    status_code=202,
    responses=error_responses(PARTITION_VALIDATION),
    tags=["feature materialization"],
)
def materialize_feature_partitions(
    name: ResourcePathName,
    version: ResourcePathVersion,
    payload: FeaturePartitionMaterializationRequest,
    session: DatabaseSession,
) -> MaterializationJobResponse:
    """Request feature partitions from the persistence repository."""
    return materialization_response(
        FeatureRepository(session).materialize(name, version, payload)
    )
