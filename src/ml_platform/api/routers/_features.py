"""Feature registration, inspection, deletion, and materialization routes."""

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import case, func, select, update
from sqlalchemy.exc import IntegrityError

from ml_platform.api._dependencies import (
    DatabaseSession,
    ResourcePathName,
    ResourcePathVersion,
)
from ml_platform.api.domain import ResourceType
from ml_platform.api.repositories import (
    Dataset,
    Feature,
    FeatureDatasetDependency,
    FeaturePartition,
    FeaturePartitionRequest,
    FeatureRepository,
    Job,
    JobRepository,
)
from ml_platform.api.routers._openapi import (
    FEATURE_VALIDATION,
    PARTITION_VALIDATION,
    error_responses,
)
from ml_platform.api.routers._responses import materialization_response
from ml_platform.api.services import attach_minio_locations
from ml_platform.common.api.dtos import (
    FeatureCreateRequest,
    FeaturePartitionMaterializationRequest,
    FeatureResponse,
    MaterializationJobResponse,
)
from ml_platform.common.api.routes import FeatureRoutes

router = APIRouter(tags=["features"])


@router.get(FeatureRoutes.LIST.template, tags=["features"])
def list_features(
    session: DatabaseSession, *, include_deleted: bool = False
) -> list[dict[str, Any]]:
    """List feature definitions with logical partition status counts."""
    partition_counts = (
        select(
            FeaturePartitionRequest.feature_name.label("name"),
            FeaturePartitionRequest.feature_version.label("version"),
            func.count(FeaturePartitionRequest.partition_id).label(
                "partition_count"
            ),
            func.count(
                case(
                    (
                        FeaturePartitionRequest.status == "created",
                        FeaturePartitionRequest.partition_id,
                    )
                )
            ).label("created_partition_count"),
            func.count(
                case(
                    (
                        FeaturePartitionRequest.status == "pending",
                        FeaturePartitionRequest.partition_id,
                    )
                )
            ).label("pending_partition_count"),
            func.count(
                case(
                    (
                        FeaturePartitionRequest.status.in_(
                            ("failed", "cancelled")
                        ),
                        FeaturePartitionRequest.partition_id,
                    )
                )
            ).label("failed_partition_count"),
        )
        .group_by(
            FeaturePartitionRequest.feature_name,
            FeaturePartitionRequest.feature_version,
        )
        .subquery()
    )
    statement = (
        select(
            Feature,
            partition_counts.c.partition_count,
            partition_counts.c.created_partition_count,
            partition_counts.c.pending_partition_count,
            partition_counts.c.failed_partition_count,
        )
        .outerjoin(
            partition_counts,
            (partition_counts.c.name == Feature.name)
            & (partition_counts.c.version == Feature.version),
        )
        .order_by(Feature.name, Feature.version.desc())
    )
    if not include_deleted:
        statement = statement.where(Feature.deleted_at.is_(None))

    return [
        {
            "name": feature.name,
            "version": feature.version,
            "dtype": feature.dtype,
            "frequency": feature.frequency,
            "keys": feature.keys,
            "description": feature.description,
            "created_at": feature.created_at,
            "deleted_at": feature.deleted_at,
            "deprecated_at": feature.deprecated_at,
            "partition_count": partition_count or 0,
            "created_partition_count": created_partition_count or 0,
            "pending_partition_count": pending_partition_count or 0,
            "failed_partition_count": failed_partition_count or 0,
        }
        for (
            feature,
            partition_count,
            created_partition_count,
            pending_partition_count,
            failed_partition_count,
        ) in session.execute(statement).all()
    ]


@router.get(FeatureRoutes.DETAIL.template, tags=["features"])
def get_feature(
    name: ResourcePathName,
    version: ResourcePathVersion,
    session: DatabaseSession,
) -> dict[str, Any]:
    """Return one feature, its dependencies, and logical partitions."""
    feature = session.get(Feature, (name, version))
    if feature is None:
        raise HTTPException(
            status_code=404, detail="Feature version does not exist."
        )
    dependencies = session.execute(
        select(FeatureDatasetDependency, Dataset.dataset_type)
        .join(
            Dataset,
            (Dataset.name == FeatureDatasetDependency.dataset_name)
            & (Dataset.version == FeatureDatasetDependency.dataset_version),
        )
        .where(
            FeatureDatasetDependency.feature_name == name,
            FeatureDatasetDependency.feature_version == version,
        )
        .order_by(
            FeatureDatasetDependency.dataset_name,
            FeatureDatasetDependency.dataset_version,
        )
    )
    latest = (
        select(
            FeaturePartitionRequest.timestamp,
            func.max(FeaturePartitionRequest.created_at).label("created_at"),
        )
        .where(
            FeaturePartitionRequest.feature_name == name,
            FeaturePartitionRequest.feature_version == version,
        )
        .group_by(FeaturePartitionRequest.timestamp)
        .subquery()
    )
    partitions = session.execute(
        select(FeaturePartitionRequest, FeaturePartition)
        .join(
            latest,
            (latest.c.timestamp == FeaturePartitionRequest.timestamp)
            & (latest.c.created_at == FeaturePartitionRequest.created_at),
        )
        .outerjoin(
            FeaturePartition,
            FeaturePartition.partition_id
            == FeaturePartitionRequest.partition_id,
        )
        .order_by(FeaturePartitionRequest.timestamp.desc())
    )
    response: dict[str, Any] = {
        "name": feature.name,
        "version": feature.version,
        "dtype": feature.dtype,
        "frequency": feature.frequency,
        "keys": feature.keys,
        "validations": feature.validations,
        "transform": feature.transform,
        "description": feature.description,
        "parameters": feature.parameters,
        "created_at": feature.created_at,
        "deleted_at": feature.deleted_at,
        "deprecated_at": feature.deprecated_at,
        "datasets": [
            {
                "name": dependency.dataset_name,
                "version": dependency.dataset_version,
                "dataset_type": dataset_type,
                "input_window": {
                    "lookback_partitions": dependency.num_lookback_partitions
                },
            }
            for dependency, dataset_type in dependencies
        ],
        "partitions": [
            {
                "partition_id": request.partition_id,
                "timestamp": request.timestamp,
                "status": request.status,
                "created_at": request.created_at,
                "updated_at": request.updated_at,
                "content_hash": None
                if receipt is None
                else receipt.content_hash,
                "snapshot_id": None if receipt is None else receipt.snapshot_id,
            }
            for request, receipt in partitions
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
                ResourceType.FEATURE, name, version
            )
        ],
    }
    attach_minio_locations(response, f"features/{name}-v{version}")
    return response


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
    feature = session.get(Feature, (name, version))
    if feature is None:
        raise HTTPException(
            status_code=404, detail="Feature version does not exist."
        )
    if feature.deleted_at is None:
        feature.deleted_at = datetime.now(UTC)
    pending_job_ids = select(FeaturePartitionRequest.job_id).where(
        FeaturePartitionRequest.feature_name == name,
        FeaturePartitionRequest.feature_version == version,
        FeaturePartitionRequest.status == "pending",
    )
    session.execute(
        update(FeaturePartitionRequest)
        .where(
            FeaturePartitionRequest.feature_name == name,
            FeaturePartitionRequest.feature_version == version,
            FeaturePartitionRequest.status == "pending",
        )
        .values(status="cancelled", updated_at=func.now())
    )
    session.execute(
        update(Job)
        .where(Job.status == "submitted", Job.id.in_(pending_job_ids))
        .values(status="cancelled", updated_at=func.now())
    )
    session.commit()


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
    for dependency in payload.datasets:
        dataset = session.get(Dataset, (dependency.name, dependency.version))
        if dataset is None or dataset.deleted_at is not None:
            raise HTTPException(
                status_code=422,
                detail="Feature datasets must reference active dataset versions.",
            )
        if dataset.dataset_type != dependency.dataset_type.value:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Feature dataset type must match the registered dataset "
                    "version."
                ),
            )
    feature = Feature(
        name=payload.name,
        version=payload.version,
        dtype=payload.dtype.value,
        frequency=payload.frequency.value,
        keys=[key.value for key in payload.keys],
        validations=[
            validation.model_dump() for validation in payload.validations
        ],
        transform=payload.transform,
        description=payload.description,
        parameters=payload.parameters,
    )
    try:
        session.add(feature)
        for dependency in payload.datasets:
            session.add(
                FeatureDatasetDependency(
                    feature_name=payload.name,
                    feature_version=payload.version,
                    dataset_name=dependency.name,
                    dataset_version=dependency.version,
                    num_lookback_partitions=(
                        dependency.input_window.lookback_partitions
                    ),
                )
            )
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(
            status_code=409, detail="Feature name and version already exist."
        ) from error
    session.refresh(feature)
    return FeatureResponse.model_validate(
        {
            "name": feature.name,
            "version": feature.version,
            "dtype": feature.dtype,
            "frequency": feature.frequency,
            "keys": feature.keys,
            "validations": feature.validations,
            "transform": feature.transform,
            "description": feature.description,
            "parameters": feature.parameters,
            "created_at": feature.created_at,
            "datasets": [item.model_dump() for item in payload.datasets],
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
