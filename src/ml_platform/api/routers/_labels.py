"""Label registration, inspection, deletion, and materialization routes."""

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
    Job,
    JobRepository,
    Label,
    LabelDatasetDependency,
    LabelPartition,
    LabelPartitionRequest,
    LabelRepository,
)
from ml_platform.api.routers._openapi import (
    FEATURE_VALIDATION,
    PARTITION_VALIDATION,
    error_responses,
)
from ml_platform.api.routers._responses import materialization_response
from ml_platform.api.services import attach_minio_locations
from ml_platform.common.api.dtos import (
    LabelCreateRequest,
    LabelPartitionMaterializationRequest,
    LabelResponse,
    MaterializationJobResponse,
)
from ml_platform.common.api.routes import LabelRoutes

router = APIRouter(tags=["labels"])


@router.get(LabelRoutes.LIST.template, tags=["labels"])
def list_labels(
    session: DatabaseSession, *, include_deleted: bool = False
) -> list[dict[str, Any]]:
    """List label definitions with logical partition status counts."""
    partition_counts = (
        select(
            LabelPartitionRequest.label_name.label("name"),
            LabelPartitionRequest.label_version.label("version"),
            func.count(LabelPartitionRequest.partition_id).label(
                "partition_count"
            ),
            func.count(
                case(
                    (
                        LabelPartitionRequest.status == "created",
                        LabelPartitionRequest.partition_id,
                    )
                )
            ).label("created_partition_count"),
            func.count(
                case(
                    (
                        LabelPartitionRequest.status == "pending",
                        LabelPartitionRequest.partition_id,
                    )
                )
            ).label("pending_partition_count"),
            func.count(
                case(
                    (
                        LabelPartitionRequest.status.in_(
                            ("failed", "cancelled")
                        ),
                        LabelPartitionRequest.partition_id,
                    )
                )
            ).label("failed_partition_count"),
        )
        .group_by(
            LabelPartitionRequest.label_name,
            LabelPartitionRequest.label_version,
        )
        .subquery()
    )
    statement = (
        select(
            Label,
            partition_counts.c.partition_count,
            partition_counts.c.created_partition_count,
            partition_counts.c.pending_partition_count,
            partition_counts.c.failed_partition_count,
        )
        .outerjoin(
            partition_counts,
            (partition_counts.c.name == Label.name)
            & (partition_counts.c.version == Label.version),
        )
        .order_by(Label.name, Label.version.desc())
    )
    if not include_deleted:
        statement = statement.where(Label.deleted_at.is_(None))

    return [
        {
            "name": label.name,
            "version": label.version,
            "dtype": label.dtype,
            "frequency": label.frequency,
            "keys": label.keys,
            "description": label.description,
            "created_at": label.created_at,
            "deleted_at": label.deleted_at,
            "deprecated_at": label.deprecated_at,
            "partition_count": partition_count or 0,
            "created_partition_count": created_partition_count or 0,
            "pending_partition_count": pending_partition_count or 0,
            "failed_partition_count": failed_partition_count or 0,
        }
        for (
            label,
            partition_count,
            created_partition_count,
            pending_partition_count,
            failed_partition_count,
        ) in session.execute(statement).all()
    ]


@router.get(LabelRoutes.DETAIL.template, tags=["labels"])
def get_label(
    name: ResourcePathName,
    version: ResourcePathVersion,
    session: DatabaseSession,
) -> dict[str, Any]:
    """Return one label, its dependencies, and logical partitions."""
    label = session.get(Label, (name, version))
    if label is None:
        raise HTTPException(
            status_code=404, detail="Label version does not exist."
        )
    dependencies = session.execute(
        select(LabelDatasetDependency, Dataset.dataset_type)
        .join(
            Dataset,
            (Dataset.name == LabelDatasetDependency.dataset_name)
            & (Dataset.version == LabelDatasetDependency.dataset_version),
        )
        .where(
            LabelDatasetDependency.label_name == name,
            LabelDatasetDependency.label_version == version,
        )
        .order_by(
            LabelDatasetDependency.dataset_name,
            LabelDatasetDependency.dataset_version,
        )
    )
    latest = (
        select(
            LabelPartitionRequest.timestamp,
            func.max(LabelPartitionRequest.created_at).label("created_at"),
        )
        .where(
            LabelPartitionRequest.label_name == name,
            LabelPartitionRequest.label_version == version,
        )
        .group_by(LabelPartitionRequest.timestamp)
        .subquery()
    )
    partitions = session.execute(
        select(LabelPartitionRequest, LabelPartition)
        .join(
            latest,
            (latest.c.timestamp == LabelPartitionRequest.timestamp)
            & (latest.c.created_at == LabelPartitionRequest.created_at),
        )
        .outerjoin(
            LabelPartition,
            LabelPartition.partition_id == LabelPartitionRequest.partition_id,
        )
        .order_by(LabelPartitionRequest.timestamp.desc())
    )
    response: dict[str, Any] = {
        "name": label.name,
        "version": label.version,
        "dtype": label.dtype,
        "frequency": label.frequency,
        "keys": label.keys,
        "validations": label.validations,
        "transform": label.transform,
        "description": label.description,
        "parameters": label.parameters,
        "created_at": label.created_at,
        "deleted_at": label.deleted_at,
        "deprecated_at": label.deprecated_at,
        "datasets": [
            {
                "name": dependency.dataset_name,
                "version": dependency.dataset_version,
                "dataset_type": dataset_type,
                "input_window": {
                    "lookback_partitions": dependency.num_lookback_partitions,
                    "lookahead_partitions": dependency.num_lookahead_partitions,
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
                ResourceType.LABEL, name, version
            )
        ],
    }
    attach_minio_locations(response, f"labels/{name}-v{version}")
    return response


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
    for dependency in payload.datasets:
        dataset = session.get(Dataset, (dependency.name, dependency.version))
        if dataset is None or dataset.deleted_at is not None:
            raise HTTPException(
                status_code=422,
                detail="Label datasets must reference active dataset versions.",
            )
        if dataset.dataset_type != dependency.dataset_type.value:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Label dataset type must match the registered dataset version."
                ),
            )
    label = Label(
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
        session.add(label)
        for dependency in payload.datasets:
            session.add(
                LabelDatasetDependency(
                    label_name=payload.name,
                    label_version=payload.version,
                    dataset_name=dependency.name,
                    dataset_version=dependency.version,
                    num_lookback_partitions=(
                        dependency.input_window.lookback_partitions
                    ),
                    num_lookahead_partitions=(
                        dependency.input_window.lookahead_partitions
                    ),
                )
            )
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(
            status_code=409, detail="Label name and version already exist."
        ) from error
    session.refresh(label)
    return LabelResponse.model_validate(
        {
            "name": label.name,
            "version": label.version,
            "dtype": label.dtype,
            "frequency": label.frequency,
            "keys": label.keys,
            "validations": label.validations,
            "transform": label.transform,
            "description": label.description,
            "parameters": label.parameters,
            "created_at": label.created_at,
            "datasets": [item.model_dump() for item in payload.datasets],
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
    label = session.get(Label, (name, version))
    if label is None:
        raise HTTPException(
            status_code=404, detail="Label version does not exist."
        )
    if label.deleted_at is None:
        label.deleted_at = datetime.now(UTC)
    pending_job_ids = select(LabelPartitionRequest.job_id).where(
        LabelPartitionRequest.label_name == name,
        LabelPartitionRequest.label_version == version,
        LabelPartitionRequest.status == "pending",
    )
    session.execute(
        update(LabelPartitionRequest)
        .where(
            LabelPartitionRequest.label_name == name,
            LabelPartitionRequest.label_version == version,
            LabelPartitionRequest.status == "pending",
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
