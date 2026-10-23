"""SQLAlchemy persistence operations for materialization jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import CompoundSelect, literal, select, union_all

from ml_platform.api.domain import (
    JobPartition,
    JobStatus,
    JobTarget,
    JobType,
    MaterializationJob,
    PartitionStatus,
    ResourceType,
)
from ml_platform.api.repositories._models import (
    DatasetPartitionRequest,
    FeaturePartitionRequest,
    Job,
    LabelPartitionRequest,
)
from ml_platform.common.types import (
    CreatedAt,
    JobId,
    PartitionTimestamp,
    ResourceName,
    ResourceVersion,
    UpdatedAt,
)

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.orm import Session


@dataclass(frozen=True)
class JobRepository:
    """Read materialization jobs and their resource-target projections."""

    session: Session

    def _targets(self, job_id: JobId) -> CompoundSelect[tuple[str, str, int]]:
        """Build the stable resource-target projection for one job."""
        return union_all(
            select(
                literal("dataset").label("resource_type"),
                DatasetPartitionRequest.dataset_name.label("name"),
                DatasetPartitionRequest.dataset_version.label("version"),
            ).where(DatasetPartitionRequest.job_id == job_id),
            select(
                literal("feature").label("resource_type"),
                FeaturePartitionRequest.feature_name.label("name"),
                FeaturePartitionRequest.feature_version.label("version"),
            ).where(FeaturePartitionRequest.job_id == job_id),
            select(
                literal("label").label("resource_type"),
                LabelPartitionRequest.label_name.label("name"),
                LabelPartitionRequest.label_version.label("version"),
            ).where(LabelPartitionRequest.job_id == job_id),
        )

    def _partitions(
        self, job_id: JobId
    ) -> CompoundSelect[tuple[str, str, int, datetime, str]]:
        """Build the logical partition projection for one job."""
        return union_all(
            select(
                literal("dataset").label("resource_type"),
                DatasetPartitionRequest.dataset_name.label("name"),
                DatasetPartitionRequest.dataset_version.label("version"),
                DatasetPartitionRequest.timestamp,
                DatasetPartitionRequest.status,
            ).where(DatasetPartitionRequest.job_id == job_id),
            select(
                literal("feature").label("resource_type"),
                FeaturePartitionRequest.feature_name.label("name"),
                FeaturePartitionRequest.feature_version.label("version"),
                FeaturePartitionRequest.timestamp,
                FeaturePartitionRequest.status,
            ).where(FeaturePartitionRequest.job_id == job_id),
            select(
                literal("label").label("resource_type"),
                LabelPartitionRequest.label_name.label("name"),
                LabelPartitionRequest.label_version.label("version"),
                LabelPartitionRequest.timestamp,
                LabelPartitionRequest.status,
            ).where(LabelPartitionRequest.job_id == job_id),
        )

    @staticmethod
    def _job(model: Job) -> MaterializationJob:
        """Map an ORM job row to the API-private domain model."""
        return MaterializationJob(
            id=JobId(model.id),
            type=JobType(model.type),
            status=JobStatus(model.status),
            created_at=CreatedAt(model.created_at),
            updated_at=UpdatedAt(model.updated_at),
        )

    def get(self, job_id: JobId) -> MaterializationJob | None:
        """Return one persisted job by its immutable ID."""
        model = self.session.get(Job, job_id)
        return None if model is None else self._job(model)

    def list_jobs(self) -> tuple[MaterializationJob, ...]:
        """Return jobs ordered from newest to oldest."""
        return tuple(
            self._job(model)
            for model in self.session.scalars(
                select(Job).order_by(Job.created_at.desc())
            )
        )

    def targets(self, job_id: JobId) -> tuple[JobTarget, ...]:
        """Return distinct versioned resource targets for one job."""
        statement = (
            select(self._targets(job_id).subquery())
            .distinct()
            .order_by("resource_type", "name", "version")
        )
        return tuple(
            JobTarget(
                resource_type=ResourceType(row["resource_type"]),
                name=ResourceName(row["name"]),
                version=ResourceVersion(row["version"]),
            )
            for row in self.session.execute(statement).mappings()
        )

    def partitions(self, job_id: JobId) -> tuple[JobPartition, ...]:
        """Return all logical resource partitions targeted by one job."""
        statement = select(self._partitions(job_id).subquery()).order_by(
            "timestamp", "resource_type", "name", "version"
        )
        return tuple(
            JobPartition(
                resource_type=ResourceType(row["resource_type"]),
                name=ResourceName(row["name"]),
                version=ResourceVersion(row["version"]),
                timestamp=PartitionTimestamp(row["timestamp"]),
                status=PartitionStatus(row["status"]),
            )
            for row in self.session.execute(statement).mappings()
        )

    def _requests(
        self,
        resource_type: ResourceType,
        name: ResourceName,
        version: ResourceVersion,
    ) -> CompoundSelect[tuple[Job]]:
        """Build the typed request projection for one resource version."""
        match resource_type:
            case ResourceType.DATASET:
                return union_all(
                    select(Job)
                    .join(
                        DatasetPartitionRequest,
                        DatasetPartitionRequest.job_id == Job.id,
                    )
                    .where(
                        DatasetPartitionRequest.dataset_name == name,
                        DatasetPartitionRequest.dataset_version == version,
                    )
                )
            case ResourceType.FEATURE:
                return union_all(
                    select(Job)
                    .join(
                        FeaturePartitionRequest,
                        FeaturePartitionRequest.job_id == Job.id,
                    )
                    .where(
                        FeaturePartitionRequest.feature_name == name,
                        FeaturePartitionRequest.feature_version == version,
                    )
                )
            case ResourceType.LABEL:
                return union_all(
                    select(Job)
                    .join(
                        LabelPartitionRequest,
                        LabelPartitionRequest.job_id == Job.id,
                    )
                    .where(
                        LabelPartitionRequest.label_name == name,
                        LabelPartitionRequest.label_version == version,
                    )
                )
            case _ as unsupported_resource_type:
                raise ValueError(
                    f"Unsupported resource type {unsupported_resource_type!r}."
                )

    def for_resource(
        self,
        resource_type: ResourceType,
        name: ResourceName,
        version: ResourceVersion,
    ) -> tuple[MaterializationJob, ...]:
        """Return all jobs targeting one immutable resource version."""
        requests = self._requests(resource_type, name, version).subquery()
        statement = (
            select(Job)
            .join(requests, requests.c.id == Job.id)
            .order_by(Job.created_at.desc())
        )
        return tuple(
            self._job(model) for model in self.session.scalars(statement)
        )
