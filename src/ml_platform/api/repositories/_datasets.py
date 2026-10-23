"""SQLAlchemy persistence operations for datasets."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import case, func, select, update

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

from itertools import starmap

from ml_platform.api.domain import (
    DatasetDefinition,
    DatasetSummary,
    JobStatus,
    JobType,
    MaterializationPartition,
    MaterializationRequest,
    PartitionCounts,
    PartitionStatus,
    ResourceLifecycle,
    ResourcePartition,
)
from ml_platform.api.repositories._models import (
    Dataset,
    DatasetPartition,
    DatasetPartitionRequest,
    Job,
)
from ml_platform.common.types import (
    CreatedAt,
    Description,
    JobId,
    PartitionId,
    PartitionTimestamp,
    ResourceName,
    ResourceVersion,
    SnapshotId,
    UpdatedAt,
)


@dataclass(frozen=True)
class DatasetRepository:
    """Persist and read immutable datasets and their partition requests."""

    session: Session

    @staticmethod
    def _lifecycle(dataset: Dataset) -> ResourceLifecycle:
        """Map nullable persistence lifecycle markers to one durable state."""
        if dataset.deleted_at is not None:
            return ResourceLifecycle.DELETED
        if dataset.deprecated_at is not None:
            return ResourceLifecycle.DEPRECATED
        return ResourceLifecycle.ACTIVE

    @classmethod
    def _definition(cls, dataset: Dataset) -> DatasetDefinition:
        """Map one dataset ORM row to an immutable domain definition."""
        return DatasetDefinition(
            name=ResourceName(dataset.name),
            version=ResourceVersion(dataset.version),
            dataset_type=dataset.dataset_type,
            config=dataset.config,
            created_at=CreatedAt(dataset.created_at),
            lifecycle=cls._lifecycle(dataset),
            deleted_at=dataset.deleted_at,
            deprecated_at=dataset.deprecated_at,
        )

    @staticmethod
    def _materialization_partition(
        request: DatasetPartitionRequest,
    ) -> MaterializationPartition:
        """Map a persisted request to the caller-visible request projection."""
        return MaterializationPartition(
            id=PartitionId(request.partition_id),
            job_id=request.job_id,
            timestamp=PartitionTimestamp(request.timestamp),
            status=PartitionStatus(request.status),
            created_at=CreatedAt(request.created_at),
            updated_at=UpdatedAt(request.updated_at),
        )

    @staticmethod
    def _resource_partition(
        request: DatasetPartitionRequest,
        receipt: DatasetPartition | None,
    ) -> ResourcePartition:
        """Map the latest logical partition request and its receipt."""
        return ResourcePartition(
            id=PartitionId(request.partition_id),
            timestamp=PartitionTimestamp(request.timestamp),
            status=PartitionStatus(request.status),
            created_at=CreatedAt(request.created_at),
            updated_at=UpdatedAt(request.updated_at),
            content_hash=None if receipt is None else receipt.content_hash,
            snapshot_id=None
            if receipt is None
            else SnapshotId(receipt.snapshot_id),
        )

    def list(self, *, include_deleted: bool) -> tuple[DatasetSummary, ...]:
        """Return registered dataset versions and their partition state totals."""
        counts = (
            select(
                DatasetPartitionRequest.dataset_name.label("name"),
                DatasetPartitionRequest.dataset_version.label("version"),
                func.count(DatasetPartitionRequest.partition_id).label("total"),
                func.count(
                    case(
                        (
                            DatasetPartitionRequest.status == "created",
                            DatasetPartitionRequest.partition_id,
                        )
                    )
                ).label("created"),
                func.count(
                    case(
                        (
                            DatasetPartitionRequest.status == "pending",
                            DatasetPartitionRequest.partition_id,
                        )
                    )
                ).label("pending"),
                func.count(
                    case(
                        (
                            DatasetPartitionRequest.status.in_(
                                ("failed", "cancelled")
                            ),
                            DatasetPartitionRequest.partition_id,
                        )
                    )
                ).label("failed"),
            )
            .group_by(
                DatasetPartitionRequest.dataset_name,
                DatasetPartitionRequest.dataset_version,
            )
            .subquery()
        )
        statement = (
            select(
                Dataset,
                counts.c.total,
                counts.c.created,
                counts.c.pending,
                counts.c.failed,
            )
            .outerjoin(
                counts,
                (counts.c.name == Dataset.name)
                & (counts.c.version == Dataset.version),
            )
            .order_by(Dataset.name, Dataset.version.desc())
        )
        if not include_deleted:
            statement = statement.where(Dataset.deleted_at.is_(None))
        return tuple(
            DatasetSummary(
                definition=self._definition(dataset),
                partitions=PartitionCounts(
                    total=total or 0,
                    created=created or 0,
                    pending=pending or 0,
                    failed=failed or 0,
                ),
            )
            for dataset, total, created, pending, failed in self.session.execute(
                statement
            )
        )

    def get(
        self, name: ResourceName, version: ResourceVersion
    ) -> DatasetDefinition | None:
        """Return one immutable dataset definition when it exists."""
        dataset = self.session.get(Dataset, (name, version))
        return None if dataset is None else self._definition(dataset)

    def partitions(
        self, name: ResourceName, version: ResourceVersion
    ) -> tuple[ResourcePartition, ...]:
        """Return the newest request for each logical dataset partition."""
        latest = (
            select(
                DatasetPartitionRequest.timestamp,
                func.max(DatasetPartitionRequest.created_at).label(
                    "created_at"
                ),
            )
            .where(
                DatasetPartitionRequest.dataset_name == name,
                DatasetPartitionRequest.dataset_version == version,
            )
            .group_by(DatasetPartitionRequest.timestamp)
            .subquery()
        )
        statement = (
            select(DatasetPartitionRequest, DatasetPartition)
            .join(
                latest,
                (latest.c.timestamp == DatasetPartitionRequest.timestamp)
                & (latest.c.created_at == DatasetPartitionRequest.created_at),
            )
            .outerjoin(
                DatasetPartition,
                DatasetPartition.partition_id
                == DatasetPartitionRequest.partition_id,
            )
            .order_by(DatasetPartitionRequest.timestamp.desc())
        )
        return tuple(
            starmap(self._resource_partition, self.session.execute(statement))
        )

    def update_description(
        self,
        name: ResourceName,
        version: ResourceVersion,
        description: Description,
    ) -> DatasetDefinition | None:
        """Persist display-only description metadata for one dataset version."""
        dataset = self.session.get(Dataset, (name, version))
        if dataset is None:
            return None
        dataset.config = {**dataset.config, "description": description}
        self.session.commit()
        self.session.refresh(dataset)
        return self._definition(dataset)

    def delete(self, name: ResourceName, version: ResourceVersion) -> bool:
        """Soft-delete a dataset and cancel its unfinished materializations."""
        dataset = self.session.get(Dataset, (name, version))
        if dataset is None:
            return False
        if dataset.deleted_at is None:
            dataset.deleted_at = datetime.now(UTC)
        job_ids = select(DatasetPartitionRequest.job_id).where(
            DatasetPartitionRequest.dataset_name == name,
            DatasetPartitionRequest.dataset_version == version,
            DatasetPartitionRequest.status == PartitionStatus.PENDING.value,
        )
        self.session.execute(
            update(DatasetPartitionRequest)
            .where(
                DatasetPartitionRequest.dataset_name == name,
                DatasetPartitionRequest.dataset_version == version,
                DatasetPartitionRequest.status == PartitionStatus.PENDING.value,
            )
            .values(
                status=PartitionStatus.CANCELLED.value, updated_at=func.now()
            )
        )
        self.session.execute(
            update(Job)
            .where(Job.status == JobStatus.SUBMITTED.value, Job.id.in_(job_ids))
            .values(status=JobStatus.CANCELLED.value, updated_at=func.now())
        )
        self.session.commit()
        return True

    def create(
        self,
        *,
        name: ResourceName,
        version: ResourceVersion,
        dataset_type: str,
        config: dict[str, object],
    ) -> DatasetDefinition:
        """Persist one immutable dataset definition inside the active transaction."""
        dataset = Dataset(
            name=name,
            version=version,
            dataset_type=dataset_type,
            config=config,
        )
        self.session.add(dataset)
        self.session.commit()
        self.session.refresh(dataset)
        return self._definition(dataset)

    def materialize(
        self,
        name: ResourceName,
        version: ResourceVersion,
        timestamps: tuple[PartitionTimestamp, ...],
    ) -> MaterializationRequest | None:
        """Create only missing dataset partition requests under transaction locks."""
        dataset = self.session.get(Dataset, (name, version))
        if dataset is None or dataset.deleted_at is not None:
            return None
        existing: list[MaterializationPartition] = []
        missing: list[PartitionTimestamp] = []
        for timestamp in sorted(timestamps):
            self.session.execute(
                select(
                    func.pg_advisory_xact_lock(
                        func.hashtextextended(
                            f"dataset:{name}:{version}:{timestamp.isoformat()}",
                            0,
                        )
                    )
                )
            )
        for timestamp in timestamps:
            completed = self.session.scalars(
                select(DatasetPartitionRequest)
                .where(
                    DatasetPartitionRequest.dataset_name == name,
                    DatasetPartitionRequest.dataset_version == version,
                    DatasetPartitionRequest.timestamp == timestamp,
                    DatasetPartitionRequest.status
                    == PartitionStatus.CREATED.value,
                )
                .order_by(DatasetPartitionRequest.created_at.desc())
                .limit(1)
            ).first()
            if completed is not None:
                existing.append(self._materialization_partition(completed))
                continue
            pending = self.session.scalars(
                select(DatasetPartitionRequest)
                .where(
                    DatasetPartitionRequest.dataset_name == name,
                    DatasetPartitionRequest.dataset_version == version,
                    DatasetPartitionRequest.timestamp == timestamp,
                    DatasetPartitionRequest.status
                    == PartitionStatus.PENDING.value,
                )
                .order_by(DatasetPartitionRequest.created_at.desc())
                .limit(1)
            ).first()
            if pending is not None:
                existing.append(self._materialization_partition(pending))
                continue
            missing.append(timestamp)
        if not missing:
            self.session.commit()
            return MaterializationRequest(
                submitted_job=None, partitions=tuple(existing)
            )
        job = Job(
            type=JobType.DATASET_MATERIALIZATION.value,
            status=JobStatus.SUBMITTED.value,
        )
        self.session.add(job)
        self.session.flush()
        requests = [
            DatasetPartitionRequest(
                job_id=job.id,
                dataset_name=name,
                dataset_version=version,
                timestamp=timestamp,
                status=PartitionStatus.PENDING.value,
                callback_token=secrets.token_urlsafe(32),
            )
            for timestamp in missing
        ]
        self.session.add_all(requests)
        self.session.commit()
        for request in requests:
            self.session.refresh(request)
        self.session.refresh(job)
        from ml_platform.api.domain import MaterializationJob

        submitted = MaterializationJob(
            id=JobId(job.id),
            type=JobType(job.type),
            status=JobStatus(job.status),
            created_at=CreatedAt(job.created_at),
            updated_at=UpdatedAt(job.updated_at),
        )
        return MaterializationRequest(
            submitted_job=submitted,
            partitions=(
                *existing,
                *(
                    self._materialization_partition(request)
                    for request in requests
                ),
            ),
        )
