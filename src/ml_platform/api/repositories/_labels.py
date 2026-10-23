"""SQLAlchemy persistence operations for label materialization."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import case, func, select, update
from sqlalchemy.exc import IntegrityError

from ml_platform.api.domain import (
    JobStatus,
    JobType,
    MaterializationJob,
    MaterializationPartition,
    MaterializationRequest,
    PartitionCounts,
    PartitionStatus,
    ResourcePartition,
    ResourceType,
    TransformDatasetDependency,
    TransformDefinition,
    TransformDetail,
    TransformSummary,
)
from ml_platform.api.domain._lifecycle import ResourceLifecycle
from ml_platform.api.repositories._datasets import DatasetRepository
from ml_platform.api.repositories._jobs import JobRepository
from ml_platform.api.repositories._models import (
    Dataset,
    DatasetPartition,
    DatasetPartitionRequest,
    Job,
    Label,
    LabelDatasetDependency,
    LabelPartition,
    LabelPartitionDatasetDependency,
    LabelPartitionPlan,
    LabelPartitionRequest,
)
from ml_platform.api.repositories._partition_validation import (
    validate_partition_timestamps,
)
from ml_platform.api.repositories.exceptions import (
    DeletedLabelInputDatasetError,
    LabelDependencyDatasetUnavailableError,
    LabelPartitionPlanConflictError,
    LabelVersionExistsError,
    LabelVersionUnavailableError,
    MissingLabelInputPartitionsError,
)
from ml_platform.common.api.dtos import (
    FeatureMaterializationMode,
    LabelCreateRequest,
    LabelPartitionMaterializationRequest,
)
from ml_platform.common.types import (
    CreatedAt,
    Description,
    JobId,
    PartitionId,
    PartitionTimestamp,
    PartitionType,
    ResourceName,
    ResourceVersion,
    SnapshotId,
    UpdatedAt,
)

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.orm import Session


@dataclass(frozen=True)
class LabelRepository:
    """Persist immutable label input plans and materialization requests."""

    session: Session

    @staticmethod
    def _definition(label: Label) -> TransformDefinition:
        """Map a label ORM row to its immutable domain definition."""
        lifecycle = (
            ResourceLifecycle.DELETED
            if label.deleted_at is not None
            else ResourceLifecycle.DEPRECATED
            if label.deprecated_at is not None
            else ResourceLifecycle.ACTIVE
        )
        return TransformDefinition(
            name=ResourceName(label.name),
            version=ResourceVersion(label.version),
            dtype=label.dtype,
            frequency=label.frequency,
            keys=tuple(label.keys),
            validations=tuple(label.validations),
            transform=label.transform,
            description=Description(label.description),
            parameters=label.parameters,
            created_at=CreatedAt(label.created_at),
            lifecycle=lifecycle,
            deleted_at=label.deleted_at,
            deprecated_at=label.deprecated_at,
        )

    @staticmethod
    def _resource_partition(
        request: LabelPartitionRequest,
        receipt: LabelPartition | None,
    ) -> ResourcePartition:
        """Map a label request and receipt to its latest partition state."""
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

    def summaries(
        self, *, include_deleted: bool
    ) -> tuple[TransformSummary, ...]:
        """Return label versions with their latest logical partition counts."""
        counts = (
            select(
                LabelPartitionRequest.label_name.label("name"),
                LabelPartitionRequest.label_version.label("version"),
                func.count(LabelPartitionRequest.partition_id).label("total"),
                func.count(
                    case(
                        (
                            LabelPartitionRequest.status
                            == PartitionStatus.CREATED.value,
                            LabelPartitionRequest.partition_id,
                        )
                    )
                ).label("created"),
                func.count(
                    case(
                        (
                            LabelPartitionRequest.status
                            == PartitionStatus.PENDING.value,
                            LabelPartitionRequest.partition_id,
                        )
                    )
                ).label("pending"),
                func.count(
                    case(
                        (
                            LabelPartitionRequest.status.in_(
                                (
                                    PartitionStatus.FAILED.value,
                                    PartitionStatus.CANCELLED.value,
                                )
                            ),
                            LabelPartitionRequest.partition_id,
                        )
                    )
                ).label("failed"),
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
                counts.c.total,
                counts.c.created,
                counts.c.pending,
                counts.c.failed,
            )
            .outerjoin(
                counts,
                (counts.c.name == Label.name)
                & (counts.c.version == Label.version),
            )
            .order_by(Label.name, Label.version.desc())
        )
        if not include_deleted:
            statement = statement.where(Label.deleted_at.is_(None))
        return tuple(
            TransformSummary(
                definition=self._definition(label),
                partitions=PartitionCounts(
                    total=total or 0,
                    created=created or 0,
                    pending=pending or 0,
                    failed=failed or 0,
                ),
            )
            for label, total, created, pending, failed in self.session.execute(
                statement
            )
        )

    def detail(
        self, name: ResourceName, version: ResourceVersion
    ) -> TransformDetail | None:
        """Return a label with dependencies, latest partitions, and jobs."""
        label = self.session.get(Label, (name, version))
        if label is None:
            return None
        dependencies = tuple(
            TransformDatasetDependency(
                name=dependency.dataset_name,
                version=dependency.dataset_version,
                dataset_type=dataset_type,
                lookback_partitions=dependency.num_lookback_partitions,
                lookahead_partitions=dependency.num_lookahead_partitions,
            )
            for dependency, dataset_type in self.session.execute(
                select(LabelDatasetDependency, Dataset.dataset_type)
                .join(
                    Dataset,
                    (Dataset.name == LabelDatasetDependency.dataset_name)
                    & (
                        Dataset.version
                        == LabelDatasetDependency.dataset_version
                    ),
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
        partitions = tuple(
            self._resource_partition(request, receipt)
            for request, receipt in self.session.execute(
                select(LabelPartitionRequest, LabelPartition)
                .join(
                    latest,
                    (latest.c.timestamp == LabelPartitionRequest.timestamp)
                    & (latest.c.created_at == LabelPartitionRequest.created_at),
                )
                .outerjoin(
                    LabelPartition,
                    LabelPartition.partition_id
                    == LabelPartitionRequest.partition_id,
                )
                .order_by(LabelPartitionRequest.timestamp.desc())
            )
        )
        return TransformDetail(
            definition=self._definition(label),
            datasets=dependencies,
            partitions=partitions,
            jobs=JobRepository(self.session).for_resource(
                ResourceType.LABEL, name, version
            ),
        )

    def create(self, payload: LabelCreateRequest) -> TransformDefinition:
        """Register a label after validating every declared dataset dependency."""
        for dependency in payload.datasets:
            dataset = self.session.get(
                Dataset, (dependency.name, dependency.version)
            )
            if dataset is None or dataset.deleted_at is not None:
                raise LabelDependencyDatasetUnavailableError
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
            self.session.add(label)
            self.session.add_all(
                LabelDatasetDependency(
                    label_name=payload.name,
                    label_version=payload.version,
                    dataset_name=dependency.name,
                    dataset_version=dependency.version,
                    num_lookback_partitions=dependency.input_window.lookback_partitions,
                    num_lookahead_partitions=dependency.input_window.lookahead_partitions,
                )
                for dependency in payload.datasets
            )
            self.session.commit()
        except IntegrityError as error:
            self.session.rollback()
            raise LabelVersionExistsError from error
        self.session.refresh(label)
        return self._definition(label)

    def delete(self, name: ResourceName, version: ResourceVersion) -> bool:
        """Soft-delete a label and cancel its unfinished materializations."""
        label = self.session.get(Label, (name, version))
        if label is None:
            return False
        if label.deleted_at is None:
            label.deleted_at = datetime.now(UTC)
        pending_job_ids = select(LabelPartitionRequest.job_id).where(
            LabelPartitionRequest.label_name == name,
            LabelPartitionRequest.label_version == version,
            LabelPartitionRequest.status == PartitionStatus.PENDING.value,
        )
        self.session.execute(
            update(LabelPartitionRequest)
            .where(
                LabelPartitionRequest.label_name == name,
                LabelPartitionRequest.label_version == version,
                LabelPartitionRequest.status == PartitionStatus.PENDING.value,
            )
            .values(
                status=PartitionStatus.CANCELLED.value, updated_at=func.now()
            )
        )
        self.session.execute(
            update(Job)
            .where(
                Job.status == JobStatus.SUBMITTED.value,
                Job.id.in_(pending_job_ids),
            )
            .values(status=JobStatus.CANCELLED.value, updated_at=func.now())
        )
        self.session.commit()
        return True

    @staticmethod
    def _partition(request: LabelPartitionRequest) -> MaterializationPartition:
        """Map a label request row to its domain projection."""
        return MaterializationPartition(
            id=PartitionId(request.partition_id),
            job_id=JobId(request.job_id),
            timestamp=PartitionTimestamp(request.timestamp),
            status=PartitionStatus(request.status),
            created_at=CreatedAt(request.created_at),
            updated_at=UpdatedAt(request.updated_at),
        )

    def _plan_inputs(
        self,
        name: ResourceName,
        version: ResourceVersion,
        timestamp: PartitionTimestamp,
    ) -> list[UUID]:
        """Resolve every completed dataset revision needed by one label partition."""
        dependencies = self.session.execute(
            select(LabelDatasetDependency, Dataset)
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
        ).tuples()
        planned_partition_ids: list[UUID] = []
        for dependency, dataset in dependencies:
            partition_interval = DatasetRepository.definition(
                dataset
            ).config.partition_type.partition_interval
            first_timestamp = timestamp - (
                partition_interval * dependency.num_lookback_partitions
            )
            last_timestamp = timestamp + (
                partition_interval * dependency.num_lookahead_partitions
            )
            expected_timestamps = {
                first_timestamp + (partition_interval * offset)
                for offset in range(
                    dependency.num_lookback_partitions
                    + dependency.num_lookahead_partitions
                    + 1
                )
            }
            revisions: dict[datetime, UUID] = dict(
                self.session.execute(
                    select(
                        DatasetPartitionRequest.timestamp,
                        DatasetPartition.partition_id,
                    )
                    .join(
                        DatasetPartition,
                        DatasetPartition.partition_id
                        == DatasetPartitionRequest.partition_id,
                    )
                    .where(
                        DatasetPartitionRequest.dataset_name
                        == dependency.dataset_name,
                        DatasetPartitionRequest.dataset_version
                        == dependency.dataset_version,
                        DatasetPartitionRequest.status
                        == PartitionStatus.CREATED.value,
                        DatasetPartitionRequest.timestamp >= first_timestamp,
                        DatasetPartitionRequest.timestamp <= last_timestamp,
                    )
                    .order_by(
                        DatasetPartitionRequest.timestamp,
                        DatasetPartition.created_at.desc(),
                        DatasetPartitionRequest.created_at.desc(),
                    )
                    .distinct(DatasetPartitionRequest.timestamp)
                )
                .tuples()
                .all()
            )
            missing_timestamps = sorted(
                item.isoformat()
                for item in expected_timestamps.difference(revisions)
            )
            if missing_timestamps:
                raise MissingLabelInputPartitionsError(
                    detail=(
                        "Label partition cannot be materialized because its "
                        f"required {dependency.dataset_name}:{dependency.dataset_version} "
                        f"input partitions are missing: {missing_timestamps}"
                    ),
                )
            planned_partition_ids.extend(
                revisions[item] for item in sorted(expected_timestamps)
            )
        return planned_partition_ids

    def _completed_request(
        self,
        name: ResourceName,
        version: ResourceVersion,
        timestamp: PartitionTimestamp,
    ) -> LabelPartitionRequest | None:
        """Return the newest completed revision for one label partition."""
        return self.session.scalars(
            select(LabelPartitionRequest)
            .join(
                LabelPartition,
                LabelPartition.partition_id
                == LabelPartitionRequest.partition_id,
            )
            .where(
                LabelPartitionRequest.label_name == name,
                LabelPartitionRequest.label_version == version,
                LabelPartitionRequest.timestamp == timestamp,
                LabelPartitionRequest.status == PartitionStatus.CREATED.value,
            )
            .order_by(
                LabelPartition.created_at.desc(),
                LabelPartitionRequest.created_at.desc(),
            )
            .limit(1)
        ).first()

    def _planned_ids(self, request_id: PartitionId) -> set[UUID]:
        """Return parent partition IDs pinned when a request was submitted."""
        return set(
            self.session.scalars(
                select(LabelPartitionPlan.dataset_partition_id).where(
                    LabelPartitionPlan.label_partition_request_id == request_id
                )
            )
        )

    def _completed_ids(self, partition_id: PartitionId) -> set[UUID]:
        """Return immutable parent partition IDs retained by a receipt."""
        return set(
            self.session.scalars(
                select(
                    LabelPartitionDatasetDependency.dataset_partition_id
                ).where(
                    LabelPartitionDatasetDependency.label_partition_id
                    == partition_id
                )
            )
        )

    def _validate_materialization(
        self, name: ResourceName, version: ResourceVersion
    ) -> Label:
        """Ensure the target and every declared input dataset are active."""
        resource = self.session.get(Label, (name, version))
        if resource is None or resource.deleted_at is not None:
            raise LabelVersionUnavailableError
        deleted_input = self.session.scalar(
            select(LabelDatasetDependency.label_name)
            .join(
                Dataset,
                (Dataset.name == LabelDatasetDependency.dataset_name)
                & (Dataset.version == LabelDatasetDependency.dataset_version),
            )
            .where(
                LabelDatasetDependency.label_name == name,
                LabelDatasetDependency.label_version == version,
                Dataset.deleted_at.is_not(None),
            )
            .limit(1)
        )
        if deleted_input is not None:
            raise DeletedLabelInputDatasetError
        return resource

    def materialize(
        self,
        name: ResourceName,
        version: ResourceVersion,
        payload: LabelPartitionMaterializationRequest,
    ) -> MaterializationRequest:
        """Create only stale or missing label partitions with pinned inputs."""
        label = self._validate_materialization(name, version)
        validate_partition_timestamps(
            name=name,
            version=version,
            resource_type=ResourceType.LABEL,
            partition_type=PartitionType(label.frequency),
            timestamps=payload.timestamps,
        )

        for timestamp in sorted(payload.timestamps):
            self.session.execute(
                select(
                    func.pg_advisory_xact_lock(
                        func.hashtextextended(
                            f"label:{name}:{version}:{timestamp.isoformat()}",
                            0,
                        )
                    )
                )
            )

        existing_requests: list[LabelPartitionRequest] = []
        planned_by_timestamp: dict[PartitionTimestamp, list[UUID]] = {}
        for timestamp in payload.timestamps:
            planned_partition_ids = self._plan_inputs(name, version, timestamp)
            completed = self._completed_request(name, version, timestamp)
            if completed is not None:
                completed_ids = self._completed_ids(
                    PartitionId(completed.partition_id)
                )
                is_current = completed_ids == set(
                    planned_partition_ids
                ) and bool(completed_ids)
                if payload.mode is FeatureMaterializationMode.CREATE_ONLY or (
                    payload.mode is FeatureMaterializationMode.ENSURE_CURRENT
                    and is_current
                ):
                    existing_requests.append(completed)
                    continue

            pending = self.session.scalars(
                select(LabelPartitionRequest)
                .where(
                    LabelPartitionRequest.label_name == name,
                    LabelPartitionRequest.label_version == version,
                    LabelPartitionRequest.timestamp == timestamp,
                    LabelPartitionRequest.status
                    == PartitionStatus.PENDING.value,
                )
                .order_by(LabelPartitionRequest.created_at.desc())
                .limit(1)
            ).first()
            if pending is not None:
                if self._planned_ids(PartitionId(pending.partition_id)) == set(
                    planned_partition_ids
                ):
                    existing_requests.append(pending)
                    continue
                raise LabelPartitionPlanConflictError
            planned_by_timestamp[timestamp] = planned_partition_ids

        if not planned_by_timestamp:
            self.session.commit()
            return MaterializationRequest(
                submitted_job=None,
                partitions=tuple(
                    self._partition(request) for request in existing_requests
                ),
            )

        job = Job(
            type=JobType.LABEL_MATERIALIZATION.value,
            status=JobStatus.SUBMITTED.value,
        )
        self.session.add(job)
        self.session.flush()
        requests: list[LabelPartitionRequest] = []
        for timestamp, planned_partition_ids in planned_by_timestamp.items():
            request = LabelPartitionRequest(
                job_id=job.id,
                label_name=name,
                label_version=version,
                timestamp=timestamp,
                status=PartitionStatus.PENDING.value,
                callback_token=secrets.token_urlsafe(32),
            )
            self.session.add(request)
            self.session.flush()
            self.session.add_all(
                LabelPartitionPlan(
                    label_partition_request_id=request.partition_id,
                    dataset_partition_id=dataset_partition_id,
                )
                for dataset_partition_id in planned_partition_ids
            )
            requests.append(request)
        self.session.commit()
        self.session.refresh(job)
        for request in requests:
            self.session.refresh(request)
        return MaterializationRequest(
            submitted_job=MaterializationJob(
                id=JobId(job.id),
                type=JobType(job.type),
                status=JobStatus(job.status),
                created_at=CreatedAt(job.created_at),
                updated_at=UpdatedAt(job.updated_at),
            ),
            partitions=tuple(
                self._partition(request)
                for request in [*existing_requests, *requests]
            ),
        )
