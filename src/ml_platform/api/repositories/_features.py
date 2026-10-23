"""SQLAlchemy persistence operations for feature materialization."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
from ml_platform.api.repositories._jobs import JobRepository
from ml_platform.api.repositories._models import (
    Dataset,
    DatasetPartition,
    DatasetPartitionRequest,
    Feature,
    FeatureDatasetDependency,
    FeaturePartition,
    FeaturePartitionDatasetDependency,
    FeaturePartitionPlan,
    FeaturePartitionRequest,
    Job,
)
from ml_platform.api.repositories.exceptions import (
    DeletedFeatureInputDatasetError,
    FeatureDependencyDatasetTypeMismatchError,
    FeatureDependencyDatasetUnavailableError,
    FeaturePartitionPlanConflictError,
    FeatureVersionExistsError,
    FeatureVersionUnavailableError,
    MissingFeatureInputPartitionsError,
)
from ml_platform.common.api.dtos import (
    FeatureCreateRequest,
    FeatureMaterializationMode,
    FeaturePartitionMaterializationRequest,
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

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.orm import Session


@dataclass(frozen=True)
class FeatureRepository:
    """Persist immutable feature input plans and materialization requests."""

    session: Session

    @staticmethod
    def _definition(feature: Feature) -> TransformDefinition:
        """Map a feature ORM row to its immutable domain definition."""
        lifecycle = (
            ResourceLifecycle.DELETED
            if feature.deleted_at is not None
            else ResourceLifecycle.DEPRECATED
            if feature.deprecated_at is not None
            else ResourceLifecycle.ACTIVE
        )
        return TransformDefinition(
            name=ResourceName(feature.name),
            version=ResourceVersion(feature.version),
            dtype=feature.dtype,
            frequency=feature.frequency,
            keys=tuple(feature.keys),
            validations=tuple(feature.validations),
            transform=feature.transform,
            description=Description(feature.description),
            parameters=feature.parameters,
            created_at=CreatedAt(feature.created_at),
            lifecycle=lifecycle,
            deleted_at=feature.deleted_at,
            deprecated_at=feature.deprecated_at,
        )

    @staticmethod
    def _resource_partition(
        request: FeaturePartitionRequest,
        receipt: FeaturePartition | None,
    ) -> ResourcePartition:
        """Map a feature request and receipt to its latest partition state."""
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
        """Return feature versions with their latest logical partition counts."""
        counts = (
            select(
                FeaturePartitionRequest.feature_name.label("name"),
                FeaturePartitionRequest.feature_version.label("version"),
                func.count(FeaturePartitionRequest.partition_id).label("total"),
                func.count(
                    case(
                        (
                            FeaturePartitionRequest.status
                            == PartitionStatus.CREATED.value,
                            FeaturePartitionRequest.partition_id,
                        )
                    )
                ).label("created"),
                func.count(
                    case(
                        (
                            FeaturePartitionRequest.status
                            == PartitionStatus.PENDING.value,
                            FeaturePartitionRequest.partition_id,
                        )
                    )
                ).label("pending"),
                func.count(
                    case(
                        (
                            FeaturePartitionRequest.status.in_(
                                (
                                    PartitionStatus.FAILED.value,
                                    PartitionStatus.CANCELLED.value,
                                )
                            ),
                            FeaturePartitionRequest.partition_id,
                        )
                    )
                ).label("failed"),
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
                counts.c.total,
                counts.c.created,
                counts.c.pending,
                counts.c.failed,
            )
            .outerjoin(
                counts,
                (counts.c.name == Feature.name)
                & (counts.c.version == Feature.version),
            )
            .order_by(Feature.name, Feature.version.desc())
        )
        if not include_deleted:
            statement = statement.where(Feature.deleted_at.is_(None))
        return tuple(
            TransformSummary(
                definition=self._definition(feature),
                partitions=PartitionCounts(
                    total=total or 0,
                    created=created or 0,
                    pending=pending or 0,
                    failed=failed or 0,
                ),
            )
            for feature, total, created, pending, failed in self.session.execute(
                statement
            )
        )

    def detail(
        self, name: ResourceName, version: ResourceVersion
    ) -> TransformDetail | None:
        """Return a feature with its dependencies, latest partitions, and jobs."""
        feature = self.session.get(Feature, (name, version))
        if feature is None:
            return None
        dependencies = tuple(
            TransformDatasetDependency(
                name=dependency.dataset_name,
                version=dependency.dataset_version,
                dataset_type=dataset_type,
                lookback_partitions=dependency.num_lookback_partitions,
                lookahead_partitions=0,
            )
            for dependency, dataset_type in self.session.execute(
                select(FeatureDatasetDependency, Dataset.dataset_type)
                .join(
                    Dataset,
                    (Dataset.name == FeatureDatasetDependency.dataset_name)
                    & (
                        Dataset.version
                        == FeatureDatasetDependency.dataset_version
                    ),
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
        )
        latest = (
            select(
                FeaturePartitionRequest.timestamp,
                func.max(FeaturePartitionRequest.created_at).label(
                    "created_at"
                ),
            )
            .where(
                FeaturePartitionRequest.feature_name == name,
                FeaturePartitionRequest.feature_version == version,
            )
            .group_by(FeaturePartitionRequest.timestamp)
            .subquery()
        )
        partitions = tuple(
            self._resource_partition(request, receipt)
            for request, receipt in self.session.execute(
                select(FeaturePartitionRequest, FeaturePartition)
                .join(
                    latest,
                    (latest.c.timestamp == FeaturePartitionRequest.timestamp)
                    & (
                        latest.c.created_at
                        == FeaturePartitionRequest.created_at
                    ),
                )
                .outerjoin(
                    FeaturePartition,
                    FeaturePartition.partition_id
                    == FeaturePartitionRequest.partition_id,
                )
                .order_by(FeaturePartitionRequest.timestamp.desc())
            )
        )
        return TransformDetail(
            definition=self._definition(feature),
            datasets=dependencies,
            partitions=partitions,
            jobs=JobRepository(self.session).for_resource(
                ResourceType.FEATURE, name, version
            ),
        )

    def create(self, payload: FeatureCreateRequest) -> TransformDefinition:
        """Register a feature after validating every declared dataset dependency."""
        for dependency in payload.datasets:
            dataset = self.session.get(
                Dataset, (dependency.name, dependency.version)
            )
            if dataset is None or dataset.deleted_at is not None:
                raise FeatureDependencyDatasetUnavailableError
            if dataset.dataset_type != dependency.dataset_type.value:
                raise FeatureDependencyDatasetTypeMismatchError
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
            self.session.add(feature)
            self.session.add_all(
                FeatureDatasetDependency(
                    feature_name=payload.name,
                    feature_version=payload.version,
                    dataset_name=dependency.name,
                    dataset_version=dependency.version,
                    num_lookback_partitions=dependency.input_window.lookback_partitions,
                )
                for dependency in payload.datasets
            )
            self.session.commit()
        except IntegrityError as error:
            self.session.rollback()
            raise FeatureVersionExistsError from error
        self.session.refresh(feature)
        return self._definition(feature)

    def delete(self, name: ResourceName, version: ResourceVersion) -> bool:
        """Soft-delete a feature and cancel its unfinished materializations."""
        feature = self.session.get(Feature, (name, version))
        if feature is None:
            return False
        if feature.deleted_at is None:
            feature.deleted_at = datetime.now(UTC)
        pending_job_ids = select(FeaturePartitionRequest.job_id).where(
            FeaturePartitionRequest.feature_name == name,
            FeaturePartitionRequest.feature_version == version,
            FeaturePartitionRequest.status == PartitionStatus.PENDING.value,
        )
        self.session.execute(
            update(FeaturePartitionRequest)
            .where(
                FeaturePartitionRequest.feature_name == name,
                FeaturePartitionRequest.feature_version == version,
                FeaturePartitionRequest.status == PartitionStatus.PENDING.value,
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
    def _partition(
        request: FeaturePartitionRequest,
    ) -> MaterializationPartition:
        """Map a feature request row to its domain projection."""
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
        """Resolve every completed dataset revision needed by one partition."""
        dependencies = self.session.scalars(
            select(FeatureDatasetDependency)
            .where(
                FeatureDatasetDependency.feature_name == name,
                FeatureDatasetDependency.feature_version == version,
            )
            .order_by(
                FeatureDatasetDependency.dataset_name,
                FeatureDatasetDependency.dataset_version,
            )
        )
        planned_partition_ids: list[UUID] = []
        for dependency in dependencies:
            first_timestamp = timestamp - timedelta(
                days=dependency.num_lookback_partitions
            )
            expected_timestamps = {
                first_timestamp + timedelta(days=offset)
                for offset in range(dependency.num_lookback_partitions + 1)
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
                        DatasetPartitionRequest.timestamp <= timestamp,
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
            missing_timestamps = tuple(
                sorted(expected_timestamps.difference(revisions))
            )
            if missing_timestamps:
                raise MissingFeatureInputPartitionsError(
                    dataset_name=dependency.dataset_name,
                    dataset_version=dependency.dataset_version,
                    missing_timestamps=missing_timestamps,
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
    ) -> FeaturePartitionRequest | None:
        """Return the newest completed revision for one feature partition."""
        return self.session.scalars(
            select(FeaturePartitionRequest)
            .join(
                FeaturePartition,
                FeaturePartition.partition_id
                == FeaturePartitionRequest.partition_id,
            )
            .where(
                FeaturePartitionRequest.feature_name == name,
                FeaturePartitionRequest.feature_version == version,
                FeaturePartitionRequest.timestamp == timestamp,
                FeaturePartitionRequest.status == PartitionStatus.CREATED.value,
            )
            .order_by(
                FeaturePartition.created_at.desc(),
                FeaturePartitionRequest.created_at.desc(),
            )
            .limit(1)
        ).first()

    def _planned_ids(self, request_id: PartitionId) -> set[UUID]:
        """Return parent partition IDs pinned when a request was submitted."""
        return set(
            self.session.scalars(
                select(FeaturePartitionPlan.dataset_partition_id).where(
                    FeaturePartitionPlan.feature_partition_request_id
                    == request_id
                )
            )
        )

    def _completed_ids(self, partition_id: PartitionId) -> set[UUID]:
        """Return immutable parent partition IDs retained by a receipt."""
        return set(
            self.session.scalars(
                select(
                    FeaturePartitionDatasetDependency.dataset_partition_id
                ).where(
                    FeaturePartitionDatasetDependency.feature_partition_id
                    == partition_id
                )
            )
        )

    def _validate_materialization(
        self, name: ResourceName, version: ResourceVersion
    ) -> None:
        """Ensure the target and every declared input dataset are active."""
        resource = self.session.get(Feature, (name, version))
        if resource is None or resource.deleted_at is not None:
            raise FeatureVersionUnavailableError
        deleted_input = self.session.scalar(
            select(FeatureDatasetDependency.feature_name)
            .join(
                Dataset,
                (Dataset.name == FeatureDatasetDependency.dataset_name)
                & (Dataset.version == FeatureDatasetDependency.dataset_version),
            )
            .where(
                FeatureDatasetDependency.feature_name == name,
                FeatureDatasetDependency.feature_version == version,
                Dataset.deleted_at.is_not(None),
            )
            .limit(1)
        )
        if deleted_input is not None:
            raise DeletedFeatureInputDatasetError

    def materialize(
        self,
        name: ResourceName,
        version: ResourceVersion,
        payload: FeaturePartitionMaterializationRequest,
    ) -> MaterializationRequest:
        """Create only stale or missing feature partitions with pinned inputs."""
        self._validate_materialization(name, version)

        for timestamp in sorted(payload.timestamps):
            self.session.execute(
                select(
                    func.pg_advisory_xact_lock(
                        func.hashtextextended(
                            f"feature:{name}:{version}:{timestamp.isoformat()}",
                            0,
                        )
                    )
                )
            )

        existing_requests: list[FeaturePartitionRequest] = []
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
                select(FeaturePartitionRequest)
                .where(
                    FeaturePartitionRequest.feature_name == name,
                    FeaturePartitionRequest.feature_version == version,
                    FeaturePartitionRequest.timestamp == timestamp,
                    FeaturePartitionRequest.status
                    == PartitionStatus.PENDING.value,
                )
                .order_by(FeaturePartitionRequest.created_at.desc())
                .limit(1)
            ).first()
            if pending is not None:
                if self._planned_ids(PartitionId(pending.partition_id)) == set(
                    planned_partition_ids
                ):
                    existing_requests.append(pending)
                    continue
                raise FeaturePartitionPlanConflictError
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
            type=JobType.FEATURE_MATERIALIZATION.value,
            status=JobStatus.SUBMITTED.value,
        )
        self.session.add(job)
        self.session.flush()
        requests: list[FeaturePartitionRequest] = []
        for timestamp, planned_partition_ids in planned_by_timestamp.items():
            request = FeaturePartitionRequest(
                job_id=job.id,
                feature_name=name,
                feature_version=version,
                timestamp=timestamp,
                status=PartitionStatus.PENDING.value,
                callback_token=secrets.token_urlsafe(32),
            )
            self.session.add(request)
            self.session.flush()
            self.session.add_all(
                FeaturePartitionPlan(
                    feature_partition_request_id=request.partition_id,
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
