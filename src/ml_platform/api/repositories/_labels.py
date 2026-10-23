"""SQLAlchemy persistence operations for label materialization."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from fastapi import HTTPException
from sqlalchemy import func, select

from ml_platform.api.domain import (
    JobStatus,
    JobType,
    MaterializationPartition,
    MaterializationRequest,
    PartitionStatus,
)
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
from ml_platform.common.api.dtos import (
    FeatureMaterializationMode,
    LabelPartitionMaterializationRequest,
)
from ml_platform.common.types import (
    CreatedAt,
    JobId,
    PartitionId,
    PartitionTimestamp,
    ResourceName,
    ResourceVersion,
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
        dependencies = self.session.scalars(
            select(LabelDatasetDependency)
            .where(
                LabelDatasetDependency.label_name == name,
                LabelDatasetDependency.label_version == version,
            )
            .order_by(
                LabelDatasetDependency.dataset_name,
                LabelDatasetDependency.dataset_version,
            )
        )
        planned_partition_ids: list[UUID] = []
        for dependency in dependencies:
            first_timestamp = timestamp - timedelta(
                days=dependency.num_lookback_partitions
            )
            last_timestamp = timestamp + timedelta(
                days=dependency.num_lookahead_partitions
            )
            expected_timestamps = {
                first_timestamp + timedelta(days=offset)
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
                raise HTTPException(
                    status_code=409,
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
    ) -> None:
        """Ensure the target and every declared input dataset are active."""
        resource = self.session.get(Label, (name, version))
        if resource is None or resource.deleted_at is not None:
            raise HTTPException(
                status_code=404, detail="Label version does not exist."
            )
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
            raise HTTPException(
                status_code=409,
                detail=(
                    "Label partition cannot be materialized because an input "
                    "dataset version is deleted."
                ),
            )

    def materialize(
        self,
        name: ResourceName,
        version: ResourceVersion,
        payload: LabelPartitionMaterializationRequest,
    ) -> MaterializationRequest:
        """Create only stale or missing label partitions with pinned inputs."""
        self._validate_materialization(name, version)

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
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Label partition is already being materialized against a "
                        "different input revision plan."
                    ),
                )
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
        from ml_platform.api.domain import MaterializationJob

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
