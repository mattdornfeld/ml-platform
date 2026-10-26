"""SQLAlchemy persistence operations used by the Kubernetes scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import exists, func, literal, select, tuple_, update
from sqlalchemy.orm import Session

from ml_platform.api._exceptions import UnsupportedScheduledJobTypeError
from ml_platform.api.domain import JobStatus, JobType, PartitionStatus
from ml_platform.api.infrastructure import engine_for
from ml_platform.api.repositories._datasets import DatasetRepository
from ml_platform.api.repositories._models import (
    Dataset,
    DatasetPartition,
    DatasetPartitionRequest,
    Feature,
    FeatureDatasetDependency,
    FeaturePartitionPlan,
    FeaturePartitionRequest,
    Job,
    Label,
    LabelDatasetDependency,
    LabelPartitionPlan,
    LabelPartitionRequest,
)
from ml_platform.api.repositories.exceptions import (
    DatasetMaterializationJobMultipleTargetsError,
    DatasetMaterializationJobWithoutPartitionsError,
    FeatureMaterializationJobMultipleTargetsError,
    FeatureMaterializationJobWithoutPartitionsError,
    LabelMaterializationJobMultipleTargetsError,
    LabelMaterializationJobWithoutPartitionsError,
    ScheduledMaterializationJobNotFoundError,
    UnsupportedScheduledRequestTypeError,
)

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.engine import Engine

    from ml_platform.common.types import DatabaseUrl, JobId


@dataclass(frozen=True)
class SchedulerRepository:
    """Read schedulable jobs and persist observed job outcomes."""

    database_url: DatabaseUrl

    @property
    def _engine(self) -> Engine:
        """Configured synchronous engine for short scheduler sessions."""
        return engine_for(self.database_url)

    @staticmethod
    def _job_type(job_type: str) -> JobType:
        """Resolve a persisted scheduler job type to its typed enum value."""
        try:
            return JobType(job_type)
        except ValueError as error:
            raise UnsupportedScheduledJobTypeError(job_type) from error

    @staticmethod
    def _session_job(job: Job) -> dict[str, object]:
        """Adapt a persisted scheduler job to the service's stable projection."""
        return {
            "id": job.id,
            "type": job.type,
            "created_at": job.created_at,
        }

    @staticmethod
    def _callbacks(
        requests: list[
            DatasetPartitionRequest
            | FeaturePartitionRequest
            | LabelPartitionRequest
        ],
        plans: dict[UUID, list[dict[str, object]]],
    ) -> list[dict[str, object]]:
        """Adapt ordered partition requests into materializer callback config."""
        callbacks: list[dict[str, object]] = []
        for request in requests:
            callback = {
                "partition_id": request.partition_id,
                "timestamp": request.timestamp.isoformat(),
                "callback_token": request.callback_token,
            }
            if not isinstance(request, DatasetPartitionRequest):
                callback["partition_dependencies"] = plans[request.partition_id]
            callbacks.append(callback)
        return callbacks

    @staticmethod
    def _plans(
        session: Session,
        job_id: JobId,
        request_type: type[FeaturePartitionRequest]
        | type[LabelPartitionRequest],
    ) -> dict[UUID, list[dict[str, object]]]:
        """Load every pinned dataset receipt required by a derived job."""
        # SQLAlchemy mapped attributes are carried through the explicit branches
        # below so each query retains its typed, static schema.
        if request_type is FeaturePartitionRequest:
            rows = session.execute(
                select(
                    FeaturePartitionPlan.feature_partition_request_id,
                    FeaturePartitionPlan.dataset_partition_id,
                    DatasetPartition.snapshot_id,
                    DatasetPartitionRequest.dataset_name,
                    DatasetPartitionRequest.dataset_version,
                    DatasetPartitionRequest.timestamp,
                )
                .join(
                    DatasetPartition,
                    DatasetPartition.partition_id
                    == FeaturePartitionPlan.dataset_partition_id,
                )
                .join(
                    DatasetPartitionRequest,
                    DatasetPartitionRequest.partition_id
                    == DatasetPartition.partition_id,
                )
                .join(
                    FeaturePartitionRequest,
                    FeaturePartitionRequest.partition_id
                    == FeaturePartitionPlan.feature_partition_request_id,
                )
                .where(FeaturePartitionRequest.job_id == job_id)
                .order_by(
                    FeaturePartitionPlan.feature_partition_request_id,
                    DatasetPartitionRequest.timestamp,
                    FeaturePartitionPlan.dataset_partition_id,
                )
            )
        elif request_type is LabelPartitionRequest:
            rows = session.execute(
                select(
                    LabelPartitionPlan.label_partition_request_id,
                    LabelPartitionPlan.dataset_partition_id,
                    DatasetPartition.snapshot_id,
                    DatasetPartitionRequest.dataset_name,
                    DatasetPartitionRequest.dataset_version,
                    DatasetPartitionRequest.timestamp,
                )
                .join(
                    DatasetPartition,
                    DatasetPartition.partition_id
                    == LabelPartitionPlan.dataset_partition_id,
                )
                .join(
                    DatasetPartitionRequest,
                    DatasetPartitionRequest.partition_id
                    == DatasetPartition.partition_id,
                )
                .join(
                    LabelPartitionRequest,
                    LabelPartitionRequest.partition_id
                    == LabelPartitionPlan.label_partition_request_id,
                )
                .where(LabelPartitionRequest.job_id == job_id)
                .order_by(
                    LabelPartitionPlan.label_partition_request_id,
                    DatasetPartitionRequest.timestamp,
                    LabelPartitionPlan.dataset_partition_id,
                )
            )
        else:
            raise UnsupportedScheduledRequestTypeError(request_type.__name__)
        plan_by_request: dict[UUID, list[dict[str, object]]] = {}
        for (
            partition_request_id,
            dataset_partition_id,
            snapshot_id,
            dataset_name,
            dataset_version,
            timestamp,
        ) in rows:
            plan_by_request.setdefault(partition_request_id, []).append(
                {
                    "partition_id": dataset_partition_id,
                    "dataset_name": dataset_name,
                    "dataset_version": dataset_version,
                    "timestamp": timestamp.isoformat(),
                    "snapshot_id": snapshot_id,
                }
            )
        return plan_by_request

    def submitted_jobs(self) -> list[dict[str, object]]:
        """Return active jobs in deterministic API submission order."""
        dataset_deleted = (
            select(DatasetPartitionRequest.partition_id)
            .join(
                Dataset,
                (Dataset.name == DatasetPartitionRequest.dataset_name)
                & (Dataset.version == DatasetPartitionRequest.dataset_version),
            )
            .where(
                DatasetPartitionRequest.job_id == Job.id,
                Dataset.deleted_at.is_not(None),
            )
        )
        feature_deleted = (
            select(FeaturePartitionRequest.partition_id)
            .join(
                Feature,
                (Feature.name == FeaturePartitionRequest.feature_name)
                & (Feature.version == FeaturePartitionRequest.feature_version),
            )
            .where(
                FeaturePartitionRequest.job_id == Job.id,
                Feature.deleted_at.is_not(None),
            )
        )
        label_deleted = (
            select(LabelPartitionRequest.partition_id)
            .join(
                Label,
                (Label.name == LabelPartitionRequest.label_name)
                & (Label.version == LabelPartitionRequest.label_version),
            )
            .where(
                LabelPartitionRequest.job_id == Job.id,
                Label.deleted_at.is_not(None),
            )
        )
        statement = (
            select(Job)
            .where(
                Job.type.in_(tuple(item.value for item in JobType)),
                Job.status == JobStatus.SUBMITTED.value,
                ~exists(dataset_deleted),
                ~exists(feature_deleted),
                ~exists(label_deleted),
            )
            .order_by(Job.created_at, Job.id)
        )
        with Session(self._engine) as session:
            return [
                self._session_job(job) for job in session.scalars(statement)
            ]

    def dataset_job(self, job_id: JobId) -> dict[str, object]:
        """Load immutable dataset config and callbacks for one job."""
        statement = (
            select(DatasetPartitionRequest, Dataset)
            .join(
                Dataset,
                (Dataset.name == DatasetPartitionRequest.dataset_name)
                & (Dataset.version == DatasetPartitionRequest.dataset_version),
            )
            .where(DatasetPartitionRequest.job_id == job_id)
            .order_by(DatasetPartitionRequest.timestamp)
        )
        with Session(self._engine) as session:
            rows = list(session.execute(statement))
        if not rows:
            raise DatasetMaterializationJobWithoutPartitionsError(str(job_id))
        first_request, dataset = rows[0]
        if any(
            request.dataset_name != first_request.dataset_name
            or request.dataset_version != first_request.dataset_version
            for request, _ in rows
        ):
            raise DatasetMaterializationJobMultipleTargetsError(str(job_id))
        return {
            "name": dataset.name,
            "version": dataset.version,
            "dataset_type": dataset.dataset_type,
            "config": DatasetRepository.definition(dataset).config.model_dump(
                mode="json"
            ),
            "callbacks": self._callbacks([request for request, _ in rows], {}),
        }

    def feature_job(self, job_id: JobId) -> dict[str, object]:
        """Load immutable feature metadata, dependencies, and callbacks."""
        target_statement = (
            select(FeaturePartitionRequest, Feature)
            .join(
                Feature,
                (Feature.name == FeaturePartitionRequest.feature_name)
                & (Feature.version == FeaturePartitionRequest.feature_version),
            )
            .where(FeaturePartitionRequest.job_id == job_id)
            .order_by(FeaturePartitionRequest.timestamp)
        )
        with Session(self._engine) as session:
            rows = list(session.execute(target_statement))
            if not rows:
                raise FeatureMaterializationJobWithoutPartitionsError(
                    str(job_id)
                )
            first_request, feature = rows[0]
            dependencies = list(
                session.execute(
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
                        FeatureDatasetDependency.feature_name == feature.name,
                        FeatureDatasetDependency.feature_version
                        == feature.version,
                    )
                    .order_by(
                        FeatureDatasetDependency.dataset_name,
                        FeatureDatasetDependency.dataset_version,
                    )
                )
            )
            plans = self._plans(
                session,
                job_id,
                FeaturePartitionRequest,
            )
        if any(
            request.feature_name != first_request.feature_name
            or request.feature_version != first_request.feature_version
            for request, _ in rows
        ):
            raise FeatureMaterializationJobMultipleTargetsError(str(job_id))
        return {
            "config": {
                "name": feature.name,
                "version": feature.version,
                "dtype": feature.dtype,
                "frequency": feature.frequency,
                "keys": feature.keys,
                "validations": feature.validations,
                "transform": feature.transform,
                "description": feature.description,
                "parameters": feature.parameters,
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
            },
            "callbacks": self._callbacks(
                [request for request, _ in rows], plans
            ),
        }

    def label_job(self, job_id: JobId) -> dict[str, object]:
        """Load immutable label metadata, dependencies, and callbacks."""
        target_statement = (
            select(LabelPartitionRequest, Label)
            .join(
                Label,
                (Label.name == LabelPartitionRequest.label_name)
                & (Label.version == LabelPartitionRequest.label_version),
            )
            .where(LabelPartitionRequest.job_id == job_id)
            .order_by(LabelPartitionRequest.timestamp)
        )
        with Session(self._engine) as session:
            rows = list(session.execute(target_statement))
            if not rows:
                raise LabelMaterializationJobWithoutPartitionsError(str(job_id))
            first_request, label = rows[0]
            dependencies = list(
                session.execute(
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
                        LabelDatasetDependency.label_name == label.name,
                        LabelDatasetDependency.label_version == label.version,
                    )
                    .order_by(
                        LabelDatasetDependency.dataset_name,
                        LabelDatasetDependency.dataset_version,
                    )
                )
            )
            plans = self._plans(
                session,
                job_id,
                LabelPartitionRequest,
            )
        if any(
            request.label_name != first_request.label_name
            or request.label_version != first_request.label_version
            for request, _ in rows
        ):
            raise LabelMaterializationJobMultipleTargetsError(str(job_id))
        return {
            "config": {
                "name": label.name,
                "version": label.version,
                "dtype": label.dtype,
                "frequency": label.frequency,
                "keys": label.keys,
                "validations": label.validations,
                "transform": label.transform,
                "description": label.description,
                "parameters": label.parameters,
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
            },
            "callbacks": self._callbacks(
                [request for request, _ in rows], plans
            ),
        }

    def is_eligible(self, job_id: JobId, job_type: str) -> bool:
        """Allow only the oldest submitted job for every partition it targets."""
        match self._job_type(job_type):
            case JobType.DATASET_MATERIALIZATION:
                request_table = DatasetPartitionRequest.__table__
                name_column = request_table.c.dataset_name
                version_column = request_table.c.dataset_version
            case JobType.FEATURE_MATERIALIZATION:
                request_table = FeaturePartitionRequest.__table__
                name_column = request_table.c.feature_name
                version_column = request_table.c.feature_version
            case JobType.LABEL_MATERIALIZATION:
                request_table = LabelPartitionRequest.__table__
                name_column = request_table.c.label_name
                version_column = request_table.c.label_version
            case _ as unsupported_job_type:
                raise UnsupportedScheduledJobTypeError(
                    str(unsupported_job_type)
                )
        older_request = request_table.alias("older_request")
        jobs_table = Job.__table__
        older_job = jobs_table.alias("older_job")
        with Session(self._engine) as session:
            current_job = session.get(Job, job_id)
            if current_job is None:
                raise ScheduledMaterializationJobNotFoundError(str(job_id))
            targets = session.execute(
                select(name_column, version_column, request_table.c.timestamp)
                .where(request_table.c.job_id == job_id)
                .order_by(
                    name_column, version_column, request_table.c.timestamp
                )
            )
            for name, version, timestamp in targets:
                session.execute(
                    select(
                        func.pg_advisory_xact_lock(
                            func.hashtextextended(
                                f"{job_type}:{name}:{version}:{timestamp.isoformat()}",
                                0,
                            )
                        )
                    )
                )
                has_older_conflict = session.scalar(
                    select(
                        exists(
                            select(older_request.c.partition_id)
                            .select_from(
                                older_request.join(
                                    older_job,
                                    older_job.c.id == older_request.c.job_id,
                                )
                            )
                            .where(
                                older_request.c[name_column.name] == name,
                                older_request.c[version_column.name] == version,
                                older_request.c.timestamp == timestamp,
                                older_job.c.status == JobStatus.SUBMITTED.value,
                                tuple_(
                                    older_job.c.created_at,
                                    older_job.c.id,
                                )
                                < tuple_(
                                    literal(current_job.created_at),
                                    literal(current_job.id),
                                ),
                            )
                        )
                    )
                )
                if has_older_conflict:
                    return False
        return True

    def finish(self, job_id: JobId, outcome: str, job_type: str) -> None:
        """Persist a terminal outcome after confirming all receipts exist."""
        match self._job_type(job_type):
            case JobType.DATASET_MATERIALIZATION:
                request_table = DatasetPartitionRequest.__table__
            case JobType.FEATURE_MATERIALIZATION:
                request_table = FeaturePartitionRequest.__table__
            case JobType.LABEL_MATERIALIZATION:
                request_table = LabelPartitionRequest.__table__
            case _ as unsupported_job_type:
                raise UnsupportedScheduledJobTypeError(
                    str(unsupported_job_type)
                )
        with Session(self._engine) as session:
            job = session.get(Job, job_id)
            if job is None or job.status != JobStatus.SUBMITTED.value:
                return
            terminal_status = JobStatus(outcome)
            if (
                terminal_status is JobStatus.SUCCEEDED
                and session.scalar(
                    select(request_table.c.partition_id)
                    .where(
                        request_table.c.job_id == job_id,
                        request_table.c.status != PartitionStatus.CREATED.value,
                    )
                    .limit(1)
                )
                is not None
            ):
                terminal_status = JobStatus.FAILED
            if terminal_status in {JobStatus.FAILED, JobStatus.CANCELLED}:
                partition_status = (
                    PartitionStatus.FAILED
                    if terminal_status is JobStatus.FAILED
                    else PartitionStatus.CANCELLED
                )
                match self._job_type(job_type):
                    case JobType.DATASET_MATERIALIZATION:
                        session.execute(
                            update(DatasetPartitionRequest)
                            .where(
                                DatasetPartitionRequest.job_id == job_id,
                                DatasetPartitionRequest.status
                                == PartitionStatus.PENDING.value,
                            )
                            .values(
                                status=partition_status.value,
                                updated_at=func.now(),
                            )
                        )
                    case JobType.FEATURE_MATERIALIZATION:
                        session.execute(
                            update(FeaturePartitionRequest)
                            .where(
                                FeaturePartitionRequest.job_id == job_id,
                                FeaturePartitionRequest.status
                                == PartitionStatus.PENDING.value,
                            )
                            .values(
                                status=partition_status.value,
                                updated_at=func.now(),
                            )
                        )
                    case JobType.LABEL_MATERIALIZATION:
                        session.execute(
                            update(LabelPartitionRequest)
                            .where(
                                LabelPartitionRequest.job_id == job_id,
                                LabelPartitionRequest.status
                                == PartitionStatus.PENDING.value,
                            )
                            .values(
                                status=partition_status.value,
                                updated_at=func.now(),
                            )
                        )
                    case _ as unsupported_job_type:
                        raise UnsupportedScheduledJobTypeError(
                            str(unsupported_job_type)
                        )
            job.status = terminal_status.value
            session.commit()
