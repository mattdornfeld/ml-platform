"""SQLAlchemy persistence operations for atomic partition completion."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from sqlalchemy import select

from ml_platform.api.domain import JobStatus, PartitionStatus
from ml_platform.api.repositories._models import (
    Dataset,
    DatasetPartition,
    DatasetPartitionRequest,
    Feature,
    FeaturePartition,
    FeaturePartitionDatasetDependency,
    FeaturePartitionPlan,
    FeaturePartitionRequest,
    Job,
    Label,
    LabelPartition,
    LabelPartitionDatasetDependency,
    LabelPartitionPlan,
    LabelPartitionRequest,
)
from ml_platform.api.repositories.exceptions import (
    CompletionBatchSchemaConflictError,
    DeletedCompletionResourceError,
    DuplicateCompletionPartitionIdError,
    ExistingReceiptSchemaConflictError,
    InactivePartitionJobError,
    InvalidPartitionCallbackTokenError,
    MissingCompletedPartitionReceiptError,
    PartitionDependencyPlanMismatchError,
    PartitionReceiptConflictError,
    UnsupportedPartitionCompletionResourceTypeError,
    UnsupportedPartitionRequestStatusError,
)
from ml_platform.common.api.dtos import (
    PartitionCompletionRequest,
    PartitionResultResponse,
    ResourceType,
)

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.orm import Session

    from ml_platform.common.types import PartitionId


class PartitionRequest(Protocol):
    """Structural contract needed to validate a partition request lifecycle."""

    status: str


class Resource(Protocol):
    """Structural contract needed to validate a resource lifecycle."""

    deleted_at: datetime | None


@dataclass(frozen=True)
class PartitionRepository:
    """Persist all receipt records atomically through one SQLAlchemy session."""

    session: Session

    def _planned_ids(
        self,
        resource_type: ResourceType,
        partition_id: PartitionId,
    ) -> set[UUID]:
        """Return pinned dataset revisions from a derived partition request."""
        match resource_type:
            case ResourceType.FEATURE:
                statement = select(
                    FeaturePartitionPlan.dataset_partition_id
                ).where(
                    FeaturePartitionPlan.feature_partition_request_id
                    == partition_id
                )
            case ResourceType.LABEL:
                statement = select(
                    LabelPartitionPlan.dataset_partition_id
                ).where(
                    LabelPartitionPlan.label_partition_request_id
                    == partition_id
                )
            case ResourceType.DATASET:
                return set()
            case _ as unsupported_resource_type:
                raise UnsupportedPartitionCompletionResourceTypeError(
                    str(unsupported_resource_type)
                )
        return set(self.session.scalars(statement))

    def _request(
        self,
        item: PartitionCompletionRequest,
    ) -> tuple[
        DatasetPartitionRequest
        | FeaturePartitionRequest
        | LabelPartitionRequest,
        Dataset | Feature | Label,
        Job,
    ]:
        """Lock and load one partition request with its definition and job."""
        match item.resource_type:
            case ResourceType.DATASET:
                request, resource, job = self.session.execute(
                    select(DatasetPartitionRequest, Dataset, Job)
                    .join(
                        Dataset,
                        (Dataset.name == DatasetPartitionRequest.dataset_name)
                        & (
                            Dataset.version
                            == DatasetPartitionRequest.dataset_version
                        ),
                    )
                    .join(Job, Job.id == DatasetPartitionRequest.job_id)
                    .where(
                        DatasetPartitionRequest.partition_id
                        == item.partition_id
                    )
                    .with_for_update(of=(DatasetPartitionRequest, Dataset))
                ).one()
                return request, resource, job
            case ResourceType.FEATURE:
                request, resource, job = self.session.execute(
                    select(FeaturePartitionRequest, Feature, Job)
                    .join(
                        Feature,
                        (Feature.name == FeaturePartitionRequest.feature_name)
                        & (
                            Feature.version
                            == FeaturePartitionRequest.feature_version
                        ),
                    )
                    .join(Job, Job.id == FeaturePartitionRequest.job_id)
                    .where(
                        FeaturePartitionRequest.partition_id
                        == item.partition_id
                    )
                    .with_for_update(of=(FeaturePartitionRequest, Feature))
                ).one()
                return request, resource, job
            case ResourceType.LABEL:
                request, resource, job = self.session.execute(
                    select(LabelPartitionRequest, Label, Job)
                    .join(
                        Label,
                        (Label.name == LabelPartitionRequest.label_name)
                        & (
                            Label.version == LabelPartitionRequest.label_version
                        ),
                    )
                    .join(Job, Job.id == LabelPartitionRequest.job_id)
                    .where(
                        LabelPartitionRequest.partition_id == item.partition_id
                    )
                    .with_for_update(of=(LabelPartitionRequest, Label))
                ).one()
                return request, resource, job
            case _ as unsupported_resource_type:
                raise UnsupportedPartitionCompletionResourceTypeError(
                    str(unsupported_resource_type)
                )

    def _existing_schemas(
        self,
        item: PartitionCompletionRequest,
        name: str,
        version: int,
    ) -> set[tuple[tuple[str, str], ...]]:
        """Return persisted schemas for the Iceberg table owning one receipt."""
        match item.resource_type:
            case ResourceType.DATASET:
                statement = (
                    select(DatasetPartition.iceberg_schema)
                    .join(
                        DatasetPartitionRequest,
                        DatasetPartitionRequest.partition_id
                        == DatasetPartition.partition_id,
                    )
                    .where(
                        DatasetPartitionRequest.dataset_name == name,
                        DatasetPartitionRequest.dataset_version == version,
                    )
                )
            case ResourceType.FEATURE:
                statement = (
                    select(FeaturePartition.iceberg_schema)
                    .join(
                        FeaturePartitionRequest,
                        FeaturePartitionRequest.partition_id
                        == FeaturePartition.partition_id,
                    )
                    .where(
                        FeaturePartitionRequest.feature_name == name,
                        FeaturePartitionRequest.feature_version == version,
                    )
                )
            case ResourceType.LABEL:
                statement = (
                    select(LabelPartition.iceberg_schema)
                    .join(
                        LabelPartitionRequest,
                        LabelPartitionRequest.partition_id
                        == LabelPartition.partition_id,
                    )
                    .where(
                        LabelPartitionRequest.label_name == name,
                        LabelPartitionRequest.label_version == version,
                    )
                )
            case _ as unsupported_resource_type:
                raise UnsupportedPartitionCompletionResourceTypeError(
                    str(unsupported_resource_type)
                )
        return {
            tuple(sorted(schema.items()))
            for schema in self.session.scalars(statement)
        }

    def _receipt(
        self,
        item: PartitionCompletionRequest,
    ) -> DatasetPartition | FeaturePartition | LabelPartition | None:
        """Load an existing completion receipt for the request, if any."""
        match item.resource_type:
            case ResourceType.DATASET:
                return self.session.get(DatasetPartition, item.partition_id)
            case ResourceType.FEATURE:
                return self.session.get(FeaturePartition, item.partition_id)
            case ResourceType.LABEL:
                return self.session.get(LabelPartition, item.partition_id)
            case _ as unsupported_resource_type:
                raise UnsupportedPartitionCompletionResourceTypeError(
                    str(unsupported_resource_type)
                )

    def _create_receipt(
        self,
        item: PartitionCompletionRequest,
    ) -> DatasetPartition | FeaturePartition | LabelPartition:
        """Build the receipt model for one new Iceberg partition completion."""
        receipt_values = {
            "partition_id": item.partition_id,
            "content_hash": item.content_hash,
            "snapshot_id": item.snapshot_id,
            "row_count": item.row_count,
            "iceberg_schema": item.iceberg_schema,
        }
        match item.resource_type:
            case ResourceType.DATASET:
                return DatasetPartition(**receipt_values)
            case ResourceType.FEATURE:
                return FeaturePartition(**receipt_values)
            case ResourceType.LABEL:
                return LabelPartition(**receipt_values)
            case _ as unsupported_resource_type:
                raise UnsupportedPartitionCompletionResourceTypeError(
                    str(unsupported_resource_type)
                )

    def _record_lineage(self, item: PartitionCompletionRequest) -> None:
        """Copy a derived request's immutable dependency plan into its receipt."""
        match item.resource_type:
            case ResourceType.DATASET:
                return
            case ResourceType.FEATURE:
                self.session.add_all(
                    FeaturePartitionDatasetDependency(
                        feature_partition_id=item.partition_id,
                        dataset_partition_id=dataset_partition_id,
                    )
                    for dataset_partition_id in self._planned_ids(
                        item.resource_type, item.partition_id
                    )
                )
            case ResourceType.LABEL:
                self.session.add_all(
                    LabelPartitionDatasetDependency(
                        label_partition_id=item.partition_id,
                        dataset_partition_id=dataset_partition_id,
                    )
                    for dataset_partition_id in self._planned_ids(
                        item.resource_type, item.partition_id
                    )
                )
            case _ as unsupported_resource_type:
                raise UnsupportedPartitionCompletionResourceTypeError(
                    str(unsupported_resource_type)
                )

    def _update_job(
        self,
        item: PartitionCompletionRequest,
        request: DatasetPartitionRequest
        | FeaturePartitionRequest
        | LabelPartitionRequest,
        job: Job,
    ) -> None:
        """Mark a job successful once all of its requests have receipts."""
        match item.resource_type:
            case ResourceType.DATASET:
                statement = select(DatasetPartitionRequest.partition_id).where(
                    DatasetPartitionRequest.job_id == request.job_id,
                    DatasetPartitionRequest.status
                    != PartitionStatus.CREATED.value,
                )
            case ResourceType.FEATURE:
                statement = select(FeaturePartitionRequest.partition_id).where(
                    FeaturePartitionRequest.job_id == request.job_id,
                    FeaturePartitionRequest.status
                    != PartitionStatus.CREATED.value,
                )
            case ResourceType.LABEL:
                statement = select(LabelPartitionRequest.partition_id).where(
                    LabelPartitionRequest.job_id == request.job_id,
                    LabelPartitionRequest.status
                    != PartitionStatus.CREATED.value,
                )
            case _ as unsupported_resource_type:
                raise UnsupportedPartitionCompletionResourceTypeError(
                    str(unsupported_resource_type)
                )
        if (
            job.status == JobStatus.SUBMITTED.value
            and self.session.scalar(statement.limit(1)) is None
        ):
            job.status = JobStatus.SUCCEEDED.value

    @staticmethod
    def _result(
        item: PartitionCompletionRequest,
        receipt: DatasetPartition | FeaturePartition | LabelPartition,
    ) -> PartitionResultResponse:
        """Adapt a persisted receipt to the public completion response DTO."""
        return PartitionResultResponse.model_validate(
            {
                "partition_id": receipt.partition_id,
                "resource_type": item.resource_type,
                "content_hash": receipt.content_hash,
                "snapshot_id": receipt.snapshot_id,
                "row_count": receipt.row_count,
                "iceberg_schema": receipt.iceberg_schema,
                "created_at": receipt.created_at,
            }
        )

    @staticmethod
    def _validate_request_lifecycle(
        request: PartitionRequest,
        resource: Resource,
        job: Job,
    ) -> None:
        """Ensure a callback is authorized by the request and job lifecycles."""
        if resource.deleted_at is not None:
            raise DeletedCompletionResourceError
        match request.status:
            case PartitionStatus.PENDING.value:
                if job.status != JobStatus.SUBMITTED.value:
                    raise InactivePartitionJobError
            case PartitionStatus.CREATED.value:
                pass
            case _ as unsupported_partition_status:
                raise UnsupportedPartitionRequestStatusError(
                    unsupported_partition_status
                )

    def _complete_item(
        self,
        item: PartitionCompletionRequest,
        schemas_by_resource: dict[
            tuple[ResourceType, str, int], tuple[tuple[str, str], ...]
        ],
    ) -> tuple[
        DatasetPartition | FeaturePartition | LabelPartition,
        DatasetPartitionRequest
        | FeaturePartitionRequest
        | LabelPartitionRequest,
        Job,
    ]:
        """Validate and persist one receipt inside the caller's transaction."""
        request, resource, job = self._request(item)
        if not secrets.compare_digest(
            request.callback_token, item.callback_token
        ):
            raise InvalidPartitionCallbackTokenError
        self._validate_request_lifecycle(request, resource, job)
        if item.resource_type is not ResourceType.DATASET and self._planned_ids(
            item.resource_type, item.partition_id
        ) != set(item.dataset_partition_ids or []):
            raise PartitionDependencyPlanMismatchError
        resource_key = (item.resource_type, resource.name, resource.version)
        schema = tuple(sorted(item.iceberg_schema.items()))
        prior_schema = schemas_by_resource.get(resource_key)
        if prior_schema is not None and prior_schema != schema:
            raise CompletionBatchSchemaConflictError
        schemas_by_resource[resource_key] = schema
        if any(
            existing_schema != schema
            for existing_schema in self._existing_schemas(
                item, resource.name, resource.version
            )
        ):
            raise ExistingReceiptSchemaConflictError
        receipt = self._receipt(item)
        if receipt is not None:
            if (
                receipt.content_hash != item.content_hash
                or receipt.snapshot_id != item.snapshot_id
                or receipt.row_count != item.row_count
                or receipt.iceberg_schema != item.iceberg_schema
            ):
                raise PartitionReceiptConflictError
        elif request.status != PartitionStatus.PENDING.value:
            raise MissingCompletedPartitionReceiptError
        else:
            receipt = self._create_receipt(item)
            self.session.add(receipt)
            request.status = PartitionStatus.CREATED.value
            self._record_lineage(item)
        return receipt, request, job

    def complete(
        self, payload: list[PartitionCompletionRequest]
    ) -> list[PartitionResultResponse]:
        """Authenticate and persist one complete Iceberg receipt batch atomically."""
        partition_ids = [item.partition_id for item in payload]
        if len(set(partition_ids)) != len(partition_ids):
            raise DuplicateCompletionPartitionIdError
        receipts: dict[
            PartitionId, DatasetPartition | FeaturePartition | LabelPartition
        ] = {}
        jobs: list[
            tuple[
                PartitionCompletionRequest,
                DatasetPartitionRequest
                | FeaturePartitionRequest
                | LabelPartitionRequest,
                Job,
            ]
        ] = []
        schemas_by_resource: dict[
            tuple[ResourceType, str, int], tuple[tuple[str, str], ...]
        ] = {}
        with self.session.begin():
            for item in payload:
                receipt, request, job = self._complete_item(
                    item, schemas_by_resource
                )
                receipts[item.partition_id] = receipt
                jobs.append((item, request, job))
            self.session.flush()
            for item, request, job in jobs:
                self._update_job(item, request, job)
        for receipt in receipts.values():
            self.session.refresh(receipt)
        return [
            self._result(item, receipts[item.partition_id]) for item in payload
        ]
