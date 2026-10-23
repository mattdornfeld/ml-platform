"""Typed API errors raised while enforcing persisted resource invariants."""

from datetime import datetime

from fastapi import status

from ml_platform.api._exceptions import _ApiError


class FeatureDependencyDatasetUnavailableError(_ApiError):
    """A feature definition references an absent or deleted dataset version."""

    def __init__(self) -> None:
        """Build the invalid-feature-dependency response."""
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Feature datasets must reference active dataset versions.",
        )


class DatasetVersionExistsError(_ApiError):
    """A dataset name and version is already registered."""

    def __init__(self) -> None:
        """Build the duplicate-dataset response."""
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail="Dataset name and version already exist.",
        )


class FeatureDependencyDatasetTypeMismatchError(_ApiError):
    """A feature definition declares the wrong type for an input dataset."""

    def __init__(self) -> None:
        """Build the invalid-feature-dataset-type response."""
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Feature dataset type must match the registered dataset version.",
        )


class FeatureVersionExistsError(_ApiError):
    """A feature name and version is already registered."""

    def __init__(self) -> None:
        """Build the duplicate-feature response."""
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail="Feature name and version already exist.",
        )


class LabelDependencyDatasetUnavailableError(_ApiError):
    """A label definition references an absent or deleted dataset version."""

    def __init__(self) -> None:
        """Build the invalid-label-dependency response."""
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Label datasets must reference active dataset versions.",
        )


class LabelDependencyDatasetTypeMismatchError(_ApiError):
    """A label definition declares the wrong type for an input dataset."""

    def __init__(self) -> None:
        """Build the invalid-label-dataset-type response."""
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Label dataset type must match the registered dataset version.",
        )


class LabelVersionExistsError(_ApiError):
    """A label name and version is already registered."""

    def __init__(self) -> None:
        """Build the duplicate-label response."""
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail="Label name and version already exist.",
        )


class MissingFeatureInputPartitionsError(_ApiError):
    """A feature target lacks the dataset partitions required for its inputs."""

    def __init__(
        self,
        dataset_name: str,
        dataset_version: int,
        missing_timestamps: tuple[datetime, ...],
    ) -> None:
        """Build a conflict response identifying unavailable dataset inputs."""
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Feature partition cannot be materialized because its required "
                f"{dataset_name}:{dataset_version} input partitions are missing: "
                f"{tuple(item.isoformat() for item in missing_timestamps)}"
            ),
        )


class FeatureVersionUnavailableError(_ApiError):
    """A feature version is absent or deleted when materialization is requested."""

    def __init__(self) -> None:
        """Build the feature-not-found response."""
        super().__init__(
            status_code=404, detail="Feature version does not exist."
        )


class DeletedFeatureInputDatasetError(_ApiError):
    """A feature input dataset was deleted."""

    def __init__(self) -> None:
        """Build the immutable-input conflict response."""
        super().__init__(
            status_code=409,
            detail="Feature partition cannot be materialized because an input dataset version is deleted.",
        )


class FeaturePartitionPlanConflictError(_ApiError):
    """An active feature partition request has a different input plan."""

    def __init__(self) -> None:
        """Build the active-plan conflict response."""
        super().__init__(
            status_code=409,
            detail="Feature partition is already being materialized against a different input revision plan.",
        )


class LabelVersionUnavailableError(_ApiError):
    """A label version is absent or deleted when materialization is requested."""

    def __init__(self) -> None:
        """Build the label-not-found response."""
        super().__init__(
            status_code=404, detail="Label version does not exist."
        )


class DeletedLabelInputDatasetError(_ApiError):
    """A label input dataset was deleted."""

    def __init__(self) -> None:
        """Build the immutable-input conflict response."""
        super().__init__(
            status_code=409,
            detail="Label partition cannot be materialized because an input dataset version is deleted.",
        )


class LabelPartitionPlanConflictError(_ApiError):
    """An active label partition request has a different input plan."""

    def __init__(self) -> None:
        """Build the active-plan conflict response."""
        super().__init__(
            status_code=409,
            detail="Label partition is already being materialized against a different input revision plan.",
        )


class MissingLabelInputPartitionsError(_ApiError):
    """A label target lacks the dataset partitions required for its inputs."""

    def __init__(self, detail: str) -> None:
        """Build the missing-input conflict response."""
        super().__init__(status_code=409, detail=detail)


class _StaticRepositoryError(_ApiError):
    """Repository error with a fixed API response."""

    status_code: int
    detail: str

    def __init__(self) -> None:
        """Initialize the declared fixed API response."""
        super().__init__(status_code=self.status_code, detail=self.detail)


class DeletedCompletionResourceError(_StaticRepositoryError):
    """A completion callback targets a deleted resource."""

    status_code = 409
    detail = "Partition request is not eligible for completion."


class InactivePartitionJobError(_StaticRepositoryError):
    """A completion callback belongs to an inactive job."""

    status_code = 409
    detail = "Partition job is not active."


class InvalidPartitionCallbackTokenError(_StaticRepositoryError):
    """A completion callback token does not authenticate its request."""

    status_code = 403
    detail = "Invalid partition callback token."


class PartitionDependencyPlanMismatchError(_StaticRepositoryError):
    """A callback supplies different upstream revisions than its plan."""

    status_code = 409
    detail = "Callback dataset revisions do not match the immutable input plan."


class CompletionBatchSchemaConflictError(_StaticRepositoryError):
    """One completion batch supplies conflicting schemas for one table."""

    status_code = 409
    detail = "Completion batch has conflicting schemas for one Iceberg table."


class ExistingReceiptSchemaConflictError(_StaticRepositoryError):
    """A completion schema conflicts with committed receipts."""

    status_code = 409
    detail = (
        "Completion schema conflicts with an existing Iceberg table receipt."
    )


class PartitionReceiptConflictError(_StaticRepositoryError):
    """A retry supplies data different from its existing receipt."""

    status_code = 409
    detail = "Partition receipt conflicts with an existing completion."


class MissingCompletedPartitionReceiptError(_StaticRepositoryError):
    """A completed request has no persisted completion receipt."""

    status_code = 409
    detail = "Completed partition is missing its receipt."


class DuplicateCompletionPartitionIdError(_StaticRepositoryError):
    """A completion batch repeats a partition identifier."""

    status_code = 409
    detail = "Completion batch contains duplicate partition IDs."


class UnsupportedPartitionRequestStatusError(_ApiError):
    """A callback targets a partition request status outside its lifecycle."""

    def __init__(self, partition_status: str) -> None:
        """Build the ineligible-partition conflict response."""
        super().__init__(
            status_code=409,
            detail=(
                "Partition request is not eligible for completion: "
                f"{partition_status!r}."
            ),
        )


class UnsupportedScheduledRequestTypeError(_ApiError):
    """The scheduler cannot resolve an unsupported persisted request model."""

    def __init__(self, request_type: str) -> None:
        """Build the scheduler invariant response."""
        super().__init__(
            status_code=500,
            detail=f"Unsupported scheduled partition request type {request_type}.",
        )


class DatasetMaterializationJobWithoutPartitionsError(_ApiError):
    """A dataset job was submitted without target partitions."""

    def __init__(self, job_id: str) -> None:
        """Build the scheduler invariant response."""
        super().__init__(
            status_code=500,
            detail=f"Dataset materialization job {job_id} has no partitions.",
        )


class DatasetMaterializationJobMultipleTargetsError(_ApiError):
    """A dataset job targets more than one dataset version."""

    def __init__(self, job_id: str) -> None:
        """Build the scheduler invariant response."""
        super().__init__(
            status_code=500,
            detail=(
                f"Dataset materialization job {job_id} has multiple target "
                "resources."
            ),
        )


class FeatureMaterializationJobWithoutPartitionsError(_ApiError):
    """A feature job was submitted without target partitions."""

    def __init__(self, job_id: str) -> None:
        """Build the scheduler invariant response."""
        super().__init__(
            status_code=500,
            detail=f"Feature materialization job {job_id} has no partitions.",
        )


class FeatureMaterializationJobMultipleTargetsError(_ApiError):
    """A feature job targets more than one feature version."""

    def __init__(self, job_id: str) -> None:
        """Build the scheduler invariant response."""
        super().__init__(
            status_code=500,
            detail=(
                f"Feature materialization job {job_id} has multiple target "
                "resources."
            ),
        )


class LabelMaterializationJobWithoutPartitionsError(_ApiError):
    """A label job was submitted without target partitions."""

    def __init__(self, job_id: str) -> None:
        """Build the scheduler invariant response."""
        super().__init__(
            status_code=500,
            detail=f"Label materialization job {job_id} has no partitions.",
        )


class LabelMaterializationJobMultipleTargetsError(_ApiError):
    """A label job targets more than one label version."""

    def __init__(self, job_id: str) -> None:
        """Build the scheduler invariant response."""
        super().__init__(
            status_code=500,
            detail=(
                f"Label materialization job {job_id} has multiple target "
                "resources."
            ),
        )


class ScheduledMaterializationJobNotFoundError(_ApiError):
    """The scheduler was asked to reconcile an absent job."""

    def __init__(self, job_id: str) -> None:
        """Build the absent-job response."""
        super().__init__(
            status_code=404,
            detail=f"Materialization job {job_id} does not exist.",
        )


class UnsupportedDatasetStatusError(_ApiError):
    """Dataset listing received an unimplemented lifecycle status."""

    def __init__(self, dataset_status: str) -> None:
        """Build the unsupported-status response."""
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unsupported dataset lifecycle status {dataset_status!r}.",
        )


class UnsupportedJobResourceTypeError(_ApiError):
    """Job lookup received a resource type without a request-table mapping."""

    def __init__(self, resource_type: str) -> None:
        """Build the unsupported-resource-type response."""
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unsupported job resource type {resource_type!r}.",
        )


class UnsupportedPartitionCompletionResourceTypeError(_ApiError):
    """Partition completion received a resource type without receipt handling."""

    def __init__(self, resource_type: str) -> None:
        """Build the unsupported-completion-resource response."""
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Unsupported partition completion resource type "
                f"{resource_type!r}."
            ),
        )
