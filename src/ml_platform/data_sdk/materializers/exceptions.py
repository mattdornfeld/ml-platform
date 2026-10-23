"""Public exceptions raised while materializing Iceberg-backed resources."""

from ml_platform.data_sdk._exceptions import DataSdkError


class MaterializerError(DataSdkError):
    """Base exception for an API-dispatched resource materialization."""


class DuplicateValidationOverrideError(MaterializerError):
    """Raised when runtime validation overrides repeat a discriminator."""


class MissingIcebergSnapshotError(MaterializerError):
    """Raised when a successful table write has no committed Iceberg snapshot."""


class DatasetPartitionPlanError(MaterializerError):
    """Raised when an output partition lacks its immutable upstream data plan."""


class DuplicateDatasetNameError(MaterializerError):
    """Raised when declared datasets resolve to the same transform argument."""


class TransformImportError(MaterializerError):
    """Raised when the configured transform cannot be imported."""


class TransformTypeError(MaterializerError):
    """Raised when an imported transform does not implement its required contract."""


class TransformDatasetInjectionError(MaterializerError):
    """Raised when transform constructor datasets differ from the declared inputs."""


class TransformParameterError(MaterializerError):
    """Raised when transform parameters do not match its constructor contract."""


class UnsupportedDatasetTypeError(MaterializerError):
    """Raised when no dataset transform supports the requested dataset type."""


class DataFrameValidationError(MaterializerError):
    """Base exception for a materialized DataFrame that violates a declared policy."""


class ExactColumnsValidationError(DataFrameValidationError):
    """Raised when output columns differ from the declared table schema."""


class NonNullValuesValidationError(DataFrameValidationError):
    """Raised when a required output value is null or cannot be cast."""


class UniqueKeyValidationError(DataFrameValidationError):
    """Raised when logical output keys do not uniquely identify output rows."""


class RequestedPartitionsValidationError(DataFrameValidationError):
    """Raised when output contains a partition that was not requested."""


class MinimumRowsValidationError(DataFrameValidationError):
    """Raised when a selected output partition has too few rows."""


class FiniteValuesValidationError(DataFrameValidationError):
    """Raised when a numeric output value is NaN or infinite."""


class InputKeysValidationError(DataFrameValidationError):
    """Raised when output entity keys cannot be traced to an input dataset."""


class TransformLayoutValidationError(DataFrameValidationError):
    """Raised when transform-declared Iceberg layout differs from materializer configuration."""


class UnsupportedValidationOverrideError(DataFrameValidationError):
    """Raised when a caller overrides a validation absent from the default policy."""
