"""Public exception hierarchy for the ML Platform data SDK."""

from ml_platform.data_sdk._exceptions import (
    DataSdkError,
    MarketSymbolConfigurationError,
)
from ml_platform.data_sdk.configs._configs import (
    InvalidTimeRangeError,
    TransformDependencyConfigurationError,
)
from ml_platform.data_sdk.dataset_transforms._market_data import (
    MarketDataPartitionError,
    YahooFinanceDataUnavailableError,
    YahooFinancePayloadError,
)
from ml_platform.data_sdk.materializers.exceptions import (
    DataFrameValidationError,
    DatasetPartitionPlanError,
    DuplicateDatasetNameError,
    DuplicateValidationOverrideError,
    ExactColumnsValidationError,
    FiniteValuesValidationError,
    InputKeysValidationError,
    MaterializerError,
    MinimumRowsValidationError,
    MissingIcebergSnapshotError,
    NonNullValuesValidationError,
    RequestedPartitionsValidationError,
    TransformDatasetInjectionError,
    TransformImportError,
    TransformLayoutValidationError,
    TransformParameterError,
    TransformTypeError,
    UniqueKeyValidationError,
    UnsupportedDatasetTypeError,
    UnsupportedValidationOverrideError,
)
from ml_platform.data_sdk.transforms._base import DataTransformSchemaError
from ml_platform.data_sdk.transforms._types import PartitionTypeResolutionError
from ml_platform.data_sdk.utils._record import (
    RecordIdentityError,
    RecordInitializationError,
)
from ml_platform.data_sdk.utils.data_explorer import ParquetExplorerSourceError

__all__ = [
    "DataFrameValidationError",
    "DataSdkError",
    "DataTransformSchemaError",
    "DatasetPartitionPlanError",
    "DuplicateDatasetNameError",
    "DuplicateValidationOverrideError",
    "ExactColumnsValidationError",
    "FiniteValuesValidationError",
    "InputKeysValidationError",
    "InvalidTimeRangeError",
    "MarketDataPartitionError",
    "MarketSymbolConfigurationError",
    "MaterializerError",
    "MinimumRowsValidationError",
    "MissingIcebergSnapshotError",
    "NonNullValuesValidationError",
    "ParquetExplorerSourceError",
    "PartitionTypeResolutionError",
    "RecordIdentityError",
    "RecordInitializationError",
    "RequestedPartitionsValidationError",
    "TransformDatasetInjectionError",
    "TransformDependencyConfigurationError",
    "TransformImportError",
    "TransformLayoutValidationError",
    "TransformParameterError",
    "TransformTypeError",
    "UniqueKeyValidationError",
    "UnsupportedDatasetTypeError",
    "UnsupportedValidationOverrideError",
    "YahooFinanceDataUnavailableError",
    "YahooFinancePayloadError",
]
