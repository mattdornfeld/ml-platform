"""Canonical shared ML Platform domain models."""

from ml_platform.common.domain._base import DomainConfig
from ml_platform.common.domain._exceptions import (
    DomainConfigurationError,
    MarketSymbolConfigurationError,
    TransformDependencyConfigurationError,
)
from ml_platform.common.domain.datasets import (
    DatasetConfig,
    DatasetDefinitionConfig,
    MarketDataConfig,
    MarketDataDatasetConfig,
    MarketDataDatasetDefinitionConfig,
    deserialize_dataset_definition_config,
)
from ml_platform.common.domain.dependencies import (
    DatasetDependency,
    FeatureDatasetDependency,
    FeatureInputWindow,
    LabelDatasetDependency,
    LabelInputWindow,
)
from ml_platform.common.domain.features import FeatureConfig
from ml_platform.common.domain.labels import LabelConfig, TripleBarrierConfig
from ml_platform.common.domain.resources import DataConfig, TransformConfig

__all__ = [
    "DataConfig",
    "DatasetConfig",
    "DatasetDefinitionConfig",
    "DatasetDependency",
    "DomainConfig",
    "DomainConfigurationError",
    "FeatureConfig",
    "FeatureDatasetDependency",
    "FeatureInputWindow",
    "LabelConfig",
    "LabelDatasetDependency",
    "LabelInputWindow",
    "MarketDataConfig",
    "MarketDataDatasetConfig",
    "MarketDataDatasetDefinitionConfig",
    "MarketSymbolConfigurationError",
    "TransformConfig",
    "TransformDependencyConfigurationError",
    "TripleBarrierConfig",
    "deserialize_dataset_definition_config",
]
