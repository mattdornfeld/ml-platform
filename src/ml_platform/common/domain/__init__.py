"""Canonical shared ML Platform domain models."""

from ml_platform.common.domain._base import DomainConfig
from ml_platform.common.domain._exceptions import (
    DatasetTypeConfigurationError,
    DomainConfigurationError,
    MarketSymbolConfigurationError,
    TransformDependencyConfigurationError,
)
from ml_platform.common.domain.datasets import (
    DatasetConfig,
    DatasetConfigModel,
    MarketDataConfig,
    MarketDataDatasetConfig,
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
    "DatasetConfigModel",
    "DatasetDependency",
    "DatasetTypeConfigurationError",
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
    "MarketSymbolConfigurationError",
    "TransformConfig",
    "TransformDependencyConfigurationError",
    "TripleBarrierConfig",
]
