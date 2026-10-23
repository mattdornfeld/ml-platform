"""Tests for strict checked-in resource configuration models."""

import unittest

from pydantic import ValidationError

from ml_platform.common.domain import (
    DatasetConfig,
    FeatureConfig,
    FeatureDatasetDependency,
    FeatureInputWindow,
    LabelConfig,
    LabelDatasetDependency,
    LabelInputWindow,
    MarketDataConfig,
    MarketDataDatasetConfig,
)
from ml_platform.common.types import (
    DatasetType,
    DataType,
    EntityKey,
    PartitionType,
    RunFrequency,
)
from ml_platform.data_sdk.materializers import MinimumRowsValidationConfig


class DatasetConfigTest(unittest.TestCase):
    """Verify common resource configuration behavior."""

    def test_partition_type_is_explicit_dataset_metadata(self) -> None:
        """A dataset definition declares its physical partition cadence."""
        config = MarketDataConfig(
            name="market_data",
            version=1,
            partition_type=PartitionType.HOURLY,
            keys=[EntityKey.SYMBOL],
            symbols=["AAPL"],
        )

        self.assertIs(config.partition_type, PartitionType.HOURLY)

    def test_explicit_partition_type_is_preserved(self) -> None:
        """A resource can explicitly choose a partition cadence distinct from its run cadence."""
        config = MarketDataConfig(
            name="market_data",
            version=1,
            run_frequency=RunFrequency.HOURLY,
            partition_type=PartitionType.DAILY,
            keys=[EntityKey.SYMBOL],
            symbols=["AAPL"],
        )

        self.assertIs(config.partition_type, PartitionType.DAILY)

    def test_persisted_dataset_config_uses_dataset_type_discriminator(
        self,
    ) -> None:
        """The type stored in configuration selects the concrete model."""
        config = DatasetConfig.deserialize(
            {
                "dataset_type": "market_data",
                "keys": ["symbol"],
                "partition_type": "daily",
                "symbols": ["AAPL"],
            }
        )

        self.assertIsInstance(config, MarketDataDatasetConfig)
        self.assertIs(config.dataset_type, DatasetType.MARKET_DATA)
        self.assertIs(config.partition_type, PartitionType.DAILY)

    def test_partition_validations_default_to_all_requested_partitions(
        self,
    ) -> None:
        """Partition-level validations default to every requested partition."""
        config = MarketDataConfig(
            name="market_data",
            version=1,
            run_frequency=RunFrequency.DAILY,
            partition_type=PartitionType.DAILY,
            keys=[EntityKey.SYMBOL],
            symbols=["AAPL"],
            validations=[
                MinimumRowsValidationConfig(
                    minimum_rows=1,
                    reason="Each requested partition requires a row.",
                )
            ],
        )

        self.assertEqual(
            config.validations[0].temporal_filter.type, "all_partitions"
        )


class TransformConfigTest(unittest.TestCase):
    """Verify feature and label dependency contracts remain distinct."""

    def test_feature_config_accepts_only_historical_dependency_windows(
        self,
    ) -> None:
        """A feature dependency provides only preceding source partitions."""
        config = FeatureConfig(
            name="trailing_volatility",
            version=1,
            description="Trailing volatility.",
            dtype=DataType.DOUBLE,
            frequency=RunFrequency.DAILY,
            datasets=[
                FeatureDatasetDependency(
                    name="sp500",
                    version=3,
                    dataset_type=DatasetType.MARKET_DATA,
                    input_window=FeatureInputWindow(lookback_partitions=1),
                )
            ],
            transform="ml_platform.data_sdk.TrailingVolatilityTransform",
            parameters={"lookback_bars": 24},
        )

        self.assertIsInstance(config.datasets[0], FeatureDatasetDependency)

    def test_label_config_accepts_realized_outcome_dependency_windows(
        self,
    ) -> None:
        """A label dependency provides prior and following source partitions."""
        config = LabelConfig(
            name="triple_barrier",
            version=1,
            description="Triple barrier label.",
            dtype=DataType.INT,
            frequency=RunFrequency.DAILY,
            datasets=[
                LabelDatasetDependency(
                    name="sp500",
                    version=3,
                    dataset_type=DatasetType.MARKET_DATA,
                    input_window=LabelInputWindow(
                        lookback_partitions=1,
                        lookahead_partitions=1,
                    ),
                )
            ],
            transform="ml_platform.data_sdk.TripleBarrierLabelTransform",
            parameters={"horizon_bars": 24},
        )

        self.assertIsInstance(config.datasets[0], LabelDatasetDependency)

    def test_dataset_names_must_be_unique_for_transform_injection(self) -> None:
        """A transform cannot receive two versions through one dataset keyword."""
        config_payload: dict[str, object] = {
            "name": "joined_feature",
            "version": 1,
            "description": "Joined feature.",
            "dtype": DataType.DOUBLE,
            "frequency": RunFrequency.DAILY,
            "datasets": [
                {
                    "name": "market_data",
                    "version": 1,
                    "dataset_type": DatasetType.MARKET_DATA,
                    "input_window": {"lookback_partitions": 1},
                },
                {
                    "name": "market_data",
                    "version": 2,
                    "dataset_type": DatasetType.MARKET_DATA,
                    "input_window": {"lookback_partitions": 1},
                },
            ],
            "transform": "ml_platform.data_sdk.TrailingVolatilityTransform",
            "parameters": {"lookback_bars": 24},
        }

        with self.assertRaisesRegex(
            ValidationError, "datasets must have unique names"
        ):
            FeatureConfig.model_validate(config_payload)

    def test_parameter_values_must_be_scalar(self) -> None:
        """Transform parameters reject nested values outside ParameterValue."""
        config_payload: dict[str, object] = {
            "name": "trailing_volatility",
            "version": 1,
            "description": "Trailing volatility.",
            "dtype": "double",
            "frequency": "daily",
            "datasets": [
                {
                    "name": "sp500",
                    "version": 3,
                    "dataset_type": "market_data",
                    "input_window": {"lookback_partitions": 1},
                }
            ],
            "transform": "ml_platform.data_sdk.TrailingVolatilityTransform",
            "parameters": {"lookback_bars": [24]},
        }

        with self.assertRaises(ValidationError):
            FeatureConfig.model_validate(config_payload)
