"""Tests for strict checked-in resource configuration models."""

import unittest

from pydantic import ValidationError

from ml_platform.data_sdk import (
    DatasetType,
    FeatureDatasetDependency,
    FeatureInputWindow,
    LabelDatasetDependency,
    LabelInputWindow,
)
from ml_platform.data_sdk.configs import (
    DatasetConfig,
    DataType,
    EntityKey,
    FeatureConfig,
    LabelConfig,
    PartitionType,
    RunFrequency,
)
from ml_platform.data_sdk.materializers import MinimumRowsValidationConfig


class DatasetConfigTest(unittest.TestCase):
    """Verify common resource configuration behavior."""

    def test_partition_type_defaults_to_run_frequency(self) -> None:
        """An omitted partition type preserves the configured cadence as a typed enum."""
        config = DatasetConfig(
            name="market_data",
            version=1,
            run_frequency=RunFrequency.HOURLY,
            keys=[EntityKey.SYMBOL],
        )

        self.assertIs(config.partition_type, PartitionType.HOURLY)

    def test_explicit_partition_type_is_preserved(self) -> None:
        """A resource can explicitly choose a partition cadence distinct from its run cadence."""
        config = DatasetConfig(
            name="market_data",
            version=1,
            run_frequency=RunFrequency.HOURLY,
            partition_type=PartitionType.DAILY,
            keys=[EntityKey.SYMBOL],
        )

        self.assertIs(config.partition_type, PartitionType.DAILY)

    def test_partition_validations_default_to_all_requested_partitions(
        self,
    ) -> None:
        """Partition-level validations default to every requested partition."""
        config = DatasetConfig(
            name="market_data",
            version=1,
            run_frequency=RunFrequency.DAILY,
            keys=[EntityKey.SYMBOL],
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
