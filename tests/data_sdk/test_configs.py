"""Tests for strict checked-in resource configuration models."""

import unittest

from ml_platform.data_sdk.configs import DatasetConfig, FeatureKey, PartitionType, RunFrequency
from ml_platform.data_sdk.materializers import MinimumRowsValidationConfig


class DatasetConfigTest(unittest.TestCase):
    """Verify common resource configuration behavior."""

    def test_partition_type_defaults_to_run_frequency(self) -> None:
        """An omitted partition type preserves the configured cadence as a typed enum."""
        config = DatasetConfig(
            name="market_data",
            version=1,
            run_frequency=RunFrequency.HOURLY,
            keys=[FeatureKey.SYMBOL],
        )

        self.assertIs(config.partition_type, PartitionType.HOURLY)

    def test_explicit_partition_type_is_preserved(self) -> None:
        """A resource can explicitly choose a partition cadence distinct from its run cadence."""
        config = DatasetConfig(
            name="market_data",
            version=1,
            run_frequency=RunFrequency.HOURLY,
            partition_type=PartitionType.DAILY,
            keys=[FeatureKey.SYMBOL],
        )

        self.assertIs(config.partition_type, PartitionType.DAILY)

    def test_partition_validations_default_to_all_requested_partitions(self) -> None:
        """Partition-level validations default to every requested partition."""
        config = DatasetConfig(
            name="market_data",
            version=1,
            run_frequency=RunFrequency.DAILY,
            keys=[FeatureKey.SYMBOL],
            validations=[MinimumRowsValidationConfig(
                minimum_rows=1,
                reason="Each requested partition requires a row.",
            )],
        )

        self.assertEqual(config.validations[0].temporal_filter.type, "all_partitions")
