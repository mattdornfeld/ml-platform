"""Tests for strict checked-in resource configuration models."""

import unittest

from ml_platform.data_sdk.configs import DataConfig, FeatureKey, PartitionType, RunFrequency


class DataConfigTest(unittest.TestCase):
    """Verify common resource configuration behavior."""

    def test_partition_type_defaults_to_run_frequency(self) -> None:
        """An omitted partition type preserves the configured cadence as a typed enum."""
        config = DataConfig(
            name="market_data",
            version=1,
            run_frequency=RunFrequency.HOURLY,
            keys=[FeatureKey.SYMBOL],
        )

        self.assertIs(config.partition_type, PartitionType.HOURLY)

    def test_explicit_partition_type_is_preserved(self) -> None:
        """A resource can explicitly choose a partition cadence distinct from its run cadence."""
        config = DataConfig(
            name="market_data",
            version=1,
            run_frequency=RunFrequency.HOURLY,
            partition_type=PartitionType.DAILY,
            keys=[FeatureKey.SYMBOL],
        )

        self.assertIs(config.partition_type, PartitionType.DAILY)

    def test_empty_partitions_are_an_explicit_validation_choice(self) -> None:
        """Resource definitions can opt into empty partition receipts."""
        config = DataConfig(
            name="market_data",
            version=1,
            run_frequency=RunFrequency.DAILY,
            keys=[FeatureKey.SYMBOL],
            validations={"allow_empty_partitions": True},
        )

        self.assertTrue(config.validations.allow_empty_partitions)
