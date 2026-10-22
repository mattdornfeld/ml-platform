"""Unit and Spark/Iceberg integration coverage for market-data features."""

from pathlib import Path
import unittest
from unittest.mock import MagicMock

from pydantic import ValidationError

from ml_platform.data_sdk.configs import FeatureConfig
from ml_platform.cli.materialization import _feature_configs
from ml_platform.data_sdk.transforms._types import DatasetType


def feature_config(
    name: str, transform: str, dataset_lookback_partitions: int = 1, frequency: str = "daily", parameters: dict[str, int] | None = None
) -> FeatureConfig:
    """Create a test feature configuration."""
    values: dict[str, object] = {
        "name": name,
        "version": 1,
        "dtype": "double",
        "frequency": frequency,
        "datasets": [
            {
                "name": "sp500",
                "version": 3, "dataset_type": DatasetType.MARKET_DATA,
                "argument_name": "market_data",
                "num_lookback_partitions": dataset_lookback_partitions,
            }
        ],
        "transform": transform,
        "description": f"Test {name} feature.",
        "parameters": parameters or {},
    }
    return FeatureConfig.model_validate(values)


class FeaturesUnitTest(unittest.TestCase):
    """Cover feature configuration and source-table validation."""

    def test_loads_each_checked_in_feature_config(self) -> None:
        """Load the checked-in feature specifications from YAML."""
        configs = _feature_configs(Path("configs"))
        self.assertEqual(sorted(name for name, _ in configs), ["log_return", "relative_volume", "trailing_volatility_1d", "trailing_volatility_7d"])
        self.assertEqual(configs[("log_return", 1)]["transform"], "ml_platform.data_sdk.LogReturnTransform")
        self.assertEqual(configs[("log_return", 1)]["datasets"][0]["num_lookback_partitions"], 1)
        self.assertEqual(configs[("trailing_volatility_1d", 1)]["datasets"][0]["num_lookback_partitions"], 1)
        self.assertEqual(configs[("trailing_volatility_7d", 1)]["datasets"][0]["num_lookback_partitions"], 7)
        self.assertEqual(configs[("relative_volume", 1)]["frequency"], "hourly")

    def test_feature_validations_only_expose_configurable_checks(self) -> None:
        """Keep schema, uniqueness, and output-partition checks unconditional."""
        values = feature_config("returns", "ml_platform.data_sdk.LogReturnTransform").model_dump()
        values["validations"] = {"unsupported_validation": False}
        with self.assertRaises(ValidationError):
            FeatureConfig.model_validate(values)

    def test_frequency_uses_daily_or_hourly_metadata(self) -> None:
        """Reject the retired partition_type field and Spark-style values."""
        values = feature_config("returns", "ml_platform.data_sdk.LogReturnTransform").model_dump()
        values["frequency"] = "day"
        with self.assertRaises(ValidationError):
            FeatureConfig.model_validate(values)
        values = feature_config("returns", "ml_platform.data_sdk.LogReturnTransform").model_dump()
        values["partition_type"] = "day"
        with self.assertRaises(ValidationError):
            FeatureConfig.model_validate(values)

    def test_requires_valid_feature_parameters(self) -> None:
        """Reject unsupported fields, names, and invalid lookback combinations."""
        with self.assertRaises(ValidationError):
            feature_config("bad-name", "ml_platform.data_sdk.LogReturnTransform")
        with self.assertRaises(ValidationError):
            FeatureConfig(name="returns", version="one", dtype="double", frequency="daily", datasets=[], transform="ml_platform.data_sdk.LogReturnTransform", description="x")
        with self.assertRaises(ValidationError):
            FeatureConfig(name="returns", version=1, dtype="double", frequency="daily", datasets=[], transform="bad-path", description="x")
        duplicate_datasets = feature_config("returns", "ml_platform.data_sdk.LogReturnTransform").model_dump()
        duplicate_datasets["datasets"].append(duplicate_datasets["datasets"][0])
        with self.assertRaises(ValidationError):
            FeatureConfig.model_validate(duplicate_datasets)
        retired_metadata = feature_config("returns", "ml_platform.data_sdk.LogReturnTransform").model_dump()
        retired_metadata["metadata"] = {"owner": "research"}
        with self.assertRaises(ValidationError):
            FeatureConfig.model_validate(retired_metadata)




if __name__ == "__main__":
    unittest.main()
