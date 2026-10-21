"""Unit and Spark/Iceberg integration coverage for market-data features."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock

from pydantic import ValidationError

from ml_platform.data_sdk.feature_transforms._trailing_volatility import TrailingVolatilityTransform
from ml_platform.data_sdk.configs import DatasetConfig, FeatureConfig
from ml_platform.cli.materialization import _feature_configs
from ml_platform.data_sdk.transforms._types import Dataset
from ml_platform.data_sdk.materializers._feature import (
    FeatureMaterializer,
)
from ml_platform.data_sdk.materializers._dataset import DatasetMaterializer


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
                "version": 3,
                "num_lookback_partitions": dataset_lookback_partitions,
            }
        ],
        "transform": transform,
        "description": f"Test {name} feature.",
        "parameters": parameters or {},
    }
    return FeatureConfig.model_validate(values)


class FeaturesIntegrationTest(unittest.TestCase):
    """Verify feature transformations materialize independently in Iceberg."""

    @classmethod
    def setUpClass(cls) -> None:
        """Start Spark with an isolated Iceberg warehouse."""
        cls.temporary_directory = TemporaryDirectory()
        cls.warehouse = Path(cls.temporary_directory.name) / "warehouse"
        cls.spark = DatasetMaterializer._create_spark_session(cls.warehouse)
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        """Stop Spark and discard the temporary warehouse."""
        cls.spark.stop()
        cls.temporary_directory.cleanup()

    def setUp(self) -> None:
        """Create a source table with deterministic close and volume history."""
        identifiers = [DatasetMaterializer._iceberg_table_identifier('sp500', 3), FeatureMaterializer._dependency_manifest_table_identifier]
        identifiers.extend(
            FeatureMaterializer._feature_table_identifier(config)
            for config in (
                feature_config("test_log_return", "ml_platform.data_sdk.LogReturnTransform"),
                feature_config("test_trailing_volatility", "ml_platform.data_sdk.TrailingVolatilityTransform", 2, parameters={"lookback_bars": 2}),
                feature_config("test_relative_volume", "ml_platform.data_sdk.RelativeVolumeTransform", 2, "hourly", {"lookback_bars": 2}),
            )
        )
        for identifier in identifiers:
            if self.spark.catalog.tableExists(identifier):
                self.spark.sql(f"DROP TABLE {identifier} PURGE")
        self._write_market_rows(
            [
                ("2026-01-02T14:30:00Z", "AAA", 100.0, 100),
                ("2026-01-02T15:30:00Z", "AAA", 110.0, 100),
                ("2026-01-02T16:30:00Z", "AAA", 121.0, 200),
                ("2026-01-02T17:30:00Z", "AAA", 121.0, 100),
                ("2026-01-02T18:30:00Z", "AAA", 121.0, 100),
            ]
        )

    def _write_market_rows(self, rows: list[tuple[str, str, float, int]]) -> None:
        """Materialize concise synthetic OHLCV rows into the test source table."""
        source = self.spark.createDataFrame(
            rows,
            "timestamp string, symbol string, Close double, Volume long",
        ).selectExpr(
            "timestamp", "symbol", "'regular' AS market_session", "Close AS Open", "Close AS High",
            "Close AS Low", "Close", "Volume",
        )
        DatasetMaterializer._materialize_test_frame(source, name="sp500", version=3, spark=self.spark)

    def test_materializes_three_configured_features(self) -> None:
        """Write each PySpark feature to its own table with expected values."""
        log_returns = feature_config("test_log_return", "ml_platform.data_sdk.LogReturnTransform")
        volatility = feature_config("test_trailing_volatility", "ml_platform.data_sdk.TrailingVolatilityTransform", 2, parameters={"lookback_bars": 2})

        self.assertEqual(FeatureMaterializer._materialize_definition(self.spark, log_returns), 4)
        self.assertEqual(FeatureMaterializer._materialize_definition(self.spark, volatility), 2)
        self.assertEqual(FeatureMaterializer._materialize_definition(self.spark, log_returns), 0)

        log_rows = self.spark.table(FeatureMaterializer._feature_table_identifier(log_returns)).orderBy("timestamp").collect()
        self.assertAlmostEqual(log_rows[0].feature_value, 0.0953101798, places=8)
        self.assertEqual(
            FeatureMaterializer._feature_storage_path(self.warehouse, log_returns),
            self.warehouse.resolve() / "features" / "test_log_return-v1",
        )
        self.assertEqual(
            sorted(path.name for path in (FeatureMaterializer._feature_storage_path(self.warehouse, log_returns) / "data").glob("timestamp_day=*")),
            ["timestamp_day=2026-01-02"],
        )
        for config in (log_returns, volatility):
            self.assertEqual(
                self.spark.table(FeatureMaterializer._feature_table_identifier(config)).columns,
                ["timestamp", "symbol", "feature_value"],
            )
        properties = dict(self.spark.sql(f"SHOW TBLPROPERTIES {FeatureMaterializer._feature_table_identifier(log_returns)}").collect())
        self.assertEqual(properties["feature.version"], "1")
        self.assertEqual(properties["feature.transform"], "ml_platform.data_sdk.LogReturnTransform")
        self.assertEqual(properties["feature.frequency"], "daily")
        self.assertEqual(
            self.spark.table(FeatureMaterializer._dependency_manifest_table_identifier)
            .where("feature_name = 'test_log_return' AND feature_version = 1")
            .count(),
            1,
        )

        corrected_source = self.spark.createDataFrame(
            [("2026-01-02T16:30:00Z", "AAA", 122.0, 200)],
            "timestamp string, symbol string, Close double, Volume long",
        ).selectExpr(
            "timestamp", "symbol", "'regular' AS market_session", "Close AS Open", "Close AS High",
            "Close AS Low", "Close", "Volume",
        )
        DatasetMaterializer._materialize_test_frame(corrected_source, name="sp500", version=3, spark=self.spark)
        self.assertEqual(self.spark.table(DatasetMaterializer._iceberg_table_identifier('sp500', 3)).where("timestamp = TIMESTAMP '2026-01-02 16:30:00'").first().Close, 122.0)
        self.assertEqual(FeatureMaterializer._materialize_definition(self.spark, log_returns), 4)
        repaired_rows = self.spark.table(FeatureMaterializer._feature_table_identifier(log_returns)).orderBy("timestamp").collect()
        self.assertEqual(len(repaired_rows), 4)
        self.assertAlmostEqual(repaired_rows[1].feature_value, 0.1035406789, places=8)

    def test_replaces_only_partitions_within_dataset_lookback(self) -> None:
        """A correction invalidates the current and configured future lookback outputs."""
        self._write_market_rows(
            [
                ("2026-01-03T14:30:00Z", "AAA", 130.0, 100),
                ("2026-01-03T15:30:00Z", "AAA", 131.0, 100),
                ("2026-01-03T16:30:00Z", "AAA", 132.0, 100),
                ("2026-01-03T17:30:00Z", "AAA", 133.0, 100),
                ("2026-01-03T18:30:00Z", "AAA", 134.0, 100),
                ("2026-01-04T14:30:00Z", "AAA", 140.0, 100),
                ("2026-01-04T15:30:00Z", "AAA", 141.0, 100),
                ("2026-01-04T16:30:00Z", "AAA", 142.0, 100),
                ("2026-01-04T17:30:00Z", "AAA", 143.0, 100),
                ("2026-01-04T18:30:00Z", "AAA", 144.0, 100),
            ]
        )
        log_returns = feature_config("test_log_return", "ml_platform.data_sdk.LogReturnTransform", 1)
        self.assertEqual(FeatureMaterializer._materialize_definition(self.spark, log_returns), 14)
        unchanged_day_four = self.spark.table(FeatureMaterializer._dependency_manifest_table_identifier).where(
            "feature_name = 'test_log_return' AND date(output_partition) = DATE '2026-01-04'"
        ).select("input_partition", "content_hash").orderBy("input_partition").collect()

        self._write_market_rows([("2026-01-02T16:30:00Z", "AAA", 122.0, 200)])

        self.assertEqual(FeatureMaterializer._materialize_definition(self.spark, log_returns), 9)
        self.assertEqual(self.spark.table(FeatureMaterializer._feature_table_identifier(log_returns)).count(), 14)
        self.assertEqual(
            self.spark.table(FeatureMaterializer._dependency_manifest_table_identifier).where(
                "feature_name = 'test_log_return' AND date(output_partition) = DATE '2026-01-04'"
            ).select("input_partition", "content_hash").orderBy("input_partition").collect(),
            unchanged_day_four,
        )

    def test_removed_parent_deletes_outputs_that_no_longer_have_source_rows(self) -> None:
        """Removing a parent removes its outputs and rebuilds dependent outputs."""
        self._write_market_rows(
            [
                ("2026-01-03T14:30:00Z", "AAA", 130.0, 100),
                ("2026-01-03T15:30:00Z", "AAA", 131.0, 100),
                ("2026-01-03T16:30:00Z", "AAA", 132.0, 100),
                ("2026-01-03T17:30:00Z", "AAA", 133.0, 100),
                ("2026-01-03T18:30:00Z", "AAA", 134.0, 100),
            ]
        )
        log_returns = feature_config("test_log_return", "ml_platform.data_sdk.LogReturnTransform", 1)
        self.assertEqual(FeatureMaterializer._materialize_definition(self.spark, log_returns), 9)

        self.spark.sql(
            f"DELETE FROM {DatasetMaterializer._iceberg_table_identifier('sp500', 3)} "
            "WHERE date(timestamp) = DATE '2026-01-02'"
        )

        self.assertEqual(FeatureMaterializer._materialize_definition(self.spark, log_returns), 4)
        self.assertEqual(self.spark.table(FeatureMaterializer._feature_table_identifier(log_returns)).count(), 4)
        self.assertEqual(
            self.spark.table(FeatureMaterializer._feature_table_identifier(log_returns))
            .where("date(timestamp) = DATE '2026-01-02'")
            .count(),
            0,
        )

    def test_outputs_record_daily_parent_partitions(self) -> None:
        """A daily feature records its current and preceding daily parents."""
        self._write_market_rows(
            [
                ("2026-01-03T14:30:00Z", "AAA", 130.0, 100),
                ("2026-01-03T15:30:00Z", "AAA", 131.0, 100),
                ("2026-01-03T16:30:00Z", "AAA", 132.0, 100),
            ]
        )
        log_returns = feature_config("test_log_return", "ml_platform.data_sdk.LogReturnTransform", 1)
        FeatureMaterializer._materialize_definition(self.spark, log_returns)
        self.assertEqual(
            self.spark.table(FeatureMaterializer._dependency_manifest_table_identifier)
            .where(
                "feature_name = 'test_log_return' "
                "AND output_partition = TIMESTAMP '2026-01-03 00:00:00'"
            )
            .select("input_partition").distinct().count(),
            2,
        )

    def test_enforces_mandatory_transform_output_validations(self) -> None:
        """Reject malformed schemas, duplicate keys, and unrequested partitions."""
        config = feature_config("validated", "ml_platform.data_sdk.LogReturnTransform")
        source = self.spark.createDataFrame(
            [("2026-01-02T14:30:00Z", "AAA", 100.0)],
            "timestamp string, symbol string, Close double",
        ).selectExpr("cast(timestamp as timestamp) as timestamp", "symbol", "Close")
        datasets = {
            "market_data": Dataset(
                metadata=config.datasets[0],
                frame=source,
            )
        }
        requested = self.spark.createDataFrame(
            [("2026-01-02T00:00:00Z",)], "output_partition string"
        ).selectExpr("cast(output_partition as timestamp) as output_partition")

        with self.assertRaisesRegex(ValueError, "exactly"):
            FeatureMaterializer._validate_feature_frame(source.select("timestamp", "symbol"), config, requested, datasets)

        duplicate = source.selectExpr(
            "timestamp", "symbol", "Close as feature_value"
        ).unionByName(source.selectExpr("timestamp", "symbol", "Close as feature_value"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            FeatureMaterializer._validate_feature_frame(duplicate, config, requested, datasets)

        outside_partition = self.spark.createDataFrame(
            [("2026-01-03T14:30:00Z", "AAA", 1.0)],
            "timestamp string, symbol string, feature_value double",
        ).selectExpr("cast(timestamp as timestamp) as timestamp", "symbol", "feature_value")
        with self.assertRaisesRegex(ValueError, "requested partitions"):
            FeatureMaterializer._validate_feature_frame(outside_partition, config, requested, datasets)


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
        dataset = Dataset(
            metadata=DatasetConfig(
                name="sp500", version=3, num_lookback_partitions=1
            ),
            frame=MagicMock(),
        )
        with self.assertRaisesRegex(ValueError, "parameter discrepancy"):
            FeatureMaterializer._instantiate_transform(
                TrailingVolatilityTransform,
                {"lookback_bars": "24"},
                {"market_data": dataset},
            )
        with self.assertRaisesRegex(ValueError, "dataset discrepancy"):
            FeatureMaterializer._instantiate_transform(TrailingVolatilityTransform, {}, {})

    def test_rejects_missing_market_table(self) -> None:
        """Fail before applying a feature when no market source table exists."""
        spark = MagicMock()
        spark.catalog.tableExists.return_value = False
        with self.assertRaisesRegex(ValueError, "Feature dataset table does not exist"):
            FeatureMaterializer._materialize_definition(spark, feature_config("returns", "ml_platform.data_sdk.LogReturnTransform")
            )


if __name__ == "__main__":
    unittest.main()
