"""Tests for API-to-Spark materializer configuration contracts."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ml_platform.data_sdk.configs import (
    DataConfig,
    DataValidations,
    DatasetConfig,
    FeatureConfig,
    FeatureValidations,
    LabelConfig,
    MarketDataConfig,
    TimeRange,
    TripleBarrierConfig,
)
from ml_platform.data_sdk.materializers import (
    DatasetMaterializer,
    DatasetMaterializationConfig,
    DatasetDependency,
    FeatureMaterializer,
    FeatureMaterializationConfig,
    LabelMaterializer,
    LabelMaterializationConfig,
    Materializer,
    MaterializerConfig,
    PartionDependency,
    MaterializerPartition,
    MaterializerValidations,
)
from ml_platform.data_sdk.transforms._base import DataTransform
from ml_platform.data_sdk.transforms._dataset_transform import DatasetTransform
from ml_platform.data_sdk.transforms._feature_transform import FeatureTransform
from ml_platform.data_sdk.transforms._label_transform import LabelTransform


class MaterializerContractTest(unittest.TestCase):
    """Keep Spark job contracts distinct from checked-in YAML config models."""

    def test_data_transform_requires_explicit_storage_layouts(self) -> None:
        """Every transform must statically declare its own Iceberg storage layout."""
        for transform in (DataTransform, DatasetTransform, FeatureTransform, LabelTransform):
            with self.subTest(transform=transform.__name__):
                self.assertTrue(issubclass(transform, DataTransform))
        self.assertIn("value_schema", DataTransform.__abstractmethods__)
        self.assertIn("value_schema", DatasetTransform.__abstractmethods__)
        self.assertNotIn("value_schema", FeatureTransform.__abstractmethods__)
        self.assertNotIn("value_schema", LabelTransform.__abstractmethods__)
        self.assertNotIn("iceberg_schema", DataTransform.__abstractmethods__)
        self.assertNotIn("iceberg_partition_spec", DataTransform.__abstractmethods__)

    def test_dataset_materializer_deserializes_its_own_job_config(self) -> None:
        """Dataset worker JSON contains the API-supplied identity and type."""
        config = DatasetMaterializer._config_from_json(json.dumps({
            "name": "sp500", "version": 3, "dataset_type": "market_data", "keys": ["symbol"], "partition_type": "daily",
            "description": "Hourly prices.", "validations": {}, "symbols": ["aapl"],
            "warehouse_dir": "s3a://warehouse",
            "partitions": [{"partition_id": "partition-1", "timestamp": "2026-01-01T00:00:00Z", "callback_token": "token"}],
        }))
        self.assertEqual(config.symbols, ["AAPL"])
        self.assertEqual(config.dataset_type, "market_data")
        with self.assertRaises(ValueError):
            DatasetMaterializer._config_from_json(json.dumps({
                "name": "sp500", "version": 3, "dataset_type": "market_data", "keys": ["symbol"], "partition_type": "daily", "description": "Hourly prices.",
                "validations": {}, "symbols": ["AAPL"], "warehouse_dir": "s3a://warehouse",
                "time_range": {"start_date": "2026-01-01", "end_date": "2026-01-02"},
                "partitions": [{"partition_id": "partition-1", "timestamp": "2026-01-01T00:00:00Z", "callback_token": "token"}],
            }))

    def test_feature_materializer_deserializes_its_own_job_config(self) -> None:
        """Feature worker JSON carries dependencies and validation behavior."""
        config = FeatureMaterializer._config_from_json(json.dumps({
            "name": "log_return", "version": 1, "dtype": "double", "frequency": "daily",
            "feature_keys": ["symbol"], "description": "Returns.", "warehouse_dir": "s3a://warehouse",
            "partitions": [{"partition_id": "feature-partition", "timestamp": "2026-01-01T00:00:00Z", "callback_token": "token", "partition_dependencies": [{"partition_id": "dataset-partition", "dataset_name": "sp500", "dataset_version": 3, "timestamp": "2026-01-01T00:00:00Z", "snapshot_id": 1}]}],
            "datasets": [{"name": "sp500", "version": 3, "argument_name": "market_data", "num_lookback_partitions": 1}],
            "transform": "ml_platform.data_sdk.feature_transforms._log_return.LogReturnTransform",
            "parameters": {"lookback_bars": 1},
            "validations": {"min_rows_per_partition": 2, "require_finite_feature_value": True, "require_feature_keys_from_input": True},
        }))
        self.assertEqual(config.datasets[0].id, "sp500_v3")
        self.assertEqual(config.validations.min_rows_per_partition, 2)

    def test_label_materializer_deserializes_transform_contract(self) -> None:
        """Label worker JSON describes its transform, inputs, and parameters."""
        config = LabelMaterializer._config_from_json(json.dumps({
            "name": "hourly_triple_barrier", "version": 1, "description": "Labels.", "warehouse_dir": "s3a://warehouse",
            "validations": {}, "partitions": [{"partition_id": "label-partition", "timestamp": "2026-01-01T00:00:00Z", "callback_token": "token", "partition_dependencies": [{"partition_id": "parent-partition", "dataset_name": "sp500", "dataset_version": 3, "timestamp": "2026-01-01T00:00:00Z", "snapshot_id": 1}]}], "datasets": [{"name": "sp500", "version": 3, "num_lookback_partitions": 2}],
            "transform": "ml_platform.data_sdk.label_transforms._triple_barrier.TripleBarrierLabelTransform",
            "parameters": {"horizon_bars": 24, "volatility_lookback_bars": 24, "profit_taking_multiplier": 1.0, "stop_loss_multiplier": 1.0},
        }))
        self.assertEqual(config.name, "hourly_triple_barrier")

    def test_label_materializer_requires_pinned_inputs(self) -> None:
        """A label worker cannot acknowledge output without exact upstream revisions."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            LabelMaterializationConfig.model_validate_json(json.dumps({
                "name": "hourly_triple_barrier", "version": 1, "description": "Labels.",
                "warehouse_dir": "s3a://warehouse",
                "partitions": [{"partition_id": "label-partition", "timestamp": "2026-01-01T00:00:00Z", "callback_token": "token"}],
                "datasets": [{"name": "sp500", "version": 3, "num_lookback_partitions": 2}],
                "transform": "ml_platform.data_sdk.label_transforms._triple_barrier.TripleBarrierLabelTransform",
            }))

    def test_materializer_requires_declared_class_properties(self) -> None:
        """Every concrete worker declares the complete abstract class-property contract."""
        required_properties = {
            "config_model",
            "spark_application_name",
            "iceberg_namespace",
            "callback_resource",
            "allow_empty_partitions",
            "write_mode",
        }
        self.assertTrue(required_properties.issubset(Materializer.__abstractmethods__))

        class MissingNamespaceMaterializer(Materializer[MaterializerConfig]):
            config_model = MaterializerConfig

            def _create_transform(self, spark: object) -> object:
                raise NotImplementedError

            def materialize(self, spark: object, transform: object) -> object:
                raise NotImplementedError

        self.assertTrue(required_properties.issubset(MissingNamespaceMaterializer.__abstractmethods__))
        with self.assertRaises(TypeError):
            MissingNamespaceMaterializer.__new__(MissingNamespaceMaterializer)
        for materializer in (DatasetMaterializer, FeatureMaterializer, LabelMaterializer):
            self.assertFalse(materializer.__abstractmethods__)

    def test_dataset_and_feature_workers_share_the_base_cli_entrypoint(self) -> None:
        """The common entry point parses JSON and dispatches through each worker hook."""
        configurations = (
            (
                DatasetMaterializer,
                {
                    "name": "sp500", "version": 3, "dataset_type": "market_data", "keys": ["symbol"], "partition_type": "daily",
                    "description": "Hourly prices.", "validations": {}, "symbols": ["AAPL"],
                    "warehouse_dir": "s3a://warehouse",
                    "partitions": [{"partition_id": "partition-1", "timestamp": "2026-01-01T00:00:00Z", "callback_token": "token"}],
                },
            ),
            (
                FeatureMaterializer,
                {
                    "name": "log_return", "version": 1, "dtype": "double", "frequency": "daily",
                    "feature_keys": ["symbol"], "description": "Returns.", "warehouse_dir": "s3a://warehouse",
                    "partitions": [{"partition_id": "feature-partition", "timestamp": "2026-01-01T00:00:00Z", "callback_token": "token", "partition_dependencies": [{"partition_id": "dataset-partition", "dataset_name": "sp500", "dataset_version": 3, "timestamp": "2026-01-01T00:00:00Z", "snapshot_id": 1}]}],
                    "datasets": [{"name": "sp500", "version": 3, "argument_name": "market_data", "num_lookback_partitions": 1}],
                    "transform": "ml_platform.data_sdk.feature_transforms._log_return.LogReturnTransform",
                    "parameters": {"lookback_bars": 1}, "validations": {},
                },
            ),
        )
        for materializer, document in configurations:
            with (
                self.subTest(materializer=materializer.__name__),
                patch.object(materializer, "_run_config") as _run_config,
                patch("sys.argv", [materializer.__name__, "--config-json", json.dumps(document)]),
                patch("builtins.print") as output,
            ):
                materializer.main()
                self.assertEqual(_run_config.call_args.args[0].name, document["name"])
                output.assert_called_once_with("Materialization completed.")

    def test_materializers_use_versioned_tables_in_the_shared_catalog(self) -> None:
        """Runtime table names derive from the resource identity and materializer namespace."""
        self.assertEqual(DatasetMaterializer._iceberg_table_identifier("sp500", 3), "ml_platform.market_data.`sp500-v3`")
        self.assertEqual(FeatureMaterializer._iceberg_table_identifier("log_return", 1), "ml_platform.features.`log_return-v1`")
        self.assertEqual(LabelMaterializer._iceberg_table_identifier("triple_barrier", 2), "ml_platform.labels.`triple_barrier-v2`")

    def test_shared_fields_are_declared_by_materializer_config(self) -> None:
        """All materializer contracts inherit the common resource and partition fields."""
        for field in ("name", "version", "description", "validations", "partitions"):
            self.assertIn(field, MaterializerConfig.model_fields)
            self.assertIn(field, DatasetMaterializer.config_model.model_fields)
            self.assertIn(field, FeatureMaterializer.config_model.model_fields)
            self.assertIn(field, LabelMaterializer.config_model.model_fields)

    def test_data_sdk_model_fields_have_schema_descriptions(self) -> None:
        """Every Pydantic field exposed by the data SDK is self-documenting."""
        models = (
            DataValidations,
            FeatureValidations,
            TimeRange,
            DataConfig,
            DatasetConfig,
            LabelConfig,
            FeatureConfig,
            MarketDataConfig,
            TripleBarrierConfig,
            DatasetDependency,
            PartionDependency,
            MaterializerPartition,
            MaterializerValidations,
            MaterializerConfig,
            DatasetMaterializationConfig,
            FeatureMaterializationConfig,
            LabelMaterializationConfig,
        )
        for model in models:
            missing = [name for name, field in model.model_fields.items() if not field.description]
            self.assertEqual(missing, [], f"{model.__name__} has undocumented fields: {missing}")

    def test_materializer_config_rejects_unknown_fields_and_coercion(self) -> None:
        """The shared runtime base applies strict immutable settings to every job contract."""
        with self.assertRaises(ValueError):
            MaterializerConfig.model_validate({"warehouse_dir": 123})
        with self.assertRaises(ValueError):
            DatasetMaterializer._config_from_json(json.dumps({
                "name": "sp500", "version": 3, "dataset_type": "market_data", "keys": ["symbol"], "partition_type": "daily", "description": "Hourly prices.",
                "validations": {}, "symbols": ["AAPL"], "warehouse_dir": "s3a://warehouse",
                "partitions": [{"partition_id": "partition-1", "timestamp": "2026-01-01T00:00:00Z", "callback_token": "token"}],
                "unexpected": True,
            }))

    def test_runtime_transforms_do_not_import_checked_in_config_models(self) -> None:
        """Runtime transform contracts remain independent from the YAML config layer."""
        root = Path("src/ml_platform/data_sdk/transforms")
        for source in root.glob("*.py"):
            self.assertNotIn("data_sdk.configs", source.read_text(), source)

    def test_materialization_mutations_use_the_dataframe_dsl(self) -> None:
        """Keep row and table mutations readable as PySpark method chains."""
        materializers = Path("src/ml_platform/data_sdk/materializers")
        for filename in ("_dataset.py", "_feature.py", "_label.py"):
            self.assertNotIn(".sql(", (materializers / filename).read_text())

        base = (materializers / "_base.py").read_text()
        self.assertEqual(base.count("spark.sql("), 4)
        self.assertIn("CREATE NAMESPACE IF NOT EXISTS", base)
        self.assertIn("SHOW TBLPROPERTIES", base)
        self.assertIn("SET TBLPROPERTIES", base)
        self.assertIn("UNSET TBLPROPERTIES", base)


if __name__ == "__main__":
    unittest.main()
