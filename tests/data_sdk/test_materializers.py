"""Contracts for API-dispatched materializers and their Iceberg tables."""

import json
import unittest
from pathlib import Path

from ml_platform.data_sdk.configs import (
    DataConfig,
    DatasetConfig,
    FeatureConfig,
    LabelConfig,
    MarketDataConfig,
    TripleBarrierConfig,
)
from ml_platform.data_sdk.materializers import (
    DatasetMaterializationConfig,
    DatasetMaterializer,
    FeatureMaterializationConfig,
    FeatureMaterializer,
    IcebergTable,
    LabelMaterializationConfig,
    LabelMaterializer,
    Materializer,
    MaterializerConfig,
    MaterializerPartition,
    MinimumRowsValidationConfig,
    PartitionDependency,
)
from ml_platform.data_sdk.transforms._base import DataTransform
from ml_platform.data_sdk.transforms._dataset_transform import DatasetTransform
from ml_platform.data_sdk.transforms._feature_transform import FeatureTransform
from ml_platform.data_sdk.transforms._label_transform import LabelTransform
from ml_platform.data_sdk.transforms._types import DatasetDependency


class MaterializerContractTest(unittest.TestCase):
    """Keep materializer concerns separate from table persistence concerns."""

    def test_data_transforms_define_static_output_layouts(self) -> None:
        """Every concrete transform family inherits the shared output contract."""
        for transform in (
            DataTransform,
            DatasetTransform,
            FeatureTransform,
            LabelTransform,
        ):
            self.assertTrue(issubclass(transform, DataTransform))
        self.assertIn("value_schema", DataTransform.__abstractmethods__)
        self.assertNotIn("value_schema", FeatureTransform.__abstractmethods__)
        self.assertNotIn("value_schema", LabelTransform.__abstractmethods__)

    def test_materializer_requires_an_iceberg_table(self) -> None:
        """Storage configuration is an explicit abstract materializer dependency."""
        self.assertIn("iceberg_table", Materializer.__abstractmethods__)
        self.assertFalse(DatasetMaterializer.__abstractmethods__)

    def test_dataset_config_creates_market_data_table(self) -> None:
        """Dataset storage settings are injected into a dedicated table object."""
        config = DatasetMaterializer._config_from_json(
            json.dumps(
                {
                    "name": "sp500",
                    "version": 3,
                    "dataset_type": "market_data",
                    "keys": ["symbol"],
                    "partition_type": "daily",
                    "description": "Hourly prices.",
                    "validations": [],
                    "symbols": ["aapl"],
                    "warehouse_dir": "s3a://warehouse",
                    "partitions": [
                        {
                            "partition_id": "partition-1",
                            "timestamp": "2026-01-01T00:00:00Z",
                            "callback_token": "token",
                        }
                    ],
                }
            )
        )
        table = DatasetMaterializer(config).iceberg_table
        self.assertEqual(config.symbols, ["AAPL"])
        self.assertEqual(table.id, "ml_platform.market_data.`sp500-v3`")
        self.assertEqual(table.resource_type, "dataset")

    def test_table_identifier_is_independent_of_materializer_classes(
        self,
    ) -> None:
        """Iceberg table IDs are owned by the table value object."""
        self.assertEqual(
            IcebergTable.id_for("ml_platform", "features", "log_return", 1),
            "ml_platform.features.`log_return-v1`",
        )

    def test_shared_fields_are_declared_by_materializer_config(self) -> None:
        """All runtime contracts inherit shared resource and partition metadata."""
        for field in (
            "name",
            "version",
            "description",
            "validations",
            "partitions",
        ):
            self.assertIn(field, MaterializerConfig.model_fields)
            self.assertIn(field, DatasetMaterializationConfig.model_fields)
            self.assertIn(field, FeatureMaterializationConfig.model_fields)
            self.assertIn(field, LabelMaterializationConfig.model_fields)

    def test_data_sdk_model_fields_have_schema_descriptions(self) -> None:
        """Pydantic models exposed by the data SDK document every field."""
        models = (
            DataConfig,
            DatasetConfig,
            LabelConfig,
            FeatureConfig,
            MarketDataConfig,
            TripleBarrierConfig,
            DatasetDependency,
            PartitionDependency,
            MaterializerPartition,
            MinimumRowsValidationConfig,
            MaterializerConfig,
            DatasetMaterializationConfig,
            FeatureMaterializationConfig,
            LabelMaterializationConfig,
        )
        for model in models:
            missing = [
                name
                for name, field in model.model_fields.items()
                if not field.description
            ]
            self.assertEqual(
                missing,
                [],
                f"{model.__name__} has undocumented fields: {missing}",
            )

    def test_transforms_remain_decoupled_from_configs_and_tables(self) -> None:
        """User transforms never import configuration or persistence implementation modules."""
        for source in Path("src/ml_platform/data_sdk/transforms").glob("*.py"):
            text = source.read_text()
            self.assertNotIn("data_sdk.configs", text, source)
            self.assertNotIn("materializers._iceberg_table", text, source)


if __name__ == "__main__":
    unittest.main()


class TransformMaterializerContractTest(unittest.TestCase):
    """Verify labels and features share their table lifecycle contract."""

    def test_feature_and_label_tables_use_the_shared_transform_definition(
        self,
    ) -> None:
        """Only static table values differ between the two transform resource kinds."""
        common = {
            "name": "signal",
            "version": 1,
            "description": "Synthetic transform output.",
            "warehouse_dir": "s3a://warehouse",
            "dtype": "double",
            "frequency": "daily",
            "keys": ["symbol"],
            "datasets": [
                {
                    "name": "sp500",
                    "version": 3,
                    "dataset_type": "market_data",
                    "input_window": {"lookback_partitions": 1},
                }
            ],
            "transform": "ml_platform.data_sdk.LogReturnTransform",
            "partitions": [
                {
                    "partition_id": "partition-1",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "callback_token": "token",
                    "partition_dependencies": [
                        {
                            "partition_id": "market-partition-1",
                            "dataset_name": "sp500",
                            "dataset_version": 3,
                            "timestamp": "2026-01-01T00:00:00Z",
                            "snapshot_id": 1,
                        }
                    ],
                }
            ],
        }
        feature_table = FeatureMaterializer(
            FeatureMaterializationConfig.model_validate_json(json.dumps(common))
        ).iceberg_table
        label_common = common | {
            "datasets": [
                {
                    **common["datasets"][0],
                    "input_window": {
                        "lookback_partitions": 1,
                        "lookahead_partitions": 1,
                    },
                }
            ]
        }
        label_table = LabelMaterializer(
            LabelMaterializationConfig.model_validate_json(
                json.dumps(label_common)
            )
        ).iceberg_table

        self.assertEqual(
            feature_table.schema,
            {
                "timestamp": "timestamp",
                "symbol": "string",
                "feature_value": "double",
            },
        )
        self.assertEqual(
            label_table.schema,
            {
                "timestamp": "timestamp",
                "symbol": "string",
                "label_value": "double",
            },
        )
        self.assertEqual(
            feature_table.partition_type, label_table.partition_type
        )
        self.assertEqual(feature_table.resource_type, "feature")
        self.assertEqual(label_table.resource_type, "label")


class PublicExceptionContractTest(unittest.TestCase):
    """Verify callers can import every public data-SDK failure type from one module."""

    def test_public_exception_types_share_the_data_sdk_root(self) -> None:
        """The exceptions entrypoint exposes one typed hierarchy for callers."""
        from ml_platform.data_sdk import exceptions

        for exception_name in exceptions.__all__:
            exception_type = getattr(exceptions, exception_name)
            self.assertTrue(
                issubclass(exception_type, exceptions.DataSdkError),
                f"{exception_name} must inherit DataSdkError.",
            )
