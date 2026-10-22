"""Runtime configs for Spark integration fixtures that write Iceberg tables."""

from pathlib import Path
import json

from pyspark.sql import DataFrame, SparkSession

from ml_platform.data_sdk.materializers import DatasetMaterializationConfig, DatasetMaterializer
from ml_platform.data_sdk.transforms._dataset_transform import DatasetTransform
from ml_platform.data_sdk.transforms._types import FeatureKey
from ml_platform.data_sdk.utils import record


def market_config(warehouse: Path, *, name: str = "sp500", version: int = 3) -> DatasetMaterializationConfig:
    """Describe the synthetic market table and its normal resource properties."""
    return DatasetMaterializationConfig.model_validate_json(json.dumps({
        "name": name,
        "version": version,
        "description": "Synthetic hourly market bars for Spark integration tests.",
        "dataset_type": "market_data", "keys": ["symbol"], "partition_type": "daily",
        "symbols": ["AAA"],
        "warehouse_dir": str(warehouse),
        "partitions": [{
            "partition_id": "synthetic-market-day",
            "timestamp": "2026-01-02T00:00:00Z",
            "callback_token": "synthetic-market-token",
        }],
    }))


@record
class StaticMarketDataTransform(DatasetTransform):
    """Return synthetic typed market data through the DatasetTransform contract."""

    data_frame: DataFrame

    @property
    def value_schema(self) -> dict[str, str]:
        """Declare the fixed market-data value columns."""
        return {
            "market_session": "string",
            "Open": "double",
            "High": "double",
            "Low": "double",
            "Close": "double",
            "Volume": "bigint",
        }

    def __call__(self) -> DataFrame:
        """Return the synthetic transform output unchanged."""
        return self.data_frame


def materialize_market_data(
    spark: SparkSession,
    data_frame: DataFrame,
    config: DatasetMaterializationConfig,
) -> int:
    """Persist synthetic market output through the production dataset lifecycle."""
    materializer = DatasetMaterializer(config)
    transform = StaticMarketDataTransform(
        keys=[FeatureKey.SYMBOL],
        partition_type=config.partition_type,
        data_frame=data_frame,
    )
    requested_partitions = materializer._requested_partitions(spark)
    materialized_data_frame = materializer._filter_partitions(transform(), requested_partitions)
    materializer.validate(materialized_data_frame)
    materializer.iceberg_table.create(spark)
    materializer.iceberg_table.write(materialized_data_frame)
    return materialized_data_frame.count()
