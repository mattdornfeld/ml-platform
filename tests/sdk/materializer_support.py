"""Runtime configs for Spark integration fixtures that write Iceberg tables."""

from pathlib import Path
import json

from ml_platform.data_sdk.materializers import DatasetMaterializationConfig


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
