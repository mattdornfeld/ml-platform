"""Dataset worker entrypoint dispatched by the materialization API."""

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
import os

from ml_platform.data_sdk.dataset.market_data import DATA_ROOT, materialize_market_data_with_receipts
from ml_platform.sdk.configs import DatasetType, MarketDataConfig


def materialize_dataset(
    dataset_type: DatasetType,
    config: dict[str, Any],
    *,
    warehouse_dir: str | Path = DATA_ROOT,
    partition_callbacks: list[dict[str, Any]] | None = None,
) -> int:
    """Materialize one typed dataset definition through its Spark implementation."""
    if dataset_type is DatasetType.MARKET_DATA:
        result = materialize_market_data_with_receipts(MarketDataConfig.model_validate({"dataset_type": dataset_type, **config}), warehouse_dir=warehouse_dir, requested_timestamps=[str(callback["timestamp"]) for callback in partition_callbacks or []])
        row_count = result.row_count
        for callback in partition_callbacks or []:
            timestamp = str(callback["timestamp"])
            body = json.dumps({"content_hash": result.partition_hashes[timestamp], "snapshot_id": result.snapshot_id}).encode()
            request = Request(f"{os.environ.get('MATERIALIZATION_API_URL', 'http://ml-control-plane:8000')}/dataset-partition-update-requests/{callback['partition_id']}/complete", data=body, headers={"Content-Type": "application/json", "callback-token": callback["callback_token"]}, method="POST")
            with urlopen(request, timeout=30):
                pass
        return row_count
    raise ValueError(f"Unsupported dataset type: {dataset_type}.")



def main() -> None:
    """Run one serialized dataset definition in a dedicated Spark worker process."""
    parser = argparse.ArgumentParser(description="Materialize one registered dataset.")
    parser.add_argument("--dataset-type", required=True, choices=[item.value for item in DatasetType])
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--warehouse-dir", default=str(DATA_ROOT))
    parser.add_argument("--partition-callbacks-json", default="[]")
    args = parser.parse_args()
    row_count = materialize_dataset(
        DatasetType(args.dataset_type),
        json.loads(args.config_json),
        warehouse_dir=args.warehouse_dir,
        partition_callbacks=json.loads(args.partition_callbacks_json),
    )
    print(f"Materialized {row_count:,} dataset rows.")


if __name__ == "__main__":
    main()
