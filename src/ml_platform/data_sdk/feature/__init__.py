"""Spark worker for one API-submitted feature materialization job."""

import argparse
import json
import os
from typing import Any
from urllib.request import Request, urlopen

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ml_platform.data_sdk.dataset.market_data import MARKET_DATASET_ID, create_spark_session
from ml_platform.sdk.configs import FeatureConfig
from ml_platform.sdk.datasets import Dataset
from ml_platform.sdk.feature_harness import (
    _ensure_feature_table,
    _feature_table_identifier,
    _instantiate_transform,
    _load_transform,
    _validate_feature_frame,
)


def _planned_market_data(spark: Any, callbacks: list[dict[str, Any]]) -> DataFrame:
    """Load precisely the Iceberg snapshots and day partitions pinned by the API."""
    planned = {
        (item["partition_id"], item["timestamp"], item["snapshot_id"])
        for callback in callbacks
        for item in callback["dataset_partitions"]
    }
    if not planned:
        raise ValueError("Feature job has no immutable dataset partition plan.")
    frames = [
        spark.read.option("versionAsOf", snapshot_id).table("data.market_data.hourly_ohlcv").where(
            F.to_date("timestamp") == F.to_date(F.lit(timestamp))
        )
        for _, timestamp, snapshot_id in sorted(planned, key=lambda item: (item[1], item[0]))
    ]
    source = frames[0]
    for frame in frames[1:]:
        source = source.unionByName(frame)
    return source.dropDuplicates()


def materialize_feature(config: dict[str, Any], callbacks: list[dict[str, Any]], warehouse_dir: str) -> None:
    """Write requested feature partitions using the API-pinned upstream revisions."""
    if len(config["datasets"]) != 1:
        raise ValueError("The current worker supports exactly one dataset dependency.")
    feature = FeatureConfig.model_validate({
        **config,
        "datasets": [{"dataset_id": MARKET_DATASET_ID, "num_lookback_partitions": config["datasets"][0]["num_lookback_partitions"]}],
        "metadata": {},
    })
    spark = create_spark_session(warehouse_dir)
    try:
        target_days = [callback["timestamp"][:10] for callback in callbacks]
        source = _planned_market_data(spark, callbacks)
        datasets = {"market_data": Dataset(feature.datasets[0], source)}
        transformed = _instantiate_transform(
            _load_transform(feature.transform), feature.parameters, datasets,
        )().where(F.to_date("timestamp").isin(target_days))
        requested_partitions = spark.createDataFrame(
            [(timestamp,) for timestamp in (callback["timestamp"] for callback in callbacks)],
            "output_partition string",
        ).select(F.to_timestamp("output_partition").alias("output_partition"))
        transformed = _validate_feature_frame(transformed, feature, requested_partitions, datasets)
        _ensure_feature_table(spark, feature)
        identifier = _feature_table_identifier(feature)
        row_digest = F.sha2(F.concat_ws("\u0001", *[F.coalesce(F.col(column).cast("string"), F.lit("")) for column in ("timestamp", feature.feature_key.value, "feature_value")]), 256)
        hashes = {
            row["day"]: row["content_hash"]
            for row in transformed.withColumn("day", F.date_format("timestamp", "yyyy-MM-dd")).groupBy("day").agg(
                F.sha2(F.concat_ws("", F.sort_array(F.collect_list(row_digest))), 256).alias("content_hash")
            ).collect()
        }
        missing = set(target_days).difference(hashes)
        if missing:
            raise ValueError(f"Feature transform produced no rows for requested partitions: {sorted(missing)}.")
        quoted_days = ", ".join(f"DATE '{day}'" for day in target_days)
        spark.sql(f"DELETE FROM {identifier} WHERE to_date(timestamp) IN ({quoted_days})")
        transformed.writeTo(identifier).append()
        snapshot_id = spark.sql(f"SELECT snapshot_id FROM {identifier}.history ORDER BY made_current_at DESC LIMIT 1").first()["snapshot_id"]
        for callback, day in zip(callbacks, target_days, strict=True):
            body = json.dumps({
                "content_hash": hashes[day], "snapshot_id": snapshot_id,
                "dataset_partition_ids": [item["partition_id"] for item in callback["dataset_partitions"]],
            }).encode()
            request = Request(
                f"{os.environ.get('MATERIALIZATION_API_URL', 'http://ml-control-plane:8000')}/feature-partition-update-requests/{callback['partition_id']}/complete",
                data=body, method="POST", headers={"Content-Type": "application/json", "callback-token": callback["callback_token"]},
            )
            with urlopen(request, timeout=30):
                pass
    finally:
        spark.stop()


def main() -> None:
    """Run one serialized feature job in Spark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-json", required=True)
    parser.add_argument("--partition-callbacks-json", required=True)
    parser.add_argument("--warehouse-dir", required=True)
    args = parser.parse_args()
    materialize_feature(json.loads(args.feature_json), json.loads(args.partition_callbacks_json), args.warehouse_dir)


if __name__ == "__main__":
    main()
