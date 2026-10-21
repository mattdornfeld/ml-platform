"""Concurrent, partition-aware materialization for configured feature transforms."""

import asyncio
from importlib import import_module
from inspect import Parameter, signature
import json
from pathlib import Path
from threading import Lock
from typing import cast, get_type_hints

from pyspark.sql import DataFrame, SparkSession, functions as F

from ml_platform.data_sdk.features.base import FeatureTransform

from ml_platform.sdk.configs import (
    DatasetMetadata,
    FeatureConfig,
    FeatureMaterializationConfig,
    load_feature_config,
    load_feature_materialization_config,
    list_feature_config_names,
    parse_feature_args,
)
from ml_platform.sdk.datasets import Dataset
from ml_platform.data_sdk.dataset.market_data import (
    DATA_ROOT,
    ICEBERG_CATALOG,
    ICEBERG_TABLE_IDENTIFIER,
    MARKET_DATASET_ID,
    create_spark_session,
)

FEATURE_NAMESPACE = "features"
FEATURE_METADATA_NAMESPACE = "feature_metadata"
FEATURE_VALUE_COLUMN = "feature_value"
FREQUENCY_TO_SPARK_UNIT = {"hourly": "hour", "daily": "day"}
FREQUENCY_TO_ICEBERG_TRANSFORM = {"hourly": "hours", "daily": "days"}
DEPENDENCY_MANIFEST_TABLE_IDENTIFIER = f"{ICEBERG_CATALOG}.{FEATURE_METADATA_NAMESPACE}.feature_dependencies"
_DEPENDENCY_WRITE_LOCK = Lock()


class FeatureHarness:
    """Materialize missing or stale feature partitions from declared datasets."""

    def __init__(self, spark: SparkSession, warehouse_dir: str | Path) -> None:
        """Bind the harness to a Spark session and Iceberg warehouse."""
        self.spark = spark
        self.warehouse_dir = Path(warehouse_dir)

    def create(self, config: FeatureConfig) -> int:
        """Backfill absent partitions and replace partitions with changed parents."""
        for dataset in config.datasets:
            table = _dataset_table_identifier(dataset)
            if not self.spark.catalog.tableExists(table):
                raise ValueError(f"Feature dataset table does not exist: {table}.")
        _ensure_feature_table(self.spark, config)
        _ensure_dependency_table(self.spark)
        output_partitions = _output_partitions(self.spark, config)
        dependencies = _feature_dependencies(self.spark, config, output_partitions)
        targets = _target_output_partitions(
            self.spark, config, output_partitions, dependencies
        ).localCheckpoint(eager=True)
        if not targets.limit(1).count():
            return 0
        target_dependencies = dependencies.join(targets, "output_partition", "inner").localCheckpoint(
            eager=True
        )
        materialization_targets = target_dependencies.select("output_partition").distinct()
        if not materialization_targets.limit(1).count():
            _delete_feature_partitions(self.spark, config, targets)
            _replace_dependency_manifest(self.spark, config, target_dependencies, targets)
            return 0
        datasets = _load_datasets(self.spark, config, target_dependencies)
        transformed = _instantiate_transform(
            _load_transform(config.transform), config.parameters, datasets
        )()
        feature_frame = _validate_feature_frame(
            transformed, config, materialization_targets, datasets
        )
        row_count = feature_frame.count()
        _delete_feature_partitions(self.spark, config, targets)
        if row_count:
            (
                feature_frame.withColumn("output_partition", _partition_key(config))
                .repartition("output_partition")
                .sortWithinPartitions("output_partition", "timestamp", config.feature_key.value)
                .drop("output_partition")
                .writeTo(_feature_table_identifier(config))
                .append()
            )
        _replace_dependency_manifest(self.spark, config, target_dependencies, targets)
        return row_count


async def materialize_features(config: FeatureMaterializationConfig) -> dict[str, int]:
    """Run configured feature backfills concurrently up to the job limit."""
    spark = create_spark_session(DATA_ROOT)
    semaphore = asyncio.Semaphore(config.max_concurrent_spark_jobs)

    async def materialize_one(feature_name: str) -> tuple[str, int]:
        async with semaphore:
            feature_config = load_feature_config(feature_name)
            row_count = await asyncio.to_thread(
                FeatureHarness(spark.newSession(), DATA_ROOT).create, feature_config
            )
            return feature_name, row_count

    try:
        completed = await asyncio.gather(*(materialize_one(name) for name in list_feature_config_names()))
    finally:
        spark.stop()
    return dict(completed)


def _output_partitions(spark: SparkSession, config: FeatureConfig) -> DataFrame:
    """Return all source and already-materialized output partition keys."""
    primary = _primary_dataset(config)
    source = spark.table(_dataset_table_identifier(primary)).select(
        _partition_key(config).alias("output_partition")
    ).distinct()
    identifier = _feature_table_identifier(config)
    if not spark.catalog.tableExists(identifier):
        return source
    completed = spark.table(identifier).select(_partition_key(config).alias("output_partition")).distinct()
    return source.unionByName(completed).distinct()


def _primary_dataset(config: FeatureConfig) -> DatasetMetadata:
    """Return the dataset that defines the feature's output partition domain."""
    return config.datasets[0]


def _dataset_table_identifier(dataset: DatasetMetadata) -> str:
    """Resolve a declared dataset ID to its Iceberg table identifier."""
    if dataset.dataset_id != MARKET_DATASET_ID:
        raise ValueError(f"Unsupported feature dataset: {dataset.dataset_id}.")
    return ICEBERG_TABLE_IDENTIFIER


def _dataset_argument_name(dataset: DatasetMetadata) -> str:
    """Map a declared dataset ID to its transform constructor argument."""
    if dataset.dataset_id != MARKET_DATASET_ID:
        raise ValueError(f"Unsupported feature dataset: {dataset.dataset_id}.")
    return "market_data"


def _load_datasets(
    spark: SparkSession, config: FeatureConfig, target_dependencies: DataFrame
) -> dict[str, Dataset]:
    """Load exactly the parent partitions declared for the target outputs."""
    datasets: dict[str, Dataset] = {}
    for dataset in config.datasets:
        argument_name = _dataset_argument_name(dataset)
        if argument_name in datasets:
            raise ValueError(f"Feature datasets resolve to duplicate constructor argument: {argument_name}.")
        parent_partitions = target_dependencies.where(
            F.col("dataset_id") == dataset.dataset_id
        ).select("input_partition").distinct()
        datasets[argument_name] = Dataset(
            metadata=dataset,
            frame=(
                spark.table(_dataset_table_identifier(dataset))
                .withColumn("input_partition", F.date_trunc("day", F.col("timestamp")))
                .join(parent_partitions, "input_partition", "inner")
                .drop("input_partition")
            ),
        )
    return datasets


def _feature_dependencies(
    spark: SparkSession, config: FeatureConfig, output_partitions: DataFrame
) -> DataFrame:
    """Hash every declared parent partition required by every output partition."""
    dependencies: DataFrame | None = None
    for dataset in config.datasets:
        parent_hashes = _dataset_partition_hashes(spark, dataset)
        upper = F.date_trunc("day", F.col("output_partition"))
        lower = F.date_sub(upper, dataset.num_lookback_partitions)
        mapped = (
            output_partitions.crossJoin(parent_hashes)
            .where((F.col("input_partition") >= lower) & (F.col("input_partition") <= upper))
            .withColumn("dataset_id", F.lit(dataset.dataset_id))
            .select("output_partition", "dataset_id", "input_partition", "content_hash")
        )
        dependencies = mapped if dependencies is None else dependencies.unionByName(mapped)
    assert dependencies is not None
    return dependencies


def _dataset_partition_hashes(spark: SparkSession, dataset: DatasetMetadata) -> DataFrame:
    """Compute deterministic content hashes for physical dataset partitions."""
    frame = spark.table(_dataset_table_identifier(dataset))
    row = F.to_json(F.struct(*(F.col(column).alias(column) for column in sorted(frame.columns))))
    return (
        frame.withColumn("input_partition", F.date_trunc("day", F.col("timestamp")))
        .withColumn("row_hash", F.sha2(row, 256))
        .groupBy("input_partition")
        .agg(F.sha2(F.concat_ws("", F.sort_array(F.collect_list("row_hash"))), 256).alias("content_hash"))
    )


def _target_output_partitions(
    spark: SparkSession,
    config: FeatureConfig,
    output_partitions: DataFrame,
    dependencies: DataFrame,
) -> DataFrame:
    """Return missing outputs plus existing outputs with changed dependency hashes."""
    identifier = _feature_table_identifier(config)
    completed = spark.table(identifier).select(_partition_key(config).alias("output_partition")).distinct()
    missing = output_partitions.join(completed, "output_partition", "left_anti")
    stored = spark.table(DEPENDENCY_MANIFEST_TABLE_IDENTIFIER).where(
        (F.col("feature_name") == config.name) & (F.col("feature_version") == config.version)
    ).select("output_partition", "dataset_id", "input_partition", "content_hash")
    changed = dependencies.join(
        stored, ["output_partition", "dataset_id", "input_partition", "content_hash"], "left_anti"
    ).select("output_partition").unionByName(
        stored.join(
            dependencies, ["output_partition", "dataset_id", "input_partition", "content_hash"], "left_anti"
        ).select("output_partition")
    ).distinct()
    return missing.unionByName(changed).distinct()


def _delete_feature_partitions(spark: SparkSession, config: FeatureConfig, targets: DataFrame) -> None:
    """Delete stale output partitions before replacing them with recomputed rows."""
    identifier = _feature_table_identifier(config)
    targets.createOrReplaceTempView("feature_output_partitions_to_replace")
    spark.sql(
        f"""MERGE INTO {identifier} AS target
        USING feature_output_partitions_to_replace AS source
        ON date_trunc('{_spark_partition_unit(config)}', target.timestamp) = source.output_partition
        WHEN MATCHED THEN DELETE"""
    )


def _ensure_dependency_table(spark: SparkSession) -> None:
    """Create the feature-output to dataset-partition dependency manifest."""
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {ICEBERG_CATALOG}.{FEATURE_METADATA_NAMESPACE}")
    spark.sql(
        f"""CREATE TABLE IF NOT EXISTS {DEPENDENCY_MANIFEST_TABLE_IDENTIFIER} (
            feature_name STRING,
            feature_version INT,
            output_partition TIMESTAMP,
            dataset_id STRING,
            input_partition TIMESTAMP,
            content_hash STRING
        ) USING iceberg
        PARTITIONED BY (feature_name, feature_version, dataset_id)
        TBLPROPERTIES ('format-version' = '2', 'write.format.default' = 'parquet')"""
    )


def _replace_dependency_manifest(
    spark: SparkSession, config: FeatureConfig, dependencies: DataFrame, targets: DataFrame
) -> None:
    """Replace only the dependency records for output partitions just materialized."""
    with _DEPENDENCY_WRITE_LOCK:
        targets.createOrReplaceTempView("feature_dependency_partitions_to_replace")
        spark.sql(
            f"""MERGE INTO {DEPENDENCY_MANIFEST_TABLE_IDENTIFIER} AS target
            USING feature_dependency_partitions_to_replace AS source
            ON target.feature_name = {_sql_string(config.name)}
               AND target.feature_version = {config.version}
               AND target.output_partition = source.output_partition
            WHEN MATCHED THEN DELETE"""
        )
        replacement = dependencies.withColumn("feature_name", F.lit(config.name)).withColumn(
            "feature_version", F.lit(config.version).cast("int")
        ).select(
            "feature_name", "feature_version", "output_partition", "dataset_id", "input_partition", "content_hash"
        )
        if replacement.limit(1).count():
            replacement.writeTo(DEPENDENCY_MANIFEST_TABLE_IDENTIFIER).append()

def _partition_key(config: FeatureConfig) -> F.Column:
    """Return the timestamp bucket used by the configured Iceberg partition transform."""
    return F.date_trunc(_spark_partition_unit(config), F.col("timestamp"))


def _load_transform(import_path: str) -> type[FeatureTransform]:
    """Load a FeatureTransform class referenced by dotted YAML metadata."""
    module_name, class_name = import_path.rsplit(".", maxsplit=1)
    try:
        transform = getattr(import_module(module_name), class_name)
    except (AttributeError, ImportError) as error:
        raise ValueError(f"Cannot load feature transform: {import_path}") from error
    if not isinstance(transform, type) or not issubclass(transform, FeatureTransform):
        raise ValueError(f"Feature transform must inherit FeatureTransform: {import_path}")
    return cast(type[FeatureTransform], transform)


def _instantiate_transform(
    transform_class: type[FeatureTransform],
    parameters: dict[str, object],
    datasets: dict[str, Dataset],
) -> FeatureTransform:
    """Validate declared datasets and YAML parameters, then instantiate a transform."""
    constructor = signature(transform_class)
    constructor_parameters = {
        name: parameter
        for name, parameter in constructor.parameters.items()
        if parameter.kind in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY}
    }
    annotations = get_type_hints(transform_class.__init__)
    dataset_parameters = {
        name for name in constructor_parameters if annotations.get(name) is Dataset
    }
    if dataset_parameters != set(datasets):
        raise ValueError(
            "Feature transform dataset discrepancy "
            f"(constructor: {', '.join(sorted(dataset_parameters)) or 'none'}; "
            f"declared: {', '.join(sorted(datasets)) or 'none'})."
        )
    parameter_definitions = {
        name: parameter
        for name, parameter in constructor_parameters.items()
        if name not in dataset_parameters
    }
    required = {
        name for name, parameter in constructor_parameters.items()
        if name not in dataset_parameters and parameter.default is Parameter.empty
    }
    missing = required.difference(parameters)
    unexpected = set(parameters).difference(parameter_definitions)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing: {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"unexpected: {', '.join(sorted(unexpected))}")
        raise ValueError(f"Feature transform parameter discrepancy ({'; '.join(details)}).")
    for name, value in parameters.items():
        expected_type = annotations.get(name)
        if isinstance(expected_type, type) and type(value) is not expected_type:
            raise ValueError(
                f"Feature transform parameter discrepancy (type for {name}: "
                f"expected {expected_type.__name__}, got {type(value).__name__})."
            )
    try:
        return transform_class(**datasets, **parameters)
    except TypeError as error:
        raise ValueError(
            f"Feature transform parameter discrepancy for {transform_class.__name__}: {error}"
        ) from error


def _spark_partition_unit(config: FeatureConfig) -> str:
    """Map feature frequency to Spark's timestamp-truncation unit."""
    return FREQUENCY_TO_SPARK_UNIT[config.frequency]


def _validate_feature_frame(
    frame: DataFrame,
    config: FeatureConfig,
    requested_partitions: DataFrame,
    datasets: dict[str, Dataset],
) -> DataFrame:
    """Validate one transform result before it can replace requested partitions."""
    feature_key = config.feature_key.value
    expected_columns = {"timestamp", feature_key, FEATURE_VALUE_COLUMN}
    actual_columns = set(frame.columns)
    if actual_columns != expected_columns:
        missing = expected_columns.difference(actual_columns)
        unexpected = actual_columns.difference(expected_columns)
        details = []
        if missing:
            details.append(f"missing: {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"unexpected: {', '.join(sorted(unexpected))}")
        raise ValueError(f"Feature transform must return exactly {sorted(expected_columns)} ({'; '.join(details)}).")

    typed = frame.select(
        F.col("timestamp").cast("timestamp").alias("timestamp"),
        F.col(feature_key).cast("string").alias(feature_key),
        F.col(FEATURE_VALUE_COLUMN).cast(config.dtype).alias(FEATURE_VALUE_COLUMN),
    )
    if typed.where(
        F.col("timestamp").isNull()
        | F.col(feature_key).isNull()
        | F.col(FEATURE_VALUE_COLUMN).isNull()
    ).limit(1).count():
        raise ValueError(
            "Feature transform must emit non-null, castable timestamp, "
            f"{feature_key}, and {FEATURE_VALUE_COLUMN} values."
        )
    if config.validations.require_finite_feature_value and typed.where(
        F.isnan(F.col(FEATURE_VALUE_COLUMN).cast("double"))
        | (F.col(FEATURE_VALUE_COLUMN).cast("double") == float("inf"))
        | (F.col(FEATURE_VALUE_COLUMN).cast("double") == float("-inf"))
    ).limit(1).count():
        raise ValueError("Feature transform emitted a non-finite feature_value.")
    if typed.groupBy("timestamp", feature_key).count().where(F.col("count") > 1).limit(1).count():
        raise ValueError(f"Feature transform emitted duplicate (timestamp, {feature_key}) rows.")

    partitioned = typed.withColumn("output_partition", _partition_key(config))
    emitted_partitions = partitioned.select("output_partition").distinct()
    missing_partitions = requested_partitions.select("output_partition").distinct().join(
        emitted_partitions, "output_partition", "left_anti"
    )
    unexpected_partitions = emitted_partitions.join(
        requested_partitions.select("output_partition").distinct(), "output_partition", "left_anti"
    )
    if missing_partitions.limit(1).count() or unexpected_partitions.limit(1).count():
        raise ValueError("Feature transform output partitions do not exactly match requested partitions.")

    rows_by_partition = partitioned.groupBy("output_partition").count()
    too_small = requested_partitions.select("output_partition").distinct().join(
        rows_by_partition, "output_partition", "left"
    ).fillna(0, subset=["count"]).where(
        F.col("count") < config.validations.min_rows_per_partition
    )
    if too_small.limit(1).count():
        raise ValueError(
            "Feature transform emitted fewer than "
            f"{config.validations.min_rows_per_partition} rows for a requested partition."
        )

    if config.validations.require_feature_keys_from_input:
        input_keys: DataFrame | None = None
        for dataset in datasets.values():
            if feature_key not in dataset.frame.columns:
                raise ValueError(f"Input dataset does not contain feature key column: {feature_key}.")
            keys = dataset.frame.select(F.col(feature_key).cast("string").alias(feature_key)).distinct()
            input_keys = keys if input_keys is None else input_keys.unionByName(keys)
        assert input_keys is not None
        unknown_keys = partitioned.select(feature_key).distinct().join(
            input_keys.distinct(), feature_key, "left_anti"
        )
        if unknown_keys.limit(1).count():
            raise ValueError(f"Feature transform emitted {feature_key} values absent from input datasets.")
    return partitioned.drop("output_partition")


def _feature_table_identifier(config: FeatureConfig) -> str:
    """Return an Iceberg identifier unique to the feature name and version."""
    return f"{ICEBERG_CATALOG}.{FEATURE_NAMESPACE}.{config.name}_v{config.version}"


def _feature_storage_path(warehouse_dir: str | Path, config: FeatureConfig) -> Path:
    """Return the versioned feature directory under the configured data root."""
    return Path(warehouse_dir).resolve() / FEATURE_NAMESPACE / f"{config.name}_v{config.version}"


def _ensure_feature_table(spark: SparkSession, config: FeatureConfig) -> None:
    """Create the configured feature table and persist its metadata properties."""
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {ICEBERG_CATALOG}.{FEATURE_NAMESPACE}")
    properties = [
        "'format-version' = '2'", "'write.format.default' = 'parquet'", "'write.parquet.compression-codec' = 'zstd'",
        "'write.distribution-mode' = 'hash'", "'write.spark.fanout.enabled' = 'false'",
        f"'feature.name' = {_sql_string(config.name)}", f"'feature.version' = {_sql_string(config.version)}",
        f"'feature.dtype' = {_sql_string(config.dtype)}", f"'feature.frequency' = {_sql_string(config.frequency)}",
        f"'feature.feature_key' = {_sql_string(config.feature_key.value)}",
        f"'feature.description' = {_sql_string(config.description)}", f"'feature.transform' = {_sql_string(config.transform)}",
    ]
    properties.extend(f"'feature.metadata.{key}' = {_sql_string(value)}" for key, value in sorted(config.metadata.items()))
    properties.append(
        f"'feature.upstream_spec' = {_sql_string(json.dumps([dataset.model_dump() for dataset in config.datasets], sort_keys=True))}"
    )
    properties.append(
        f"'feature.validations' = {_sql_string(json.dumps(config.validations.model_dump(), sort_keys=True))}"
    )
    spark.sql(
        f"""CREATE TABLE IF NOT EXISTS {_feature_table_identifier(config)} (
            timestamp TIMESTAMP, {config.feature_key.value} STRING, feature_value {config.dtype.upper()}
        ) USING iceberg PARTITIONED BY ({FREQUENCY_TO_ICEBERG_TRANSFORM[config.frequency]}(timestamp))
        TBLPROPERTIES ({", ".join(properties)})"""
    )
    spark.sql(
        f"ALTER TABLE {_feature_table_identifier(config)} "
        f"SET TBLPROPERTIES ({', '.join(properties)})"
    )


def _sql_string(value: str | int) -> str:
    """Quote an Iceberg SQL string literal."""
    return f"'{str(value).replace(chr(39), chr(39) * 2)}'"


def main() -> None:
    """Run the configured concurrent feature-materialization application."""
    args = parse_feature_args()
    completed = asyncio.run(materialize_features(load_feature_materialization_config(args.config)))
    print("\n".join(f"Saved {rows:,} {name} feature rows." for name, rows in completed.items()))


if __name__ == "__main__":
    main()
