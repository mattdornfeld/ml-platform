"""Spark worker for one API-submitted feature materialization job."""
from importlib import import_module
from inspect import Parameter, signature
import json
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar, Literal, cast, get_type_hints
from urllib.request import Request, urlopen

from pydantic import Field, field_validator
from simple_classproperty import classproperty
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ml_platform.data_sdk.materializers._dataset import DatasetMaterializer
from ml_platform.data_sdk.materializers._base import Materializer, MaterializerConfig, MaterializerPartition
from ml_platform.data_sdk.materializers._constants import materialization_api_url
from ml_platform.data_sdk.transforms._feature_transform import FeatureTransform
from ml_platform.data_sdk.transforms._types import Dataset, DatasetDependency, DatasetReference


class FeatureMaterializationConfig(MaterializerConfig):
    """Complete JSON contract consumed by a feature Spark materializer."""

    dtype: Literal["double", "float", "int", "bigint"] = Field(description="Spark SQL type used to persist feature values.")
    frequency: Literal["hourly", "daily"] = Field(description="Cadence of the feature's logical output partitions.")
    feature_key: Literal["symbol"] = Field(default="symbol", description="Column that identifies the entity for each feature value.")
    datasets: list[DatasetDependency] = Field(min_length=1, description="Versioned dataset dependencies consumed by the feature transform.")
    transform: str = Field(
        pattern=r"^[a-z_][a-z0-9_.]*\.[A-Za-z_][A-Za-z0-9_]*$",
        description="Absolute import path of the FeatureTransform implementation.",
    )
    parameters: dict[str, int | float | str | bool] = Field(
        default_factory=dict, description="Typed keyword arguments passed to the feature transform constructor."
    )

    @field_validator("datasets")
    @classmethod
    def _unique_datasets(cls, datasets: list[DatasetDependency]) -> list[DatasetDependency]:
        if len({dataset.id for dataset in datasets}) != len(datasets):
            raise ValueError("Feature datasets must have unique name and version identities.")
        return datasets

    @field_validator("partitions")
    @classmethod
    def _require_input_partition_plans(cls, partitions: list[MaterializerPartition]) -> list[MaterializerPartition]:
        if any(not partition.dataset_partitions for partition in partitions):
            raise ValueError("Feature partitions must include pinned dataset_partitions.")
        return partitions




class FeatureMaterializer(Materializer[FeatureMaterializationConfig]):
    """Materialize API-pinned feature partitions with their immutable inputs."""

    config_model = FeatureMaterializationConfig
    spark_application_name: ClassVar[str] = "ml-platform-feature-materializer"
    iceberg_schema: ClassVar[str] = ""
    iceberg_partition_spec: ClassVar[str] = ""
    FEATURE_VALUE_COLUMN: ClassVar[str] = "feature_value"
    FREQUENCY_TO_SPARK_UNIT: ClassVar[dict[str, str]] = {"hourly": "hour", "daily": "day"}
    FREQUENCY_TO_ICEBERG_TRANSFORM: ClassVar[dict[str, str]] = {"hourly": "hours", "daily": "days"}
    _dependency_write_lock: ClassVar[Lock] = Lock()

    @classproperty
    def iceberg_namespace(cls) -> str:
        """Store feature resources in their dedicated Iceberg namespace."""
        return "features"

    @classproperty
    def _dependency_manifest_table_identifier(cls) -> str:
        """Return the shared feature-to-dataset dependency manifest identifier."""
        return f"{cls.config_model.iceberg_catalog}.{cls.iceberg_namespace}.dependencies"

    @staticmethod
    def _planned_market_data(spark: Any, callbacks: list[MaterializerPartition], dataset: DatasetDependency) -> DataFrame:
        """Load precisely the Iceberg snapshots and day partitions pinned by the API."""
        planned = {
            (item.partition_id, item.timestamp, item.snapshot_id)
            for callback in callbacks
            for item in callback.dataset_partitions
        }
        if not planned:
            raise ValueError("Feature job has no immutable dataset partition plan.")
        frames = [
            spark.read.option("versionAsOf", snapshot_id).table(FeatureMaterializer._dataset_table_identifier(dataset)).where(
                F.to_date("timestamp") == F.to_date(F.lit(timestamp))
            )
            for _, timestamp, snapshot_id in sorted(planned, key=lambda item: (item[1], item[0]))
        ]
        source = frames[0]
        for frame in frames[1:]:
            source = source.unionByName(frame)
        return source.dropDuplicates()

    @staticmethod
    def _materialize_definition(spark: SparkSession, config: FeatureMaterializationConfig) -> int:
        """Materialize one validated feature definition for local integration coverage."""
        for dataset in config.datasets:
            table = FeatureMaterializer._dataset_table_identifier(dataset)
            if not spark.catalog.tableExists(table):
                raise ValueError(f"Feature dataset table does not exist: {table}.")
        FeatureMaterializer._ensure_feature_table(spark, config)
        FeatureMaterializer._ensure_dependency_table(spark)
        _output_partitions = FeatureMaterializer._output_partitions(spark, config)
        dependencies = FeatureMaterializer._feature_dependencies(spark, config, _output_partitions)
        targets = FeatureMaterializer._target_output_partitions(spark, config, _output_partitions, dependencies).localCheckpoint(eager=True)
        if not targets.limit(1).count():
            return 0
        target_dependencies = dependencies.join(targets, "output_partition", "inner").localCheckpoint(eager=True)
        materialization_targets = target_dependencies.select("output_partition").distinct()
        if not materialization_targets.limit(1).count():
            FeatureMaterializer._delete_feature_partitions(spark, config, targets)
            FeatureMaterializer._replace_dependency_manifest(spark, config, target_dependencies, targets)
            return 0
        datasets = FeatureMaterializer._load_datasets(spark, config, target_dependencies)
        transformed = FeatureMaterializer._instantiate_transform(FeatureMaterializer._load_transform(config.transform), config.parameters, datasets)()
        feature_frame = FeatureMaterializer._validate_feature_frame(transformed, config, materialization_targets, datasets)
        row_count = feature_frame.count()
        FeatureMaterializer._delete_feature_partitions(spark, config, targets)
        if row_count:
            (feature_frame.withColumn("output_partition", FeatureMaterializer._partition_key(config))
                .repartition("output_partition")
                .sortWithinPartitions("output_partition", "timestamp", config.feature_key)
                .drop("output_partition")
                .writeTo(FeatureMaterializer._feature_table_identifier(config)).append())
        FeatureMaterializer._replace_dependency_manifest(spark, config, target_dependencies, targets)
        return row_count


    @staticmethod
    def _output_partitions(spark: SparkSession, config: FeatureMaterializationConfig) -> DataFrame:
        """Return all source and already-materialized output partition keys."""
        primary = FeatureMaterializer._primary_dataset(config)
        source = spark.table(FeatureMaterializer._dataset_table_identifier(primary)).select(
            FeatureMaterializer._partition_key(config).alias("output_partition")
        ).distinct()
        identifier = FeatureMaterializer._feature_table_identifier(config)
        if not spark.catalog.tableExists(identifier):
            return source
        completed = spark.table(identifier).select(FeatureMaterializer._partition_key(config).alias("output_partition")).distinct()
        return source.unionByName(completed).distinct()


    @staticmethod
    def _primary_dataset(config: FeatureMaterializationConfig) -> DatasetDependency:
        """Return the dataset that defines the feature's output partition domain."""
        return config.datasets[0]


    @staticmethod
    def _dataset_table_identifier(dataset: DatasetDependency) -> str:
        """Resolve a declared immutable dataset version to its physical table."""
        return DatasetMaterializer._iceberg_table_identifier(dataset.name, dataset.version)


    @staticmethod
    def _dataset_argument_name(dataset: DatasetDependency) -> str:
        """Map a declared dataset version to its transform constructor argument."""
        return "market_data"


    @staticmethod
    def _load_datasets(
        spark: SparkSession, config: FeatureMaterializationConfig, target_dependencies: DataFrame
    ) -> dict[str, Dataset]:
        """Load exactly the parent partitions declared for the target outputs."""
        datasets: dict[str, Dataset] = {}
        for dataset in config.datasets:
            argument_name = FeatureMaterializer._dataset_argument_name(dataset)
            if argument_name in datasets:
                raise ValueError(f"Feature datasets resolve to duplicate constructor argument: {argument_name}.")
            parent_partitions = target_dependencies.where(
                F.col("dataset_id") == dataset.id
            ).select("input_partition").distinct()
            datasets[argument_name] = Dataset(
                metadata=DatasetReference(dataset.name, dataset.version, dataset.num_lookback_partitions),
                frame=(
                    spark.table(FeatureMaterializer._dataset_table_identifier(dataset))
                    .withColumn("input_partition", F.date_trunc("day", F.col("timestamp")))
                    .join(parent_partitions, "input_partition", "inner")
                    .drop("input_partition")
                ),
            )
        return datasets


    @staticmethod
    def _feature_dependencies(
        spark: SparkSession, config: FeatureMaterializationConfig, _output_partitions: DataFrame
    ) -> DataFrame:
        """Hash every declared parent partition required by every output partition."""
        dependencies: DataFrame | None = None
        for dataset in config.datasets:
            parent_hashes = FeatureMaterializer._dataset_partition_hashes(spark, dataset)
            upper = F.date_trunc("day", F.col("output_partition"))
            lower = F.date_sub(upper, dataset.num_lookback_partitions)
            mapped = (
                _output_partitions.crossJoin(parent_hashes)
                .where((F.col("input_partition") >= lower) & (F.col("input_partition") <= upper))
                .withColumn("dataset_id", F.lit(dataset.id))
                .select("output_partition", "dataset_id", "input_partition", "content_hash")
            )
            dependencies = mapped if dependencies is None else dependencies.unionByName(mapped)
        assert dependencies is not None
        return dependencies


    @staticmethod
    def _dataset_partition_hashes(spark: SparkSession, dataset: DatasetDependency) -> DataFrame:
        """Compute deterministic content hashes for physical dataset partitions."""
        frame = spark.table(FeatureMaterializer._dataset_table_identifier(dataset))
        row = F.to_json(F.struct(*(F.col(column).alias(column) for column in sorted(frame.columns))))
        return (
            frame.withColumn("input_partition", F.date_trunc("day", F.col("timestamp")))
            .withColumn("row_hash", F.sha2(row, 256))
            .groupBy("input_partition")
            .agg(F.sha2(F.concat_ws("", F.sort_array(F.collect_list("row_hash"))), 256).alias("content_hash"))
        )


    @staticmethod
    def _target_output_partitions(
        spark: SparkSession,
        config: FeatureMaterializationConfig,
        _output_partitions: DataFrame,
        dependencies: DataFrame,
    ) -> DataFrame:
        """Return missing outputs plus existing outputs with changed dependency hashes."""
        identifier = FeatureMaterializer._feature_table_identifier(config)
        completed = spark.table(identifier).select(FeatureMaterializer._partition_key(config).alias("output_partition")).distinct()
        missing = _output_partitions.join(completed, "output_partition", "left_anti")
        stored = spark.table(FeatureMaterializer._dependency_manifest_table_identifier).where(
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


    @staticmethod
    def _delete_feature_partitions(spark: SparkSession, config: FeatureMaterializationConfig, targets: DataFrame) -> None:
        """Delete stale output partitions before replacing them with recomputed rows."""
        identifier = FeatureMaterializer._feature_table_identifier(config)
        targets.createOrReplaceTempView("feature_output_partitions_to_replace")
        spark.sql(
            f"""MERGE INTO {identifier} AS target
            USING feature_output_partitions_to_replace AS source
            ON date_trunc('{FeatureMaterializer._spark_partition_unit(config)}', target.timestamp) = source.output_partition
            WHEN MATCHED THEN DELETE"""
        )


    @staticmethod
    def _ensure_dependency_table(spark: SparkSession) -> None:
        """Create the feature-output to dataset-partition dependency manifest."""
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {FeatureMaterializationConfig.iceberg_catalog}.{FeatureMaterializer.iceberg_namespace}")
        spark.sql(
            f"""CREATE TABLE IF NOT EXISTS {FeatureMaterializer._dependency_manifest_table_identifier} (
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


    @staticmethod
    def _replace_dependency_manifest(
        spark: SparkSession, config: FeatureMaterializationConfig, dependencies: DataFrame, targets: DataFrame
    ) -> None:
        """Replace only the dependency records for output partitions just materialized."""
        with FeatureMaterializer._dependency_write_lock:
            targets.createOrReplaceTempView("feature_dependency_partitions_to_replace")
            spark.sql(
                f"""MERGE INTO {FeatureMaterializer._dependency_manifest_table_identifier} AS target
                USING feature_dependency_partitions_to_replace AS source
                ON target.feature_name = {FeatureMaterializer._sql_string(config.name)}
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
                replacement.writeTo(FeatureMaterializer._dependency_manifest_table_identifier).append()

    @staticmethod
    def _partition_key(config: FeatureMaterializationConfig) -> F.Column:
        """Return the timestamp bucket used by the configured Iceberg partition transform."""
        return F.date_trunc(FeatureMaterializer._spark_partition_unit(config), F.col("timestamp"))


    @staticmethod
    def _load_transform(import_path: str) -> type[FeatureTransform]:
        """Load a FeatureTransform class referenced by its dotted import path."""
        module_name, class_name = import_path.rsplit(".", maxsplit=1)
        try:
            transform = getattr(import_module(module_name), class_name)
        except (AttributeError, ImportError) as error:
            raise ValueError(f"Cannot load feature transform: {import_path}") from error
        if not isinstance(transform, type) or not issubclass(transform, FeatureTransform):
            raise ValueError(f"Feature transform must inherit FeatureTransform: {import_path}")
        return cast(type[FeatureTransform], transform)


    @staticmethod
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


    @staticmethod
    def _spark_partition_unit(config: FeatureMaterializationConfig) -> str:
        """Map feature frequency to Spark's timestamp-truncation unit."""
        return FeatureMaterializer.FREQUENCY_TO_SPARK_UNIT[config.frequency]


    @staticmethod
    def _validate_feature_frame(
        frame: DataFrame,
        config: FeatureMaterializationConfig,
        requested_partitions: DataFrame,
        datasets: dict[str, Dataset],
    ) -> DataFrame:
        """Validate one transform result before it can replace requested partitions."""
        feature_key = config.feature_key
        expected_columns = {"timestamp", feature_key, FeatureMaterializer.FEATURE_VALUE_COLUMN}
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
            F.col(FeatureMaterializer.FEATURE_VALUE_COLUMN).cast(config.dtype).alias(FeatureMaterializer.FEATURE_VALUE_COLUMN),
        )
        if typed.where(
            F.col("timestamp").isNull()
            | F.col(feature_key).isNull()
            | F.col(FeatureMaterializer.FEATURE_VALUE_COLUMN).isNull()
        ).limit(1).count():
            raise ValueError(
                "Feature transform must emit non-null, castable timestamp, "
                f"{feature_key}, and {FeatureMaterializer.FEATURE_VALUE_COLUMN} values."
            )
        if config.validations.require_finite_feature_value and typed.where(
            F.isnan(F.col(FeatureMaterializer.FEATURE_VALUE_COLUMN).cast("double"))
            | (F.col(FeatureMaterializer.FEATURE_VALUE_COLUMN).cast("double") == float("inf"))
            | (F.col(FeatureMaterializer.FEATURE_VALUE_COLUMN).cast("double") == float("-inf"))
        ).limit(1).count():
            raise ValueError("Feature transform emitted a non-finite feature_value.")
        if typed.groupBy("timestamp", feature_key).count().where(F.col("count") > 1).limit(1).count():
            raise ValueError(f"Feature transform emitted duplicate (timestamp, {feature_key}) rows.")

        partitioned = typed.withColumn("output_partition", FeatureMaterializer._partition_key(config))
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


    @staticmethod
    def _feature_table_identifier(config: FeatureMaterializationConfig) -> str:
        """Return an Iceberg identifier unique to the feature name and version."""
        return FeatureMaterializer._iceberg_table_identifier(config.name, config.version)


    @staticmethod
    def _feature_storage_path(warehouse_dir: str | Path, config: FeatureMaterializationConfig) -> Path:
        """Return the versioned feature directory under the configured data root."""
        return Path(warehouse_dir).resolve() / FeatureMaterializer.iceberg_namespace / config._iceberg_table_name


    @staticmethod
    def _ensure_feature_table(spark: SparkSession, config: FeatureMaterializationConfig) -> None:
        """Create the configured feature table and persist its definition properties."""
        properties = [
            "'format-version' = '2'", "'write.format.default' = 'parquet'", "'write.parquet.compression-codec' = 'zstd'",
            "'write.distribution-mode' = 'hash'", "'write.spark.fanout.enabled' = 'false'",
            f"'feature.name' = {FeatureMaterializer._sql_string(config.name)}", f"'feature.version' = {FeatureMaterializer._sql_string(config.version)}",
            f"'feature.dtype' = {FeatureMaterializer._sql_string(config.dtype)}", f"'feature.frequency' = {FeatureMaterializer._sql_string(config.frequency)}",
            f"'feature.feature_key' = {FeatureMaterializer._sql_string(config.feature_key)}",
            f"'feature.description' = {FeatureMaterializer._sql_string(config.description)}", f"'feature.transform' = {FeatureMaterializer._sql_string(config.transform)}",
        ]
        properties.append(
            f"'feature.upstream_spec' = {FeatureMaterializer._sql_string(json.dumps([dataset.model_dump() for dataset in config.datasets], sort_keys=True))}"
        )
        properties.append(
            f"'feature.validations' = {FeatureMaterializer._sql_string(json.dumps(config.validations.model_dump(), sort_keys=True))}"
        )
        FeatureMaterializer._ensure_iceberg_table(
            spark,
            name=config.name,
            version=config.version,
            schema=f"timestamp TIMESTAMP, {config.feature_key} STRING, feature_value {config.dtype.upper()}",
            partition_spec=f"{FeatureMaterializer.FREQUENCY_TO_ICEBERG_TRANSFORM[config.frequency]}(timestamp)",
            properties=tuple(properties),
        )
        spark.sql(
            f"ALTER TABLE {FeatureMaterializer._feature_table_identifier(config)} "
            f"SET TBLPROPERTIES ({', '.join(properties)})"
        )


    @staticmethod
    def _sql_string(value: str | int) -> str:
        """Quote an Iceberg SQL string literal."""
        return f"'{str(value).replace(chr(39), chr(39) * 2)}'"

    def materialize(self, spark: SparkSession) -> int:
        """Run the transform and complete all output partition callbacks."""
        if len(self.config.datasets) != 1:
            raise ValueError("The current worker supports exactly one dataset dependency.")
        target_days = [callback.timestamp.date().isoformat() for callback in self.config.partitions]
        source = FeatureMaterializer._planned_market_data(spark, self.config.partitions, self.config.datasets[0])
        input_dataset = self.config.datasets[0]
        datasets = {"market_data": Dataset(DatasetReference(input_dataset.name, input_dataset.version, input_dataset.num_lookback_partitions), source)}
        transformed = FeatureMaterializer._instantiate_transform(FeatureMaterializer._load_transform(self.config.transform), self.config.parameters, datasets)().where(F.to_date("timestamp").isin(target_days))
        requested_partitions = spark.createDataFrame([(item.timestamp,) for item in self.config.partitions], "output_partition string").select(F.to_timestamp("output_partition").alias("output_partition"))
        transformed = FeatureMaterializer._validate_feature_frame(transformed, self.config, requested_partitions, datasets)
        FeatureMaterializer._ensure_feature_table(spark, self.config)
        identifier = FeatureMaterializer._feature_table_identifier(self.config)
        row_digest = F.sha2(F.concat_ws("\u0001", *[F.coalesce(F.col(column).cast("string"), F.lit("")) for column in ("timestamp", self.config.feature_key, "feature_value")]), 256)
        hashes = {row["day"]: row["content_hash"] for row in transformed.withColumn("day", F.date_format("timestamp", "yyyy-MM-dd")).groupBy("day").agg(F.sha2(F.concat_ws("", F.sort_array(F.collect_list(row_digest))), 256).alias("content_hash")).collect()}
        missing = set(target_days).difference(hashes)
        if missing:
            raise ValueError(f"Feature transform produced no rows for requested partitions: {sorted(missing)}.")
        quoted_days = ", ".join(f"DATE '{day}'" for day in target_days)
        spark.sql(f"DELETE FROM {identifier} WHERE to_date(timestamp) IN ({quoted_days})")
        transformed.writeTo(identifier).append()
        snapshot_id = self._latest_snapshot_id(spark, identifier)
        for callback, day in zip(self.config.partitions, target_days, strict=True):
            body = json.dumps({"content_hash": hashes[day], "snapshot_id": snapshot_id, "dataset_partition_ids": [item.partition_id for item in callback.dataset_partitions]}).encode()
            request = Request(f"{materialization_api_url()}/feature-partition-update-requests/{callback.partition_id}/complete", data=body, method="POST", headers={"Content-Type": "application/json", "callback-token": callback.callback_token})
            with urlopen(request, timeout=30):
                pass
        return transformed.count()


if __name__ == "__main__":
    FeatureMaterializer.main()
