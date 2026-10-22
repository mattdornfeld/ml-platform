"""Private common contracts and Iceberg plumbing for API-dispatched materializers."""

from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping
from urllib.request import Request, urlopen

from pydantic import Field, field_validator

from ml_platform.common.types import BaseConfig
from ml_platform.data_sdk.materializers._constants import materialization_api_url, spark_master
from ml_platform.data_sdk.materializers._materialization_validator import (
    AllPartitionsTemporalFilterConfig,
    DiscriminatedValidationConfig,
    MaterializationValidator,
    MinimumRowsValidationConfig,
    RequestedPartitionsValidationConfig,
    TemporalFilter,
    ValidationConfig,
    ValidationConfigs,
)
from ml_platform.data_sdk.transforms._base import DataTransform
from ml_platform.data_sdk.transforms._types import DatasetDependency, PartitionType
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


class PartionDependency(BaseConfig):
    """One pinned parent dataset partition required by a derived output."""

    partition_id: str = Field(min_length=1, description="Control-plane identifier of the pinned parent partition.")
    dataset_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$",
                              description="Name of the parent dataset owning this partition.")
    dataset_version: int = Field(ge=1, description="Version of the parent dataset owning this partition.")
    timestamp: datetime = Field(description="UTC timestamp that identifies the parent logical partition.")
    snapshot_id: int = Field(ge=0, description="Iceberg snapshot containing the pinned parent partition revision.")


class MaterializerPartition(BaseConfig):
    """One requested output partition and its authenticated completion receipt."""

    partition_id: str = Field(min_length=1, description="Control-plane identifier for the requested output partition.")
    timestamp: datetime = Field(description="UTC timestamp that identifies the requested logical output partition.")
    callback_token: str = Field(min_length=1,
                                description="Opaque token required to report successful partition completion.")
    partition_dependencies: list[PartionDependency] = Field(
        default_factory=list,
        description="Pinned parent dataset partition revisions required to produce this output partition.",
    )


class MaterializerConfig(BaseConfig):
    """Strict immutable API runtime contract shared by all materializers."""

    @classmethod
    def iceberg_catalog(cls) -> str:
        """Return the shared Iceberg catalog used by all materializer jobs."""
        return "ml_platform"

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$", description="Stable lowercase resource name.")
    version: int = Field(ge=1, description="Positive immutable version number for the resource definition.")
    description: str = Field(min_length=1, description="Human-readable description of the materialized resource.")
    validations: list[DiscriminatedValidationConfig] = Field(
        default_factory=list,
        description="Declared runtime validation checks applied to materialized output.",
    )
    datasets: list[DatasetDependency] = Field(
        default_factory=list, description="Versioned upstream datasets required by this resource."
    )
    partitions: list[MaterializerPartition] = Field(
        min_length=1, description="Logical output partitions this job must materialize."
    )
    warehouse_dir: str = Field(min_length=1, description="Iceberg warehouse URI or local directory for output tables.")

    @field_validator("validations")
    @classmethod
    def _unique_validation_types(cls, validations: list[ValidationConfig]) -> list[ValidationConfig]:
        """Reject ambiguous YAML overrides before a materializer consumes this config."""
        validation_types = {validation.type for validation in validations}
        if len(validation_types) != len(validations):
            raise ValueError("Validation overrides must use unique types.")
        return validations

    @property
    def validation_overrides_by_type(self) -> dict[str, ValidationConfig]:
        """Return the already-validated YAML overrides by their discriminator."""
        return {validation.type: validation for validation in self.validations}

    @classmethod
    def _iceberg_table_name_for(cls, name: str, version: int) -> str:
        """Return the physical table name for an immutable resource version."""
        return f"{name}-v{version}"

    @property
    def _iceberg_table_name(self) -> str:
        """Return the physical table name for this immutable resource version."""
        return self._iceberg_table_name_for(self.name, self.version)

    def _partition_timestamps_for(self, partition_type: PartitionType) -> tuple[datetime, ...]:
        """Return requested timestamps normalized to an explicit partition cadence."""
        return tuple(
            partition_type.normalize_timestamp(partition.timestamp)
            for partition in self.partitions
        )


class WriteMode(StrEnum):
    """Iceberg persistence strategies available to materializer implementations."""

    MERGE = "merge"
    OVERWRITE = "overwrite"


@dataclass(frozen=True)
class Materializer[Config: MaterializerConfig](ABC):
    """Shared Spark and Iceberg lifecycle for one API-delivered materialization."""
    config: Config

    @classmethod
    @abstractmethod
    def config_model(cls) -> type[Config]:
        """Return the strict runtime configuration model for this worker."""

    @classmethod
    @abstractmethod
    def spark_application_name(cls) -> str:
        """Return the stable Spark application name for this worker type."""

    @classmethod
    @abstractmethod
    def iceberg_namespace(cls) -> str:
        """Return the Iceberg namespace owned by this materializer type."""

    @classmethod
    @abstractmethod
    def callback_resource(cls) -> str:
        """Return the control-plane resource type completed by this worker."""

    @classmethod
    @abstractmethod
    def write_mode(cls) -> WriteMode:
        """Return the declared Iceberg persistence strategy for this resource."""

    @classmethod
    def iceberg_table_properties(cls) -> Mapping[str, str]:
        """Return shared immutable Iceberg storage settings."""
        return {
            "format-version": "2",
            "write.format.default": "parquet",
            "write.parquet.compression-codec": "zstd",
            "write.distribution-mode": "hash",
            "write.spark.fanout.enabled": "false",
        }

    @classmethod
    def _config_from_json(cls, document: str) -> Config:
        """Deserialize and validate one materializer-specific job configuration."""
        return cls.config_model().model_validate_json(document)

    @classmethod
    def _iceberg_table_identifier(cls, name: str, version: int) -> str:
        """Resolve an immutable resource version to its Iceberg identifier."""
        namespace = cls.iceberg_namespace()
        table = cls.config_model()._iceberg_table_name_for(name, version)
        return f"{cls.config_model().iceberg_catalog()}.{namespace}.`{table}`"

    @property
    def _table_identifier(self) -> str:
        """Return this materializer's output Iceberg table identifier."""
        return self._iceberg_table_identifier(self.config.name, self.config.version)

    @classmethod
    def _create_spark_session(cls, warehouse_dir: str | Path) -> SparkSession:
        """Create the UTC Iceberg Spark session shared by all materializer workers."""
        warehouse = str(warehouse_dir)
        if "://" not in warehouse:
            warehouse = str(Path(warehouse).resolve())
        builder = SparkSession.builder.appName(cls.spark_application_name())
        if master := spark_master():
            builder = builder.master(master)
        catalog = cls.config_model().iceberg_catalog()
        return (
            builder.config("spark.driver.memory", "8g")
            .config("spark.executor.memory", "8g")
            .config("spark.sql.shuffle.partitions", "64")
            .config("spark.scheduler.mode", "FAIR")
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
            .config(f"spark.sql.catalog.{catalog}", "org.apache.iceberg.spark.SparkCatalog")
            .config(f"spark.sql.catalog.{catalog}.type", "hadoop")
            .config(f"spark.sql.catalog.{catalog}.warehouse", warehouse)
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )

    @classmethod
    def _sql_literal(cls, value: str) -> str:
        """Quote a Spark SQL string literal used for table property changes."""
        return "'" + value.replace("'", "''") + "'"

    def _create_table(self, spark: SparkSession, *, schema: str, partition_spec: str) -> None:
        """Create this materialization's Iceberg table or reconcile its metadata."""
        if not schema or not partition_spec:
            raise NotImplementedError(f"{type(self).__name__} must declare an Iceberg schema and partition spec.")
        properties = self._table_properties(partition_spec)
        identifier = self._table_identifier
        if spark.catalog.tableExists(identifier):
            existing_properties = {
                row.key: row.value for row in spark.sql(f"SHOW TBLPROPERTIES {identifier}").collect()
            }
            # Spark redacts some property values (including description and frequency).
            # The resource type marks an atomic write of this immutable definition.
            if existing_properties.get("resource.type") != properties["resource.type"]:
                resource_type = properties["resource.type"]
                obsolete_keys = {
                    f"{resource_type}.{field}"
                    for field in ("name", "version", "frequency", "description", "validations", "upstream_spec")
                }.intersection(existing_properties)
                if obsolete_keys:
                    keys = ", ".join(self._sql_literal(key) for key in sorted(obsolete_keys))
                    spark.sql(f"ALTER TABLE {identifier} UNSET TBLPROPERTIES ({keys})")
                assignments = ", ".join(
                    f"{self._sql_literal(key)} = {self._sql_literal(value)}"
                    for key, value in properties.items()
                )
                spark.sql(f"ALTER TABLE {identifier} SET TBLPROPERTIES ({assignments})")
            return
        catalog = self.config.iceberg_catalog()
        namespace = self.iceberg_namespace()
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.{namespace}")
        writer = spark.createDataFrame([], schema).writeTo(identifier).using("iceberg")
        for key, value in properties.items():
            writer = writer.tableProperty(key, value)
        writer.partitionedBy(self._partition_transform(partition_spec)).create()

    def _additional_table_properties(self) -> Mapping[str, str]:
        """Return optional storage metadata specific to one resource type."""
        return {}

    def _table_properties(self, partition_spec: str) -> Mapping[str, str]:
        """Return storage settings and the shared resource definition metadata."""
        frequency = {"day": "daily", "hour": "hourly"}[self._partition_unit(partition_spec)]
        return self.iceberg_table_properties() | {
            "resource.type": self.callback_resource(),
            "resource.name": self.config.name,
            "resource.version": str(self.config.version),
            "resource.frequency": frequency,
            "resource.description": self.config.description,
            "resource.dependencies": json.dumps(
                [dataset.model_dump(mode="json") for dataset in self.config.datasets], sort_keys=True
            ),
            "resource.validations": json.dumps(
                [validation.model_dump(mode="json") for validation in self.config.validations],
                sort_keys=True,
            ),
        } | self._additional_table_properties()

    def _prepare_frame(self, data_frame: DataFrame) -> DataFrame:
        """Cast or normalize a transform result before validation and persistence."""
        return data_frame

    def _write_to_table(self, data_frame: DataFrame) -> None:
        """Merge keyed rows or replace touched Iceberg partitions atomically."""
        if self.write_mode() is WriteMode.OVERWRITE:
            data_frame.writeTo(self._table_identifier).overwritePartitions()
            return
        raise NotImplementedError(f"{type(self).__name__} must implement merge writes.")

    @staticmethod
    def _merge_frame(data_frame: DataFrame, identifier: str, keys: tuple[str, ...]) -> None:
        """Upsert a prepared DataFrame by its resource-specific row key."""
        if not keys:
            raise ValueError("An Iceberg merge requires at least one key column.")

        source = data_frame.alias("source")
        predicate = F.col(f"source.{keys[0]}") == F.col(f"{identifier}.{keys[0]}")
        for key in keys[1:]:
            predicate = predicate & (F.col(f"source.{key}") == F.col(f"{identifier}.{key}"))
        (
            source.mergeInto(identifier, predicate)
            .whenMatched().updateAll()
            .whenNotMatched().insertAll()
            .merge()
        )

    def _partition_interval(self, partition_spec: str) -> timedelta:
        """Return the fixed duration represented by one output partition."""
        return PartitionType.from_iceberg_partition_spec(partition_spec).partition_interval

    @staticmethod
    def _partition_unit(partition_spec: str) -> str:
        """Return the Spark timestamp truncation unit for the Iceberg partition spec."""
        return PartitionType.from_iceberg_partition_spec(partition_spec).partition_unit

    def _partition_timestamp(self, timestamp: datetime, partition_spec: str) -> datetime:
        """Normalize a requested timestamp to its UTC Iceberg partition boundary."""
        partition_timestamp = timestamp.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        if self._partition_unit(partition_spec) == "day":
            partition_timestamp = partition_timestamp.replace(hour=0)
        return partition_timestamp

    def _target_partition_keys(self, partition_spec: str) -> tuple[str, ...]:
        """Return stable string keys for each requested output partition."""
        return tuple(
            self._partition_timestamp(partition.timestamp, partition_spec).strftime("%Y-%m-%dT%H:%M:%S")
            for partition in self.config.partitions
        )

    def _row_digest(self) -> F.Column:
        """Hash the persisted columns that define one output row."""
        raise NotImplementedError(f"{type(self).__name__} must implement row content hashing.")

    def _partition_content_hashes(self, spark: SparkSession, partition_spec: str) -> dict[str, str]:
        """Hash requested logical partitions from the committed Iceberg table."""
        partition_key = F.date_format(
            F.date_trunc(self._partition_unit(partition_spec), F.col("timestamp")),
            "yyyy-MM-dd'T'HH:mm:ss",
        )
        rows = (
            spark.table(self._table_identifier)
            .withColumn("partition_key", partition_key)
            .where(F.col("partition_key").isin(sorted(set(self._target_partition_keys(partition_spec)))))
            .groupBy("partition_key")
            .agg(F.sha2(F.concat_ws("", F.sort_array(F.collect_list(self._row_digest()))), 256).alias("content_hash"))
            .collect()
        )
        return {row["partition_key"]: row["content_hash"] for row in rows}

    def _partition_row_counts(self, spark: SparkSession, partition_spec: str) -> dict[str, int]:
        """Count persisted rows in each requested logical partition."""
        partition_key = F.date_format(
            F.date_trunc(self._partition_unit(partition_spec), F.col("timestamp")),
            "yyyy-MM-dd'T'HH:mm:ss",
        )
        rows = (
            spark.table(self._table_identifier)
            .withColumn("partition_key", partition_key)
            .where(F.col("partition_key").isin(sorted(set(self._target_partition_keys(partition_spec)))))
            .groupBy("partition_key")
            .count()
            .collect()
        )
        return {row["partition_key"]: int(row["count"]) for row in rows}

    def _partition_receipt_data(self, partition: MaterializerPartition) -> dict[str, object]:
        """Allow derived materializers to add their immutable input revisions."""
        return {}

    def _complete_partitions(self, spark: SparkSession, snapshot_id: int, partition_spec: str) -> None:
        """Report committed partition revisions through the control-plane API."""
        target_partition_keys = self._target_partition_keys(partition_spec)
        content_hashes = self._partition_content_hashes(spark, partition_spec)
        row_counts = self._partition_row_counts(spark, partition_spec)
        table_schema = {
            field.name: field.dataType.simpleString()
            for field in spark.table(self._table_identifier).schema.fields
        }
        empty_hash = sha256(b"").hexdigest()
        receipts = []
        for partition, partition_key in zip(self.config.partitions, target_partition_keys, strict=True):
            receipts.append({
                "partition_id": partition.partition_id,
                "resource_type": self.callback_resource(),
                "row_count": row_counts.get(partition_key, 0),
                "schema": table_schema,
                "content_hash": content_hashes.get(partition_key, empty_hash),
                "snapshot_id": snapshot_id,
                "callback_token": partition.callback_token,
                **self._partition_receipt_data(partition),
            })
        request = Request(
            f"{materialization_api_url()}/partitions/complete",
            data=json.dumps(receipts).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=30):
            pass

    @staticmethod
    def _partition_transform(partition_spec: str) -> F.Column:
        """Map the supported Iceberg partition declaration to a PySpark transform."""
        transform, column = partition_spec.removesuffix(")").split("(", maxsplit=1)
        transforms = {"days": F.days, "hours": F.hours}
        try:
            return transforms[transform](column)
        except KeyError as error:
            raise ValueError(f"Unsupported Iceberg partition transform: {partition_spec}.") from error

    @staticmethod
    def _latest_snapshot_id(spark: SparkSession, identifier: str) -> int:
        """Return the latest Iceberg snapshot committed for a table."""
        snapshot = (
            spark.table(f"{identifier}.snapshots")
            .orderBy(F.col("committed_at").desc())
            .select("snapshot_id")
            .first()
        )
        if snapshot is None:
            raise RuntimeError("Iceberg materialization completed without a snapshot.")
        return int(snapshot["snapshot_id"])

    @staticmethod
    def _stop_spark_session(spark: SparkSession) -> None:
        """Stop a worker Spark session after its materialization completes."""
        spark.stop()

    @abstractmethod
    def _create_transform(self, spark: SparkSession) -> DataTransform:
        """Instantiate the resource's declared transform for this Spark job."""

    def _iceberg_layout(self, transform: DataTransform) -> tuple[str, str]:
        """Return the output layout declared by a dataset transform."""
        return transform.iceberg_schema, transform.iceberg_partition_spec

    @property
    @abstractmethod
    def _partition_timestamps(self) -> tuple[datetime, ...]:
        """Return requested output partition boundaries for validation."""

    @property
    @abstractmethod
    def _partition_type(self) -> PartitionType:
        """Return the explicit cadence of requested output partitions."""

    @property
    def _minimum_rows_temporal_filter(self) -> TemporalFilter:
        """Select every requested partition for the shared minimum-row check."""
        return AllPartitionsTemporalFilterConfig()

    @property
    def _minimum_rows_reason(self) -> str:
        """Explain why selected output partitions must contain at least one row."""
        return "Each selected output partition must contain at least one row."

    @property
    def _base_validation_configs(self) -> tuple[ValidationConfig, ...]:
        """Declare partition validations shared by every materialization."""
        return (
            RequestedPartitionsValidationConfig(
                reason="Output rows must belong to the requested logical partitions.",
                timestamps=self._partition_timestamps,
                partition_type=self._partition_type,
            ),
            MinimumRowsValidationConfig(
                reason=self._minimum_rows_reason,
                minimum_rows=1,
                timestamps=self._partition_timestamps,
                partition_type=self._partition_type,
                temporal_filter=self._minimum_rows_temporal_filter,
            ),
        )

    @property
    @abstractmethod
    def validation_configs(self) -> tuple[ValidationConfig, ...]:
        ...

    def validate(self, data_frame: DataFrame) -> None:
        """Run this materialization's complete resolved validation sequence."""
        MaterializationValidator(
            validation_configs=ValidationConfigs(
                default_validation_configs=self._base_validation_configs + self.validation_configs,
                override_validation_configs=tuple(self.config.validations),
            ),
        ).validate(data_frame)

    @abstractmethod
    def materialize(self, spark: SparkSession, transform: DataTransform) -> DataFrame:
        """Construct the resource DataFrame without writing or reporting it."""

    @classmethod
    def main(cls) -> None:
        """Run one serialized materialization job from the standard CLI contract."""
        parser = argparse.ArgumentParser(description=f"Materialize one {cls.__name__} job.")
        parser.add_argument("--config-json", required=True, help="Serialized materializer configuration.")
        args = parser.parse_args()
        config = cls._config_from_json(args.config_json)
        materializer = cls(config)
        spark_session = materializer._create_spark_session(config.warehouse_dir)
        try:
            transform = materializer._create_transform(spark_session)
            schema, partition_spec = materializer._iceberg_layout(transform)
            data_frame = materializer._prepare_frame(materializer.materialize(spark_session, transform))
            materializer.validate(data_frame)
            materializer._create_table(spark_session, schema=schema, partition_spec=partition_spec)
            materializer._write_to_table(data_frame)
            snapshot_id = materializer._latest_snapshot_id(spark_session, materializer._table_identifier)
            materializer._complete_partitions(spark_session, snapshot_id, partition_spec)
        finally:
            materializer._stop_spark_session(spark_session)
        print("Materialization completed.")
