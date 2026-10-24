"""Private common contracts and Iceberg plumbing for API-dispatched materializers."""

import argparse
import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path

import httpx2
from pydantic import Field, field_validator
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as spark_functions
from pyspark.sql.column import Column

from ml_platform.common.api.routes import PartitionRoutes
from ml_platform.common.types import (
    BaseConfig,
    CallbackToken,
    DatasetName,
    Description,
    PartitionTimestamp,
    PartitionType,
    ResourceName,
    ResourceVersion,
    RowCount,
    SnapshotId,
)
from ml_platform.data_sdk.materializers._constants import (
    materialization_api_url,
    spark_master,
)
from ml_platform.data_sdk.materializers._iceberg_table import IcebergTable
from ml_platform.data_sdk.materializers._materialization_validator import (
    DiscriminatedValidationConfig,
    MaterializationValidator,
    RequestedPartitionsValidationConfig,
    ValidationConfig,
    ValidationConfigs,
)
from ml_platform.data_sdk.materializers.exceptions import (
    DuplicateValidationOverrideError,
)
from ml_platform.data_sdk.transforms._base import DataTransform


class PartitionDependency(BaseConfig):
    """One pinned parent dataset partition required by a derived output."""

    partition_id: str = Field(
        min_length=1,
        description="Control-plane identifier of the pinned parent partition.",
    )
    dataset_name: DatasetName = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Name of the parent dataset owning this partition.",
    )
    dataset_version: ResourceVersion = Field(
        ge=1, description="Version of the parent dataset owning this partition."
    )
    timestamp: PartitionTimestamp = Field(
        description="UTC timestamp that identifies the parent logical partition."
    )
    snapshot_id: SnapshotId = Field(
        ge=0,
        description="Iceberg snapshot containing the pinned parent partition revision.",
    )


class MaterializerPartition(BaseConfig):
    """One requested output partition and its authenticated completion receipt."""

    partition_id: str = Field(
        min_length=1,
        description="Control-plane identifier for the requested output partition.",
    )
    timestamp: PartitionTimestamp = Field(
        description="UTC timestamp that identifies the requested logical output partition."
    )
    callback_token: CallbackToken = Field(
        min_length=1,
        description="Opaque token required to report successful partition completion.",
    )
    partition_dependencies: list[PartitionDependency] = Field(
        default_factory=list,
        description="Pinned parent dataset partition revisions required to produce this output partition.",
    )


class MaterializerConfig(BaseConfig):
    """Strict immutable API runtime contract shared by all materializers."""

    name: ResourceName = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stable lowercase resource name.",
    )
    version: ResourceVersion = Field(
        ge=1,
        description="Positive immutable version number for the resource definition.",
    )
    description: Description = Field(
        min_length=1,
        description="Human-readable description of the materialized resource.",
    )
    validations: list[DiscriminatedValidationConfig] = Field(
        default_factory=list,
        description="Declared runtime validation checks applied to materialized output.",
    )
    partitions: list[MaterializerPartition] = Field(
        min_length=1,
        description="Logical output partitions this job must materialize.",
    )
    warehouse_dir: str = Field(
        min_length=1,
        description="Iceberg warehouse URI or local directory for output tables.",
    )

    @classmethod
    def iceberg_catalog(cls) -> str:
        """Return the shared Iceberg catalog used by all materializer jobs."""
        return "ml_platform"

    @field_validator("validations")
    @classmethod
    def _unique_validation_types(
        cls, validations: list[ValidationConfig]
    ) -> list[ValidationConfig]:
        """Reject ambiguous YAML overrides before a materializer consumes this config."""
        validation_types = {validation.type for validation in validations}
        if len(validation_types) != len(validations):
            raise DuplicateValidationOverrideError(
                "Validation overrides must use unique types; "
                f"received: {', '.join(validation.type for validation in validations)}."
            )
        return validations

    @property
    def validation_overrides_by_type(self) -> dict[str, ValidationConfig]:
        """Return the already-validated YAML overrides by their discriminator."""
        return {validation.type: validation for validation in self.validations}

    @cached_property
    def partition_timestamps(self) -> tuple[datetime, ...]:
        """Return requested timestamps normalized to this resource's partition cadence."""
        return tuple(
            self.partition_type.normalize_timestamp(partition.timestamp)  # type: ignore[attr-defined]
            for partition in self.partitions
        )


@dataclass(frozen=True)
class Materializer[Config: MaterializerConfig](ABC):
    """Shared Spark lifecycle and partition callbacks for one materialization."""

    @property
    @abstractmethod
    def _partition_type(self) -> PartitionType:
        """Return the output cadence declared by this concrete materializer."""

    config: Config

    @property
    @abstractmethod
    def iceberg_table(self) -> IcebergTable:
        """Return the immutable table configuration for this materialization."""

    @classmethod
    @abstractmethod
    def config_model(cls) -> type[Config]:
        """Return the strict runtime configuration model for this worker."""

    @classmethod
    @abstractmethod
    def spark_application_name(cls) -> str:
        """Return the stable Spark application name for this worker type."""

    @classmethod
    def _config_from_json(cls, document: str) -> Config:
        """Deserialize and validate one materializer-specific job configuration."""
        return cls.config_model().model_validate_json(document)

    @classmethod
    def _create_spark_session(cls, warehouse_dir: str | Path) -> SparkSession:
        """Create the UTC Iceberg Spark session shared by all materializer workers."""
        warehouse = str(warehouse_dir)
        catalog = cls.config_model().iceberg_catalog()
        builder = SparkSession.builder.appName(cls.spark_application_name())
        if master := spark_master():
            builder = builder.master(master)
        return (
            builder.config("spark.driver.memory", "8g")
            .config("spark.executor.memory", "8g")
            .config("spark.sql.shuffle.partitions", "64")
            .config("spark.scheduler.mode", "FAIR")
            .config(
                "spark.sql.extensions",
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
            )
            .config(
                f"spark.sql.catalog.{catalog}",
                "org.apache.iceberg.spark.SparkCatalog",
            )
            .config(f"spark.sql.catalog.{catalog}.type", "hadoop")
            .config(f"spark.sql.catalog.{catalog}.warehouse", warehouse)
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )

    @staticmethod
    def _stop_spark_session(spark: SparkSession) -> None:
        """Stop a worker Spark session after its materialization completes."""
        spark.stop()

    @property
    def _base_validation_configs(self) -> tuple[ValidationConfig, ...]:
        """Declare partition validations shared by every materialization."""
        return (
            RequestedPartitionsValidationConfig(
                reason="Output rows must belong to the requested logical partitions.",
                timestamps=self.config.partition_timestamps,
                partition_type=self._partition_type,
            ),
        )

    @property
    @abstractmethod
    def validation_configs(self) -> tuple[ValidationConfig, ...]:
        """Return resource-specific declarative validation configurations."""

    def validate(self, data_frame: DataFrame) -> None:
        """Run this materialization's complete resolved validation sequence."""
        validation_configs = ValidationConfigs(
            default_validation_configs=self._base_validation_configs
            + self.validation_configs,
            override_validation_configs=tuple(self.config.validations),
        )
        MaterializationValidator(
            validation_configs=validation_configs
        ).validate(data_frame)

    def _complete_partitions(
        self, spark: SparkSession, snapshot_id: SnapshotId
    ) -> None:
        """Report committed partition revisions through the control-plane API."""
        table = self.iceberg_table
        timestamps = self.config.partition_timestamps
        partition_keys = table.partition_keys(timestamps)
        content_hashes = table.partition_content_hashes(
            spark, timestamps, self._row_digest()
        )
        row_counts = table.partition_row_counts(spark, timestamps)
        table_schema = table.schema_fields(spark)
        receipts = []
        for partition, partition_key in zip(
            self.config.partitions, partition_keys, strict=True
        ):
            receipts.append(
                {
                    "partition_id": partition.partition_id,
                    "resource_type": table.resource_type,
                    "row_count": row_counts.get(partition_key, RowCount(0)),
                    "iceberg_schema": table_schema,
                    "content_hash": content_hashes.get(
                        partition_key, table.empty_content_hash()
                    ),
                    "snapshot_id": snapshot_id,
                    "callback_token": partition.callback_token,
                    **(
                        {
                            "dataset_partition_ids": [
                                dependency.partition_id
                                for dependency in partition.partition_dependencies
                            ]
                        }
                        if partition.partition_dependencies
                        else {}
                    ),
                }
            )
        response = httpx2.request(
            method=PartitionRoutes.COMPLETE.method,
            url=(
                f"{materialization_api_url()}"
                f"{PartitionRoutes.COMPLETE.template}"
            ),
            content=json.dumps(receipts),
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()

    @abstractmethod
    def _row_digest(self) -> Column:
        """Return a digest expression for one persisted output row."""

    @abstractmethod
    def _transform(self, spark: SparkSession) -> DataTransform:
        """Instantiate the resource's declared transform for this Spark job."""

    def _requested_partitions(self, spark: SparkSession) -> DataFrame:
        """Return requested output boundaries for the shared output filter."""
        rows = [(timestamp,) for timestamp in self.config.partition_timestamps]
        return spark.createDataFrame(
            rows, "requested_partition timestamp"
        ).distinct()

    def _filter_partitions(
        self, data_frame: DataFrame, requested_partitions: DataFrame
    ) -> DataFrame:
        """Keep transform output within this job's requested logical partitions."""
        return (
            data_frame.withColumn(
                "requested_partition",
                spark_functions.date_trunc(
                    self._partition_type.partition_unit,
                    spark_functions.col("timestamp"),
                ),
            )
            .join(requested_partitions, "requested_partition", "left_semi")
            .drop("requested_partition")
        )

    @classmethod
    def main(cls) -> None:
        """Run one serialized materialization job from the standard CLI contract."""
        parser = argparse.ArgumentParser(
            description=f"Materialize one {cls.__name__} job."
        )
        parser.add_argument(
            "--config-json",
            required=True,
            help="Serialized materializer configuration.",
        )
        args = parser.parse_args()
        config = cls._config_from_json(args.config_json)
        materializer = cls(config)
        spark_session = materializer._create_spark_session(config.warehouse_dir)
        try:
            transform = materializer._transform(spark_session)
            requested_partitions = materializer._requested_partitions(
                spark_session
            )
            transformed_data_frame = transform()
            data_frame = materializer._filter_partitions(
                transformed_data_frame, requested_partitions
            )
            materializer.validate(data_frame)
            materializer.iceberg_table.create(spark_session)
            materializer.iceberg_table.write(data_frame)
            snapshot_id = materializer.iceberg_table.snapshot_id(spark_session)
            materializer._complete_partitions(spark_session, snapshot_id)
        finally:
            materializer._stop_spark_session(spark_session)
        sys.stdout.write("Materialization completed.\n")
