"""Immutable Iceberg table definition and persistence operations."""

import json
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import assert_never

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as spark_functions
from pyspark.sql.column import Column

from ml_platform.data_sdk.materializers.exceptions import (
    MissingIcebergSnapshotError,
)
from ml_platform.data_sdk.transforms._types import PartitionType


@dataclass(frozen=True)
class IcebergTable:
    """All immutable configuration and persistence operations for one Iceberg table."""

    catalog: str
    namespace: str
    name: str
    version: int
    resource_type: str
    schema: dict[str, str]
    partition_type: PartitionType
    description: str
    dependencies: tuple[dict[str, object], ...]
    validations: tuple[dict[str, object], ...]
    additional_properties: tuple[tuple[str, str], ...]

    @property
    def id(self) -> str:
        """Return the fully-qualified immutable Iceberg table identifier."""
        return self.id_for(
            self.catalog, self.namespace, self.name, self.version
        )

    @property
    def partition_spec(self) -> str:
        """Return the table's declared timestamp partition transform."""
        return self.partition_type.iceberg_partition_spec

    @property
    def _properties(self) -> tuple[tuple[str, str], ...]:
        """Construct Iceberg properties from this table's typed resource definition."""
        return (
            ("format-version", "2"),
            ("write.format.default", "parquet"),
            ("write.parquet.compression-codec", "zstd"),
            ("write.distribution-mode", "hash"),
            ("write.spark.fanout.enabled", "false"),
            ("resource.type", self.resource_type),
            ("resource.name", self.name),
            ("resource.version", str(self.version)),
            ("resource.frequency", self.partition_type.value),
            ("resource.description", self.description),
            (
                "resource.dependencies",
                json.dumps(self.dependencies, sort_keys=True),
            ),
            (
                "resource.validations",
                json.dumps(self.validations, sort_keys=True),
            ),
            *self.additional_properties,
        )

    @property
    def _schema_ddl(self) -> str:
        """Translate this table's typed schema into Spark SQL column declarations."""
        return ", ".join(
            f"{name} {data_type.upper()}"
            for name, data_type in self.schema.items()
        )

    @classmethod
    def id_for(
        cls, catalog: str, namespace: str, name: str, version: int
    ) -> str:
        """Build an immutable table identifier from its resource identity."""
        return f"{catalog}.{namespace}.`{name}-v{version}`"

    def storage_path(self, warehouse_dir: str | Path) -> Path:
        """Return this table's immutable warehouse directory."""
        return (
            Path(warehouse_dir).resolve()
            / self.namespace
            / f"{self.name}-v{self.version}"
        )

    @staticmethod
    def _sql_literal(value: str) -> str:
        """Quote a Spark SQL literal used in table-property statements."""
        return "'" + value.replace("'", "''") + "'"

    def _partition_transform(self) -> Column:
        """Return the PySpark transform matching this table's partition cadence."""
        match self.partition_type:
            case PartitionType.DAILY:
                return spark_functions.days("timestamp")
            case PartitionType.HOURLY:
                return spark_functions.hours("timestamp")
            case _ as unsupported_partition_type:
                assert_never(unsupported_partition_type)

    def partition_keys(
        self, timestamps: tuple[datetime, ...]
    ) -> tuple[str, ...]:
        """Normalize requested timestamps to stable persisted partition keys."""
        return tuple(
            self.partition_type.normalize_timestamp(timestamp).strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
            for timestamp in timestamps
        )

    def _reconcile_properties(self, spark: SparkSession) -> None:
        """Set immutable definition metadata when an old table lacks it."""
        existing_properties = {
            row.key: row.value
            for row in spark.sql(f"SHOW TBLPROPERTIES {self.id}").collect()
        }
        if existing_properties.get("resource.type") == self.resource_type:
            return
        obsolete_keys = {
            f"{self.resource_type}.{field}"
            for field in (
                "name",
                "version",
                "frequency",
                "description",
                "validations",
                "upstream_spec",
            )
        }.intersection(existing_properties)
        if obsolete_keys:
            keys = ", ".join(
                self._sql_literal(key) for key in sorted(obsolete_keys)
            )
            spark.sql(f"ALTER TABLE {self.id} UNSET TBLPROPERTIES ({keys})")
        assignments = ", ".join(
            f"{self._sql_literal(key)} = {self._sql_literal(value)}"
            for key, value in self._properties
        )
        spark.sql(f"ALTER TABLE {self.id} SET TBLPROPERTIES ({assignments})")

    def create(self, spark: SparkSession) -> None:
        """Create the table or reconcile its immutable resource metadata."""
        if spark.catalog.tableExists(self.id):
            self._reconcile_properties(spark)
            return
        spark.sql(
            f"CREATE NAMESPACE IF NOT EXISTS {self.catalog}.{self.namespace}"
        )
        writer = (
            spark.createDataFrame([], self._schema_ddl)
            .writeTo(self.id)
            .using("iceberg")
        )
        for key, value in self._properties:
            writer = writer.tableProperty(key, value)
        writer.partitionedBy(self._partition_transform()).create()

    def write(self, data_frame: DataFrame) -> None:
        """Atomically replace every output partition emitted by this materialization."""
        data_frame.writeTo(self.id).overwritePartitions()

    def snapshot_id(self, spark: SparkSession) -> int:
        """Return the latest committed snapshot for this table."""
        snapshot = (
            spark.table(f"{self.id}.snapshots")
            .orderBy(spark_functions.col("committed_at").desc())
            .select("snapshot_id")
            .first()
        )
        if snapshot is None:
            raise MissingIcebergSnapshotError(
                f"Iceberg table {self.id} has no committed snapshot after materialization."
            )
        return int(snapshot["snapshot_id"])

    def partition_content_hashes(
        self,
        spark: SparkSession,
        timestamps: tuple[datetime, ...],
        row_digest: Column,
    ) -> dict[str, str]:
        """Hash requested committed partitions using the materializer's row digest."""
        target_keys = self.partition_keys(timestamps)
        partition_key = spark_functions.date_format(
            spark_functions.date_trunc(
                self.partition_type.partition_unit,
                spark_functions.col("timestamp"),
            ),
            "yyyy-MM-dd'T'HH:mm:ss",
        )
        rows = (
            spark.table(self.id)
            .withColumn("partition_key", partition_key)
            .where(
                spark_functions.col("partition_key").isin(
                    sorted(set(target_keys))
                )
            )
            .groupBy("partition_key")
            .agg(
                spark_functions.sha2(
                    spark_functions.concat_ws(
                        "",
                        spark_functions.sort_array(
                            spark_functions.collect_list(row_digest)
                        ),
                    ),
                    256,
                ).alias("content_hash")
            )
            .collect()
        )
        return {row["partition_key"]: row["content_hash"] for row in rows}

    def partition_row_counts(
        self, spark: SparkSession, timestamps: tuple[datetime, ...]
    ) -> dict[str, int]:
        """Count rows in requested committed partitions."""
        target_keys = self.partition_keys(timestamps)
        partition_key = spark_functions.date_format(
            spark_functions.date_trunc(
                self.partition_type.partition_unit,
                spark_functions.col("timestamp"),
            ),
            "yyyy-MM-dd'T'HH:mm:ss",
        )
        rows = (
            spark.table(self.id)
            .withColumn("partition_key", partition_key)
            .where(
                spark_functions.col("partition_key").isin(
                    sorted(set(target_keys))
                )
            )
            .groupBy("partition_key")
            .count()
            .collect()
        )
        return {row["partition_key"]: int(row["count"]) for row in rows}

    def schema_fields(self, spark: SparkSession) -> dict[str, str]:
        """Return the persisted Spark schema for completion receipts."""
        return {
            field.name: field.dataType.simpleString()
            for field in spark.table(self.id).schema.fields
        }

    @staticmethod
    def empty_content_hash() -> str:
        """Return the canonical digest for an empty partition."""
        return sha256(b"").hexdigest()
