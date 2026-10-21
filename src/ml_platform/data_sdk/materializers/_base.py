"""Private common contracts and Iceberg plumbing for API-dispatched materializers."""

from abc import ABC, ABCMeta, abstractmethod
import argparse
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

from pydantic import Field
from simple_classproperty import ClasspropertyMeta, classproperty

from ml_platform.common.types import BaseConfig
from ml_platform.data_sdk.materializers._constants import spark_master
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


class MaterializerInputPartition(BaseConfig):
    """One pinned parent dataset partition required by a derived output."""

    partition_id: str = Field(min_length=1, description="Control-plane identifier of the pinned parent partition.")
    timestamp: datetime = Field(description="UTC timestamp that identifies the parent logical partition.")
    snapshot_id: int = Field(ge=0, description="Iceberg snapshot containing the pinned parent partition revision.")


class MaterializerPartition(BaseConfig):
    """One requested output partition and its authenticated completion receipt."""

    partition_id: str = Field(min_length=1, description="Control-plane identifier for the requested output partition.")
    timestamp: datetime = Field(description="UTC timestamp that identifies the requested logical output partition.")
    callback_token: str = Field(min_length=1, description="Opaque token required to report successful partition completion.")
    dataset_partitions: list[MaterializerInputPartition] = Field(
        default_factory=list,
        description="Pinned parent dataset partition revisions required to produce this output partition.",
    )


class MaterializerValidations(BaseConfig):
    """Validation policy shared by materializer runtime contracts."""

    min_rows_per_partition: int = Field(
        default=1, ge=1, description="Minimum number of output rows required in each requested partition."
    )
    require_finite_feature_value: bool = Field(
        default=True, description="Whether feature values must be finite rather than NaN or infinite."
    )
    require_feature_keys_from_input: bool = Field(
        default=True, description="Whether output feature keys must be present in the declared input datasets."
    )


class MaterializerConfig(BaseConfig):
    """Strict immutable API runtime contract shared by all materializers."""

    iceberg_catalog: ClassVar[str] = "ml_platform"

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$", description="Stable lowercase resource name.")
    version: int = Field(ge=1, description="Positive immutable version number for the resource definition.")
    description: str = Field(min_length=1, description="Human-readable description of the materialized resource.")
    validations: MaterializerValidations = Field(
        default_factory=MaterializerValidations, description="Runtime validation policy for materialized output."
    )
    partitions: list[MaterializerPartition] = Field(
        min_length=1, description="Logical output partitions this job must materialize."
    )
    warehouse_dir: str = Field(min_length=1, description="Iceberg warehouse URI or local directory for output tables.")

    @classmethod
    def _iceberg_table_name_for(cls, name: str, version: int) -> str:
        """Return the physical table name for an immutable resource version."""
        return f"{name}-v{version}"

    @property
    def _iceberg_table_name(self) -> str:
        """Return the physical table name for this immutable resource version."""
        return self._iceberg_table_name_for(self.name, self.version)


Config = TypeVar("Config", bound=MaterializerConfig)


class MaterializerMeta(ClasspropertyMeta, ABCMeta):
    """Combine abstract-method and class-property support for materializers."""

    def __new__(mcls, name: str, bases: tuple[type, ...], namespace: dict[str, object], **kwargs: object) -> type:
        """Retain abstractness when the descriptor package resolves inherited properties."""
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        if any(isinstance(base, MaterializerMeta) for base in bases) and "iceberg_namespace" not in namespace:
            cls.__abstractmethods__ = frozenset((*cls.__abstractmethods__, "iceberg_namespace"))
        return cls


class Materializer(ABC, Generic[Config], metaclass=MaterializerMeta):
    """Shared Spark and Iceberg lifecycle for one API-delivered materialization."""

    config_model: type[Config]
    spark_application_name: ClassVar[str] = "ml-platform-materializer"
    iceberg_schema: ClassVar[str]
    iceberg_partition_spec: ClassVar[str]
    iceberg_table_properties: ClassVar[tuple[str, ...]] = (
        "'format-version' = '2'",
        "'write.format.default' = 'parquet'",
        "'write.parquet.compression-codec' = 'zstd'",
        "'write.distribution-mode' = 'hash'",
        "'write.spark.fanout.enabled' = 'false'",
    )

    @classproperty
    @abstractmethod
    def iceberg_namespace(cls) -> str:
        """Return the Iceberg namespace owned by this materializer type."""
        return ""

    def __init__(self, config: Config) -> None:
        self.config = config

    @classmethod
    def _config_from_json(cls, document: str) -> Config:
        """Deserialize and validate one materializer-specific job configuration."""
        return cls.config_model.model_validate_json(document)

    @classmethod
    def _run_config(cls, config: Config) -> int:
        """Materialize one validated API job configuration.

        Subclasses with additional construction dependencies may override this hook
        while preserving the shared command-line entry point.
        """
        return cls(config)()

    @classmethod
    def main(cls) -> None:
        """Run one serialized materialization job from the standard CLI contract."""
        parser = argparse.ArgumentParser(description=f"Materialize one {cls.__name__} job.")
        parser.add_argument("--config-json", required=True, help="Serialized materializer configuration.")
        args = parser.parse_args()
        config = cls._config_from_json(args.config_json)
        print(f"Materialized {cls._run_config(config):,} rows.")

    @classmethod
    def _iceberg_table_identifier(cls, name: str, version: int) -> str:
        """Resolve an immutable resource version to its Iceberg identifier."""
        namespace = cls.iceberg_namespace
        return f"{cls.config_model.iceberg_catalog}.{namespace}.{cls.config_model._iceberg_table_name_for(name, version)}"

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
        builder = SparkSession.builder.appName(cls.spark_application_name)
        if master := spark_master():
            builder = builder.master(master)
        catalog = cls.config_model.iceberg_catalog
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
    def _ensure_iceberg_table(
        cls, spark: SparkSession, *, name: str, version: int, schema: str | None = None, partition_spec: str | None = None, properties: tuple[str, ...] | None = None
    ) -> None:
        """Create the resource namespace and table using the subclass declaration."""
        namespace = cls.iceberg_namespace
        catalog = cls.config_model.iceberg_catalog
        table_schema = schema or cls.iceberg_schema
        table_partition_spec = partition_spec or cls.iceberg_partition_spec
        if not table_schema or not table_partition_spec:
            raise NotImplementedError(f"{cls.__name__} must declare an Iceberg schema and partition spec.")
        identifier = cls._iceberg_table_identifier(name, version)
        properties_sql = ", ".join(properties or cls.iceberg_table_properties)
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.{namespace}")
        spark.sql(
            f"CREATE TABLE IF NOT EXISTS {identifier} ({table_schema}) "
            f"USING iceberg PARTITIONED BY ({table_partition_spec}) "
            f"TBLPROPERTIES ({properties_sql})"
        )

    @staticmethod
    def _validate_required_columns(frame: DataFrame, required: set[str], *, context: str) -> None:
        """Reject transform output that omits required columns."""
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{context} omitted required columns: {', '.join(sorted(missing))}.")

    @staticmethod
    def _validate_non_empty(frame: DataFrame, *, context: str) -> None:
        """Reject an empty transform result before it can create a receipt."""
        if not frame.limit(1).count():
            raise ValueError(f"{context} returned no rows.")

    @staticmethod
    def _latest_snapshot_id(spark: SparkSession, identifier: str) -> int:
        """Return the latest Iceberg snapshot committed for a table."""
        snapshot = spark.sql(
            f"SELECT snapshot_id FROM {identifier}.snapshots ORDER BY committed_at DESC LIMIT 1"
        ).first()
        if snapshot is None:
            raise RuntimeError("Iceberg materialization completed without a snapshot.")
        return int(snapshot["snapshot_id"])

    @staticmethod
    def _stop_spark_session(spark: SparkSession) -> None:
        """Stop a worker Spark session after its materialization completes."""
        spark.stop()

    @abstractmethod
    def materialize(self, spark: SparkSession) -> int:
        """Materialize this resource using the Spark session owned by the parent lifecycle."""

    def __call__(self) -> int:
        """Execute the configured materialization and always close its Spark session."""
        spark = self._create_spark_session(self.config.warehouse_dir)
        try:
            return self.materialize(spark)
        finally:
            self._stop_spark_session(spark)
