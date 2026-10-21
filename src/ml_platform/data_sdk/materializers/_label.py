"""Iceberg label materializer dispatched by the control plane."""

from dataclasses import dataclass
from importlib import import_module
from inspect import Parameter, signature
from pathlib import Path
from typing import ClassVar, cast, get_type_hints

from pydantic import Field
from simple_classproperty import classproperty
from pyspark.sql import SparkSession

from ml_platform.data_sdk.materializers._base import Materializer, MaterializerConfig
from ml_platform.data_sdk.transforms._label_transform import LabelTransform
from ml_platform.data_sdk.transforms._types import Dataset, DatasetDependency


class LabelMaterializationConfig(MaterializerConfig):
    """Complete JSON contract consumed by a label Spark materializer."""

    datasets: list[DatasetDependency] = Field(min_length=1, description="Versioned dataset dependencies consumed by the label transform.")
    transform: str = Field(
        pattern=r"^[a-z_][a-z0-9_.]*\.[A-Za-z_][A-Za-z0-9_]*$",
        description="Absolute import path of the LabelTransform implementation.",
    )
    parameters: dict[str, int | float | str | bool] = Field(
        default_factory=dict, description="Typed keyword arguments passed to the label transform constructor."
    )


@dataclass(frozen=True)
class LabelResource:
    """Identity for one materialized label resource."""

    name: str
    version: int


class LabelMaterializer(Materializer[LabelMaterializationConfig]):
    """Construct, validate, persist, and report one label materialization."""

    config_model = LabelMaterializationConfig
    spark_application_name: ClassVar[str] = "ml-platform-label-materializer"
    iceberg_schema: ClassVar[str] = (
        "timestamp TIMESTAMP, symbol STRING, vertical_barrier_timestamp TIMESTAMP, close DOUBLE, "
        "volatility DOUBLE, upper_barrier DOUBLE, lower_barrier DOUBLE, horizon_bars INT, "
        "barrier_outcome STRING, long_short_label INT"
    )
    iceberg_partition_spec: ClassVar[str] = "days(timestamp)"

    @classproperty
    def iceberg_namespace(cls) -> str:
        """Store label resources in their dedicated Iceberg namespace."""
        return "labels"

    def __init__(self, config: LabelMaterializationConfig, datasets: dict[str, Dataset]) -> None:
        super().__init__(config)
        self.datasets = datasets

    @classmethod
    def _label_id(cls, resource: LabelResource) -> str:
        """Return the physical versioned label table name."""
        return cls.config_model._iceberg_table_name_for(resource.name, resource.version)

    @classmethod
    def _table_identifier_for(cls, resource: LabelResource) -> str:
        """Return the Iceberg identifier for one label resource."""
        return cls._iceberg_table_identifier(resource.name, resource.version)

    @classmethod
    def _storage_path(cls, warehouse_dir: str | Path, resource: LabelResource) -> Path:
        """Return the resource's versioned warehouse directory."""
        return Path(warehouse_dir).resolve() / cls.iceberg_namespace / cls._label_id(resource)

    @staticmethod
    def _load_transform(import_path: str) -> type[LabelTransform]:
        """Load a LabelTransform class referenced by the API-delivered definition."""
        module_name, class_name = import_path.rsplit(".", maxsplit=1)
        try:
            transform = getattr(import_module(module_name), class_name)
        except (AttributeError, ImportError) as error:
            raise ValueError(f"Cannot load label transform: {import_path}") from error
        if not isinstance(transform, type) or not issubclass(transform, LabelTransform):
            raise ValueError(f"Label transform must inherit LabelTransform: {import_path}")
        return cast(type[LabelTransform], transform)

    @staticmethod
    def _instantiate_transform(
        transform_class: type[LabelTransform], parameters: dict[str, object], datasets: dict[str, Dataset]
    ) -> LabelTransform:
        """Require the transform constructor to match declared datasets and parameters."""
        constructor_parameters = {
            name: parameter
            for name, parameter in signature(transform_class).parameters.items()
            if parameter.kind in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY}
        }
        annotations = get_type_hints(transform_class.__init__)
        dataset_parameters = {name for name in constructor_parameters if annotations.get(name) is Dataset}
        if dataset_parameters != set(datasets):
            raise ValueError(
                "Label transform dataset discrepancy "
                f"(constructor: {', '.join(sorted(dataset_parameters)) or 'none'}; "
                f"declared: {', '.join(sorted(datasets)) or 'none'})."
            )
        parameter_definitions = {
            name: parameter for name, parameter in constructor_parameters.items() if name not in dataset_parameters
        }
        required = {name for name, parameter in parameter_definitions.items() if parameter.default is Parameter.empty}
        missing, unexpected = required.difference(parameters), set(parameters).difference(parameter_definitions)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing: {', '.join(sorted(missing))}")
            if unexpected:
                details.append(f"unexpected: {', '.join(sorted(unexpected))}")
            raise ValueError(f"Label transform parameter discrepancy ({'; '.join(details)}).")
        for name, value in parameters.items():
            expected_type = annotations.get(name)
            if isinstance(expected_type, type) and type(value) is not expected_type:
                raise ValueError(
                    "Label transform parameter discrepancy "
                    f"(type for {name}: expected {expected_type.__name__}, got {type(value).__name__})."
                )
        try:
            return transform_class(**datasets, **parameters)
        except TypeError as error:
            raise ValueError(f"Label transform parameter discrepancy for {transform_class.__name__}: {error}") from error

    @classmethod
    def _materialize_transform(
        cls, resource: LabelResource, transform: LabelTransform, *, spark: SparkSession
    ) -> int:
        """Persist one transform result through the label table lifecycle."""
        labels = transform().localCheckpoint(eager=True)
        cls._validate_non_empty(labels, context="Label transform")
        cls._ensure_iceberg_table(
            spark,
            name=resource.name,
            version=resource.version,
            properties=(
                *cls.iceberg_table_properties,
                f"'label.name' = '{resource.name}'",
                f"'label.version' = '{resource.version}'",
            ),
        )
        labels.createOrReplaceTempView("incoming_labels")
        spark.sql(
            f"""
            MERGE INTO {cls._table_identifier_for(resource)} AS target
            USING incoming_labels AS source
            ON target.symbol = source.symbol
               AND target.timestamp = source.timestamp
               AND target.horizon_bars = source.horizon_bars
            WHEN MATCHED THEN UPDATE SET
                target.vertical_barrier_timestamp = source.vertical_barrier_timestamp,
                target.close = source.close,
                target.volatility = source.volatility,
                target.upper_barrier = source.upper_barrier,
                target.lower_barrier = source.lower_barrier,
                target.barrier_outcome = source.barrier_outcome,
                target.long_short_label = source.long_short_label
            WHEN NOT MATCHED THEN INSERT (
                timestamp, symbol, vertical_barrier_timestamp, close, volatility,
                upper_barrier, lower_barrier, horizon_bars, barrier_outcome, long_short_label
            ) VALUES (
                source.timestamp, source.symbol, source.vertical_barrier_timestamp, source.close,
                source.volatility, source.upper_barrier, source.lower_barrier,
                source.horizon_bars, source.barrier_outcome, source.long_short_label
            )
            """
        )
        return labels.count()

    def materialize(self, spark: SparkSession) -> int:
        """Instantiate the configured transform and persist its labels."""
        transform = self._instantiate_transform(
            self._load_transform(self.config.transform), self.config.parameters, self.datasets
        )
        return self._materialize_transform(
            LabelResource(name=self.config.name, version=self.config.version), transform, spark=spark
        )
