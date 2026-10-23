"""Spark worker for one API-submitted feature materialization job."""

from dataclasses import dataclass

from pydantic import Field

from ml_platform.data_sdk.materializers._materialization_validator import (
    KeysFromInputValidationConfig,
    ValidationConfig,
)
from ml_platform.data_sdk.materializers._transform_materializer import (
    TransformMaterializationConfig,
    TransformMaterializer,
    TransformTableDefinition,
)
from ml_platform.data_sdk.transforms._feature_transform import FeatureTransform
from ml_platform.data_sdk.transforms._types import FeatureDatasetDependency


class FeatureMaterializationConfig(TransformMaterializationConfig):
    """Complete JSON contract consumed by a feature Spark materializer."""

    datasets: list[FeatureDatasetDependency] = Field(
        min_length=1,
        description="Versioned datasets injected into feature transforms by dataset name.",
    )


@dataclass(frozen=True)
class FeatureMaterializer(
    TransformMaterializer[FeatureMaterializationConfig, FeatureTransform]
):
    """Materialize API-pinned feature partitions with immutable input revisions."""

    @property
    def _table_definition(self) -> TransformTableDefinition:
        """Declare the static Iceberg table layout for feature values."""
        return TransformTableDefinition(
            namespace="features",
            resource_type="feature",
            value_column="feature_value",
        )

    @classmethod
    def config_model(cls) -> type[FeatureMaterializationConfig]:
        """Return the strict runtime contract for feature jobs."""
        return FeatureMaterializationConfig

    @classmethod
    def spark_application_name(cls) -> str:
        """Return the stable Spark application name for feature jobs."""
        return "ml-platform-feature-materializer"

    @property
    def _transform_type(self) -> type[FeatureTransform]:
        """Return the transform contract used by feature materialization."""
        return FeatureTransform

    @property
    def validation_configs(self) -> tuple[ValidationConfig, ...]:
        """Add feature-specific input-key lineage validation."""
        return (
            *super().validation_configs,
            KeysFromInputValidationConfig(
                reason="Entity keys must occur in at least one input dataset.",
                keys=self._key_columns,
                dataset_table_identifiers=tuple(
                    self._dataset_table_identifier(dataset)
                    for dataset in self.config.datasets
                ),
            ),
        )


if __name__ == "__main__":
    FeatureMaterializer.main()
