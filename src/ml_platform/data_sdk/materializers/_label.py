"""Spark worker for one API-submitted label materialization job."""

from dataclasses import dataclass

from pydantic import Field

from ml_platform.common.domain import LabelDatasetDependency
from ml_platform.data_sdk.materializers._transform_materializer import (
    TransformMaterializationConfig,
    TransformMaterializer,
    TransformTableDefinition,
)
from ml_platform.data_sdk.transforms._label_transform import LabelTransform


class LabelMaterializationConfig(TransformMaterializationConfig):
    """Complete JSON contract consumed by a label Spark materializer."""

    datasets: list[LabelDatasetDependency] = Field(
        min_length=1,
        description="Versioned datasets injected into label transforms by dataset name.",
    )


@dataclass(frozen=True)
class LabelMaterializer(
    TransformMaterializer[LabelMaterializationConfig, LabelTransform]
):
    """Materialize API-pinned label partitions with immutable input revisions."""

    @property
    def _table_definition(self) -> TransformTableDefinition:
        """Declare the static Iceberg table layout for label values."""
        return TransformTableDefinition(
            namespace="labels",
            resource_type="label",
            value_column="label_value",
        )

    @classmethod
    def config_model(cls) -> type[LabelMaterializationConfig]:
        """Return the strict runtime contract for label jobs."""
        return LabelMaterializationConfig

    @classmethod
    def spark_application_name(cls) -> str:
        """Return the stable Spark application name for label jobs."""
        return "ml-platform-label-materializer"

    @property
    def _transform_type(self) -> type[LabelTransform]:
        """Return the transform contract used by label materialization."""
        return LabelTransform


if __name__ == "__main__":
    LabelMaterializer.main()
