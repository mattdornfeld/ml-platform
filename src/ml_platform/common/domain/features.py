"""Feature definition models shared by API clients and Spark workers."""

from pydantic import Field

from ml_platform.common.domain.dependencies import FeatureDatasetDependency
from ml_platform.common.domain.resources import TransformConfig
from ml_platform.common.types import FeatureId, FeatureName


class FeatureConfig(TransformConfig[FeatureName]):
    """A complete immutable feature resource definition."""

    name: FeatureName = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stable lowercase feature name.",
    )
    datasets: list[FeatureDatasetDependency] = Field(
        min_length=1,
        description="Versioned datasets injected into feature transforms by dataset name.",
    )

    @property
    def id(self) -> FeatureId:
        """Return the canonical immutable feature identifier."""
        return FeatureId(f"{self.name}_v{self.version}")
