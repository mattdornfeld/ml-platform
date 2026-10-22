"""Base interface and parameter validation for Spark feature transforms."""
from abc import ABC

from ml_platform.data_sdk.transforms._base import DataTransform
from ml_platform.data_sdk.transforms._types import DataType
from ml_platform.data_sdk.utils import record


@record
class FeatureTransform(DataTransform, ABC):
    """A configured transformation from injected datasets to a feature DataFrame."""
    data_type: DataType

    @property
    def value_schema(self) -> dict[str, str]:
        """Declare the single typed value column emitted by every feature transform."""
        return {"feature_value": self.data_type.value}
