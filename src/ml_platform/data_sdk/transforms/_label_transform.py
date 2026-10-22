"""Base interface for Spark label transformations."""

from abc import ABC

from ml_platform.data_sdk.transforms._base import DataTransform
from ml_platform.data_sdk.transforms._types import DataType
from ml_platform.data_sdk.utils import record


@record
class LabelTransform(DataTransform, ABC):
    """A configured transformation from one injected dataset to label rows."""

    data_type: DataType

    @property
    def value_schema(self) -> dict[str, str]:
        """Declare the single typed value column emitted by every label transform."""
        return {"label_value": self.data_type.value}
