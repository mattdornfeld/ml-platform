"""Base interface for transforms that construct partition-targeted dataset DataFrames."""

from abc import ABC

from ml_platform.data_sdk.transforms._base import DataTransform
from ml_platform.data_sdk.utils import record


@record
class DatasetTransform(DataTransform, ABC):
    """A transform that constructs rows for exactly the injected logical partitions."""
