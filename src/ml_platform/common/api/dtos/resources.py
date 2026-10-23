"""Shared resource-kind DTO definitions."""

from enum import StrEnum


class ResourceType(StrEnum):
    """Materializable resource kinds exposed by the control plane."""

    DATASET = "dataset"
    FEATURE = "feature"
    LABEL = "label"
