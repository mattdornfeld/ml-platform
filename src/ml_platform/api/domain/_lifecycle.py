"""Lifecycle state models for registered control-plane resources."""

from enum import StrEnum


class ResourceLifecycle(StrEnum):
    """Lifecycle states for a registered immutable resource definition."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DELETED = "deleted"


class PartitionStatus(StrEnum):
    """Lifecycle states for a logical partition request."""

    PENDING = "pending"
    CREATED = "created"
    FAILED = "failed"
    CANCELLED = "cancelled"
