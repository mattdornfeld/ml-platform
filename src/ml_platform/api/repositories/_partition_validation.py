"""Shared validation for resource-specific partition timestamps."""

from ml_platform.api.domain import ResourceType
from ml_platform.api.repositories.exceptions import (
    InvalidMaterializationPartitionTimestampError,
)
from ml_platform.common.types import (
    PartitionTimestamp,
    PartitionType,
    ResourceName,
    ResourceVersion,
)


def validate_partition_timestamps(
    *,
    name: ResourceName,
    version: ResourceVersion,
    resource_type: ResourceType,
    partition_type: PartitionType,
    timestamps: list[PartitionTimestamp] | tuple[PartitionTimestamp, ...],
) -> None:
    """Require requested timestamps to match a resource's stored cadence."""
    for timestamp in timestamps:
        if not partition_type.is_partition_timestamp(timestamp):
            raise InvalidMaterializationPartitionTimestampError(
                name=name,
                version=version,
                resource_type=resource_type,
                partition_type=partition_type,
                timestamp=timestamp,
            )
