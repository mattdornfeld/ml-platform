"""Atomic Iceberg partition-completion callback route."""

from typing import Annotated

from fastapi import APIRouter, Body

from ml_platform.api._dependencies import DatabaseSession
from ml_platform.api.repositories import PartitionRepository
from ml_platform.api.routers._openapi import (
    COMPLETION_VALIDATION,
    error_responses,
)
from ml_platform.common.api.dtos import (
    PartitionCompletionRequest,
    PartitionResultResponse,
)
from ml_platform.common.api.routes import PartitionRoutes

router = APIRouter(tags=["partition completion"])


@router.post(
    PartitionRoutes.COMPLETE.template,
    response_model=list[PartitionResultResponse],
    responses=error_responses(COMPLETION_VALIDATION),
    tags=["partition completion"],
)
def complete_partitions(
    payload: Annotated[list[PartitionCompletionRequest], Body(min_length=1)],
    session: DatabaseSession,
) -> list[PartitionResultResponse]:
    """Record a batch of completed partition receipts atomically."""
    return PartitionRepository(session).complete(payload)
