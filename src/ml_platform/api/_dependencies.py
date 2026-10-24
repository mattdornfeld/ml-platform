"""Typed FastAPI dependencies shared by control-plane route modules."""

from collections.abc import Generator
from typing import Annotated

from fastapi import Depends, Path
from psycopg import Connection
from sqlalchemy.orm import Session

from ml_platform.api._configuration import Dependencies
from ml_platform.api._exceptions import DependenciesNotInjectedError
from ml_platform.api.infrastructure import get_connection, get_session
from ml_platform.common.types import JobId, ResourceName, ResourceVersion


def get_dependencies() -> Dependencies:
    """Require application composition to inject validated dependencies."""
    raise DependenciesNotInjectedError


def get_database_session(
    dependencies: Annotated[Dependencies, Depends(get_dependencies)],
) -> Generator[Session, None, None]:
    """Yield the database session bound to this request's dependencies."""
    yield from get_session(dependencies)


def get_database_connection(
    dependencies: Annotated[Dependencies, Depends(get_dependencies)],
) -> Generator[Connection, None, None]:
    """Yield the database connection bound to this request's dependencies."""
    yield from get_connection(dependencies)


ApiDependencies = Annotated[Dependencies, Depends(get_dependencies)]
DatabaseConnection = Annotated[Connection, Depends(get_database_connection)]
DatabaseSession = Annotated[Session, Depends(get_database_session)]
JobPathId = Annotated[
    JobId, Path(description="UUID identifying the materialization job.")
]
ResourcePathName = Annotated[
    ResourceName,
    Path(
        min_length=1,
        description="Non-empty immutable name of the addressed resource.",
    ),
]
ResourcePathVersion = Annotated[
    ResourceVersion,
    Path(
        gt=0,
        description="Positive immutable version of the addressed resource.",
    ),
]

__all__ = [
    "ApiDependencies",
    "DatabaseConnection",
    "DatabaseSession",
    "Dependencies",
    "JobPathId",
    "ResourcePathName",
    "ResourcePathVersion",
    "get_database_connection",
    "get_database_session",
    "get_dependencies",
]
