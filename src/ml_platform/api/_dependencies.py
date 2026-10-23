"""Typed FastAPI dependencies shared by control-plane route modules.

The aliases below define every non-body value injected into a route. FastAPI
resolves database dependencies for the request and validates path parameters
before entering the route handler.
"""

from typing import Annotated

from fastapi import Depends, Path
from psycopg import Connection
from sqlalchemy.orm import Session

from ml_platform.api.infrastructure import get_connection, get_session
from ml_platform.common.types import JobId, ResourceName, ResourceVersion

# A Psycopg connection scoped to the current HTTP request.
DatabaseConnection = Annotated[Connection, Depends(get_connection)]

# A SQLAlchemy session scoped to the current HTTP request.
DatabaseSession = Annotated[Session, Depends(get_session)]

# A UUID that identifies the materialization job addressed by a route.
JobPathId = Annotated[
    JobId,
    Path(description="UUID identifying the materialization job."),
]

# A non-empty immutable dataset, feature, or label name in a route path.
ResourcePathName = Annotated[
    ResourceName,
    Path(
        min_length=1,
        description="Non-empty immutable name of the addressed resource.",
    ),
]

# A positive immutable dataset, feature, or label version in a route path.
ResourcePathVersion = Annotated[
    ResourceVersion,
    Path(
        gt=0,
        description="Positive immutable version of the addressed resource.",
    ),
]
