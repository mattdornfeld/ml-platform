"""Typed FastAPI dependencies shared by control-plane route modules."""

from typing import Annotated

from fastapi import Depends, Path
from psycopg import Connection
from sqlalchemy.orm import Session

from ml_platform.api.infrastructure import get_connection, get_session
from ml_platform.common.types import JobId, ResourceName, ResourceVersion

DatabaseConnection = Annotated[Connection, Depends(get_connection)]
DatabaseSession = Annotated[Session, Depends(get_session)]
JobPathId = Annotated[JobId, Path()]
ResourcePathName = Annotated[ResourceName, Path(min_length=1)]
ResourcePathVersion = Annotated[ResourceVersion, Path(gt=0)]
