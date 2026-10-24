"""Public FastAPI control-plane entry points."""

from ml_platform.api._configuration import Dependencies
from ml_platform.api.app import app, main

__all__ = ["Dependencies", "app", "main"]
