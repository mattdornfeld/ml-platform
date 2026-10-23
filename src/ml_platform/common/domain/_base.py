"""Pydantic configuration foundation for shared domain definitions."""

from pydantic import ConfigDict

from ml_platform.common.types import BaseConfig


class DomainConfig(BaseConfig):
    """Frozen domain model that parses standard JSON and YAML primitives."""

    model_config = ConfigDict(strict=False)
