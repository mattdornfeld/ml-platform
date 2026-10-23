"""Alembic integration for control-plane database schema upgrades."""

from pathlib import Path

from alembic import command
from alembic.config import Config

_REPOSITORY_ROOT = Path(__file__).parents[4]
_PYPROJECT = _REPOSITORY_ROOT / "pyproject.toml"


def upgrade_database() -> None:
    """Upgrade the configured PostgreSQL database to the Alembic head revision."""
    command.upgrade(Config(toml_file=str(_PYPROJECT)), "head")
