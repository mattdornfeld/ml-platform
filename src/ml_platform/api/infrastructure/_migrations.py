"""Alembic integration for control-plane database schema upgrades."""

from pathlib import Path

from alembic import command
from alembic.config import Config

from ml_platform.api._configuration import Dependencies

_REPOSITORY_ROOT = Path(__file__).parents[4]
_PYPROJECT = _REPOSITORY_ROOT / "pyproject.toml"


def upgrade_database(dependencies: Dependencies) -> None:
    """Upgrade this dependency context's database to the Alembic head revision."""
    config = Config(toml_file=str(_PYPROJECT))
    config.attributes["dependencies"] = dependencies
    command.upgrade(config, "head")
