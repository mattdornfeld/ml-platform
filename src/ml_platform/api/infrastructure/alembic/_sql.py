"""SQL-backed Alembic revision helpers."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from alembic import op

from ml_platform.api.infrastructure.alembic.exceptions import (
    DowngradeNotImplementedError,
)


@dataclass(frozen=True)
class SqlMigration:
    """Alembic metadata and execution callables backed by one SQL file."""

    revision: str
    down_revision: str | None
    branch_labels: None
    depends_on: None
    upgrade: Callable[[], None]
    downgrade: Callable[[], None]


def sql_migration(
    *, revision: str, down_revision: str | None, sql_file: str
) -> SqlMigration:
    """Construct one forward-only Alembic revision backed by `versions/sql`."""
    sql_path = Path(__file__).parent / "versions" / "sql" / sql_file

    def upgrade() -> None:
        """Apply the SQL migration body."""
        op.execute(sql_path.read_text(encoding="utf-8"))

    def downgrade() -> None:
        """Reject rollback because production migrations are forward-only."""
        raise DowngradeNotImplementedError(
            "Schema migrations are forward-only."
        )

    return SqlMigration(
        revision=revision,
        down_revision=down_revision,
        branch_labels=None,
        depends_on=None,
        upgrade=upgrade,
        downgrade=downgrade,
    )
