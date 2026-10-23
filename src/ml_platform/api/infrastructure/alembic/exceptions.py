"""Errors raised by the control-plane Alembic migration helpers."""


class DowngradeNotImplementedError(RuntimeError):
    """A caller attempted to downgrade a forward-only production migration."""
