"""Private environment-backed settings for Spark materializer workers."""

import os

DEFAULT_MATERIALIZATION_API_URL = "http://ml-control-plane:8000"
"""Default in-cluster control-plane endpoint for partition callbacks."""


def materialization_api_url() -> str:
    """Return the control-plane callback endpoint configured for this worker."""
    return os.environ.get(
        "MATERIALIZATION_API_URL", DEFAULT_MATERIALIZATION_API_URL
    )


def spark_master() -> str | None:
    """Return an optional Spark master override for local materialization runs."""
    return os.environ.get("SPARK_MASTER")
