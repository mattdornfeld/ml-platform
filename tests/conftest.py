"""Pytest configuration shared by every test tier."""

import os
from pathlib import Path

import pytest

from ml_platform.api import Dependencies
from ml_platform.common.types import Environment

os.environ.setdefault("DATABASE_URL", "postgresql://unused")

TESTS_ROOT = Path(__file__).parent


@pytest.fixture
def runtime_dependencies() -> Dependencies:
    """Build validated dependencies for an isolated unit-test call path."""
    return Dependencies(
        environment=Environment.TEST,
        database_url="postgresql://unused",
        kubernetes_namespace="ml-platform",
        materialization_image="ml-platform:latest",
        warehouse_dir="s3a://warehouse",
        s3_endpoint="http://minio:9000",
        minio_root_user="minio",
        minio_root_password="miniosecret",
        headlamp_url="http://127.0.0.1:8002",
        minio_console_url="http://127.0.0.1:9003",
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark collected tests according to their infrastructure tier directory."""
    for item in items:
        tier = item.path.relative_to(TESTS_ROOT).parts[0]
        if tier == "unit":
            item.add_marker(pytest.mark.unit)
        elif tier == "integration":
            item.add_marker(pytest.mark.integration)
        elif tier == "e2e":
            item.add_marker(pytest.mark.e2e)
