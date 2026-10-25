"""Pytest configuration shared by every test tier."""

from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).parent


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
