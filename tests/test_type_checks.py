"""Run static type checks as part of the pytest suite."""

import subprocess
import sys
from pathlib import Path


def test_mypy() -> None:
    """Require every production module to satisfy the configured mypy policy."""
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "src"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
