"""K3s-backed deployment infrastructure for opt-in end-to-end tests."""

import os
import subprocess
import tempfile
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pytest
from testcontainers.community.k3s import K3SContainer

REPOSITORY_ROOT = Path(__file__).parents[2]
K3S_IMAGE = "ml-platform-k3s:latest"
K3S_DOCKERFILE = "deploy/docker/k3s/Dockerfile"


class E2eImageHealthTimeoutError(Exception):
    """Raised when the self-contained K3s image does not become healthy."""


@dataclass(frozen=True)
class DeployedPlatform:
    """One isolated K3s cluster running every ML Platform service."""

    cluster: K3SContainer
    kubeconfig: Path

    def kubectl(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        """Run kubectl against this cluster and require a successful result."""
        return subprocess.run(
            ["kubectl", "--kubeconfig", str(self.kubeconfig), *arguments],
            check=True,
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
        )


def _run(
    command: list[str], *, environment: dict[str, str] | None = None
) -> None:
    """Run one required E2E setup command with visible failure output."""
    subprocess.run(
        command,
        check=True,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
    )


def _build_k3s_image() -> None:
    """Build the self-bootstrapping K3s image that owns deployment setup."""
    _run(
        [
            "docker",
            "build",
            "--quiet",
            "--tag",
            K3S_IMAGE,
            "--file",
            K3S_DOCKERFILE,
            ".",
        ]
    )


def _wait_for_healthy_image(cluster: K3SContainer) -> None:
    """Wait until Docker reports the complete platform image as healthy."""
    deadline = time.monotonic() + 180
    container = cluster.get_wrapped_container()
    while time.monotonic() < deadline:
        container.reload()
        health = container.attrs["State"].get("Health", {}).get("Status")
        if health == "healthy":
            return
        if health == "unhealthy":
            logs = container.logs().decode()
            raise E2eImageHealthTimeoutError(
                f"The K3s image became unhealthy while bootstrapping:\n{logs}"
            )
        time.sleep(1)
    raise E2eImageHealthTimeoutError(
        "The self-contained K3s image did not become healthy within 180 seconds."
    )


@pytest.fixture(scope="session")
def deployed_platform() -> Generator[DeployedPlatform, None, None]:
    """Build images and deploy every required ML Platform service to K3s."""
    if os.environ.get("ML_PLATFORM_RUN_E2E") != "1":
        pytest.skip("Set ML_PLATFORM_RUN_E2E=1 to run K3s end-to-end tests.")
    previous_ryuk_setting = os.environ.get("TESTCONTAINERS_RYUK_DISABLED")
    os.environ["TESTCONTAINERS_RYUK_DISABLED"] = "true"
    try:
        _build_k3s_image()
        cluster = K3SContainer(K3S_IMAGE, enable_cgroup_mount=False)
        with cluster:
            _wait_for_healthy_image(cluster)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as kubeconfig_file:
                kubeconfig_file.write(cluster.config_yaml())
                kubeconfig = Path(kubeconfig_file.name)
            try:
                yield DeployedPlatform(cluster=cluster, kubeconfig=kubeconfig)
            finally:
                kubeconfig.unlink(missing_ok=True)
    finally:
        if previous_ryuk_setting is None:
            os.environ.pop("TESTCONTAINERS_RYUK_DISABLED", None)
        else:
            os.environ["TESTCONTAINERS_RYUK_DISABLED"] = previous_ryuk_setting
