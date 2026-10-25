"""K3s-backed deployment infrastructure for opt-in end-to-end tests."""

import os
import re
import select
import subprocess
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
from testcontainers.community.k3s import K3SContainer

REPOSITORY_ROOT = Path(__file__).parents[2]
K3S_IMAGE = "ml-platform-k3s:latest"
K3S_DOCKERFILE = "deploy/docker/k3s/Dockerfile"


class E2eImageHealthTimeoutError(Exception):
    """Raised when the self-contained K3s image does not become healthy."""


class E2ePortForwardTimeoutError(Exception):
    """Raised when a Kubernetes service does not expose a local forwarded port."""


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

    def spark_application_status(self, application_name: str) -> str:
        """Return a failed Spark application's resource state and driver logs."""
        application_status = self.kubectl(
            "get",
            "sparkapplication",
            application_name,
            "--namespace",
            "ml-platform",
            "--output",
            "yaml",
        ).stdout
        driver_logs = self.kubectl(
            "logs",
            f"{application_name}-driver",
            "--namespace",
            "ml-platform",
        ).stdout
        return f"{application_status}\nDriver logs:\n{driver_logs}"

    @contextmanager
    def api_url(self) -> Generator[str, None, None]:
        """Expose the in-cluster control-plane API through one local URL."""
        process = subprocess.Popen(
            [
                "kubectl",
                "--kubeconfig",
                str(self.kubeconfig),
                "port-forward",
                "--address",
                "127.0.0.1",
                "service/ml-control-plane",
                "0:8000",
                "--namespace",
                "ml-platform",
            ],
            cwd=REPOSITORY_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            assert process.stdout is not None
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                ready, _, _ = select.select([process.stdout], [], [], 0.25)
                if not ready:
                    continue
                line = process.stdout.readline()
                match = re.search(r"127\.0\.0\.1:(\d+)", line)
                if match is not None:
                    yield f"http://127.0.0.1:{match.group(1)}"
                    return
            raise E2ePortForwardTimeoutError(
                "Timed out forwarding the ml-control-plane service."
            )
        finally:
            process.terminate()
            process.wait(timeout=10)


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
