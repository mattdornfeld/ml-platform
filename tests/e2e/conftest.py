"""K3s-backed deployment infrastructure for opt-in end-to-end tests."""

import os
import subprocess
import tempfile
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pytest
from testcontainers.community.k3s import K3SContainer

REPOSITORY_ROOT = Path(__file__).parents[2]
K3S_IMAGE_DIRECTORY = "/var/lib/rancher/k3s/agent/images"
DEPLOYMENT_SCRIPT = REPOSITORY_ROOT / "deploy/kubernetes/install.sh"


class E2eImageImportError(Exception):
    """Raised when K3s cannot import a required locally built image."""


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


def _import_image(cluster: K3SContainer, image: str, archive: Path) -> None:
    """Build one host image archive and import it into K3s containerd."""
    _run(["docker", "save", "--output", str(archive), image])
    destination = f"{K3S_IMAGE_DIRECTORY}/{archive.name}"
    cluster.copy_into_container(archive, destination)
    result = cluster.exec(["ctr", "images", "import", destination])
    if result.exit_code != 0:
        raise E2eImageImportError(result.output.decode())


def _build_and_import_images(cluster: K3SContainer) -> None:
    """Build every local deployment image before applying Kubernetes resources."""
    _run(["docker", "build", "--tag", "ml-platform:latest", "."])
    _run(
        [
            "docker",
            "build",
            "--tag",
            "ml-platform-minio:latest",
            "--file",
            "deploy/docker/minio/Dockerfile",
            ".",
        ]
    )
    with tempfile.TemporaryDirectory() as directory:
        archives = Path(directory)
        _import_image(cluster, "ml-platform:latest", archives / "platform.tar")
        _import_image(
            cluster,
            "ml-platform-minio:latest",
            archives / "minio.tar",
        )


@pytest.fixture(scope="session")
def deployed_platform() -> Generator[DeployedPlatform, None, None]:
    """Build images and deploy every required ML Platform service to K3s."""
    if os.environ.get("ML_PLATFORM_RUN_E2E") != "1":
        pytest.skip("Set ML_PLATFORM_RUN_E2E=1 to run K3s end-to-end tests.")
    previous_ryuk_setting = os.environ.get("TESTCONTAINERS_RYUK_DISABLED")
    os.environ["TESTCONTAINERS_RYUK_DISABLED"] = "true"
    try:
        with K3SContainer(
            "rancher/k3s:v1.31.4-k3s1", enable_cgroup_mount=False
        ) as cluster:
            _build_and_import_images(cluster)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as kubeconfig_file:
                kubeconfig_file.write(cluster.config_yaml())
                kubeconfig = Path(kubeconfig_file.name)
            try:
                environment = os.environ | {"KUBECONFIG": str(kubeconfig)}
                _run([str(DEPLOYMENT_SCRIPT)], environment=environment)
                yield DeployedPlatform(cluster=cluster, kubeconfig=kubeconfig)
            finally:
                kubeconfig.unlink(missing_ok=True)
    finally:
        if previous_ryuk_setting is None:
            os.environ.pop("TESTCONTAINERS_RYUK_DISABLED", None)
        else:
            os.environ["TESTCONTAINERS_RYUK_DISABLED"] = previous_ryuk_setting
