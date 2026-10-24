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
K3S_IMAGE_DIRECTORY = "/var/lib/rancher/k3s/agent/images"
K3S_IMAGE = "ml-platform-k3s:latest"
K3S_DOCKERFILE = "deploy/docker/k3s/Dockerfile"
K3S_E2E_COMMAND = (
    "server --disable traefik --tls-san=localhost "
    "--kubelet-arg=eviction-hard=imagefs.available<100Mi,nodefs.available<100Mi"
)


class E2eImageImportError(Exception):
    """Raised when K3s cannot import a required locally built image."""


class E2eDeploymentTimeoutError(Exception):
    """Raised when the K3s image does not create a required deployment."""


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


def _build_and_import_workload_images(cluster: K3SContainer) -> None:
    """Build and import every local image required by deployed workloads."""
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


def _wait_for_platform(platform: DeployedPlatform) -> None:
    """Wait until the image-managed cluster makes every workload ready."""
    for namespace, deployment in (
        ("ml-platform", "postgres"),
        ("ml-platform", "minio"),
        ("ml-platform", "ml-control-plane"),
    ):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                platform.kubectl(
                    "--namespace",
                    namespace,
                    "get",
                    f"deployment/{deployment}",
                )
            except subprocess.CalledProcessError:
                time.sleep(1)
            else:
                break
        else:
            raise E2eDeploymentTimeoutError(
                f"The K3s image did not create deployment {namespace}/{deployment}."
            )
        platform.kubectl(
            "--namespace",
            namespace,
            "rollout",
            "status",
            f"deployment/{deployment}",
            "--timeout=180s",
        )
    for job in ("create-warehouse-bucket", "configure-public-warehouse"):
        platform.kubectl(
            "--namespace",
            "ml-platform",
            "wait",
            "--for=condition=complete",
            f"job/{job}",
            "--timeout=180s",
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
        cluster.with_command(K3S_E2E_COMMAND)
        with cluster:
            _build_and_import_workload_images(cluster)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as kubeconfig_file:
                kubeconfig_file.write(cluster.config_yaml())
                kubeconfig = Path(kubeconfig_file.name)
            try:
                platform = DeployedPlatform(
                    cluster=cluster, kubeconfig=kubeconfig
                )
                _wait_for_platform(platform)
                yield platform
            finally:
                kubeconfig.unlink(missing_ok=True)
    finally:
        if previous_ryuk_setting is None:
            os.environ.pop("TESTCONTAINERS_RYUK_DISABLED", None)
        else:
            os.environ["TESTCONTAINERS_RYUK_DISABLED"] = previous_ryuk_setting
