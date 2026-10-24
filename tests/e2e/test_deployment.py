"""End-to-end verification of the complete local ML Platform deployment."""

import pytest
from tests.e2e.conftest import DeployedPlatform

pytestmark = pytest.mark.e2e


def test_required_services_are_ready(
    deployed_platform: DeployedPlatform,
) -> None:
    """Every declared platform service reaches Kubernetes-ready state."""
    deployments = deployed_platform.kubectl(
        "get",
        "deployments",
        "--all-namespaces",
        "--output",
        'jsonpath={range .items[*]}{.metadata.namespace}/{.metadata.name}{"\\n"}{end}',
    ).stdout.splitlines()

    assert "ml-platform/ml-control-plane" in deployments
    assert "ml-platform/minio" in deployments
    assert "ml-platform/postgres" in deployments
    assert "spark-operator/spark-operator-controller" in deployments
    assert "kueue-system/kueue-controller-manager" in deployments
    assert "headlamp/headlamp" in deployments
