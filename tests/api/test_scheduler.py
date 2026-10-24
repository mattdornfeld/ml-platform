"""Unit coverage for scheduler Kubernetes reconciliation."""

import json
from unittest.mock import MagicMock, patch
from uuid import UUID

from ml_platform.api import Dependencies
from ml_platform.api.services import ADMISSION_GATE, JobScheduler


@patch("ml_platform.api.services._scheduler.client.CustomObjectsApi")
def test_dataset_application_is_created_admission_gated(
    custom_objects: MagicMock, runtime_dependencies: Dependencies
) -> None:
    """The API, rather than Kueue, owns initial partition eligibility."""
    scheduler = JobScheduler(runtime_dependencies)
    job_id = UUID("00000000-0000-0000-0000-000000000001")
    scheduler._dataset_job = MagicMock(
        return_value={
            "name": "sp500",
            "version": 3,
            "dataset_type": "market_data",
            "config": {
                "description": "Market data.",
                "validations": [],
                "keys": ["symbol"],
                "partition_type": "daily",
                "symbols": ["AAPL"],
            },
            "callbacks": [],
        }
    )

    scheduler._ensure_dataset_application(job_id, "dataset-job")

    manifest = custom_objects.return_value.create_namespaced_custom_object.call_args.args[
        -1
    ]
    assert (
        manifest["metadata"]["annotations"]["kueue.x-k8s.io/admission-gated-by"]
        == ADMISSION_GATE
    )
    assert manifest["spec"]["suspend"]
    assert manifest["spec"]["arguments"].count("--config-json") == 1
    assert "--partition-callbacks-json" not in manifest["spec"]["arguments"]
    assert "--dataset-type" not in manifest["spec"]["arguments"]
    config = json.loads(
        manifest["spec"]["arguments"][
            manifest["spec"]["arguments"].index("--config-json") + 1
        ]
    )
    assert config["dataset_type"] == "market_data"
    assert config["name"] == "sp500"
    assert config["version"] == 3
    assert config["environment"] == "test"
    assert config["warehouse_dir"] == "s3a://warehouse"
    assert config["partitions"] == []


@patch("ml_platform.api.services._scheduler.client.CustomObjectsApi")
def test_terminal_spark_states_finish_the_api_job(
    custom_objects: MagicMock, runtime_dependencies: Dependencies
) -> None:
    """A Spark failure is persisted as a failed control-plane job."""
    scheduler = JobScheduler(runtime_dependencies)
    custom_objects.return_value.get_namespaced_custom_object.return_value = {
        "status": {"applicationState": {"state": "FAILED"}}
    }
    scheduler._finish = MagicMock()

    scheduler._observe_application(
        UUID("00000000-0000-0000-0000-000000000001"),
        "dataset-job",
        "dataset_materialization",
    )

    scheduler._finish.assert_called_once_with(
        UUID("00000000-0000-0000-0000-000000000001"),
        "failed",
        "dataset_materialization",
    )
