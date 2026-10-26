"""Unit coverage for scheduler Kubernetes reconciliation."""

import json
from unittest.mock import MagicMock, call, patch
from uuid import UUID

from kubernetes.client.exceptions import ApiException

from ml_platform.api import Dependencies
from ml_platform.api.domain import JobStatus
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


@patch("ml_platform.api.services._scheduler.client.CustomObjectsApi")
def test_cancelled_spark_application_cancels_the_api_job(
    custom_objects: MagicMock, runtime_dependencies: Dependencies
) -> None:
    """A cancelled Spark application persists a cancelled control-plane job."""
    scheduler = JobScheduler(runtime_dependencies)
    job_id = UUID("00000000-0000-0000-0000-000000000002")
    custom_objects.return_value.get_namespaced_custom_object.return_value = {
        "status": {"applicationState": {"state": "CANCELLED"}}
    }
    scheduler._finish = MagicMock()

    scheduler._observe_application(
        job_id,
        "dataset-job",
        "dataset_materialization",
    )

    scheduler._finish.assert_called_once_with(
        job_id,
        JobStatus.CANCELLED,
        "dataset_materialization",
    )


@patch("ml_platform.api.services._scheduler.client.CustomObjectsApi")
def test_unrecognized_spark_application_state_is_not_terminal(
    custom_objects: MagicMock, runtime_dependencies: Dependencies
) -> None:
    """An unrecognized Spark state remains submitted for later reconciliation."""
    scheduler = JobScheduler(runtime_dependencies)
    custom_objects.return_value.get_namespaced_custom_object.return_value = {
        "status": {"applicationState": {"state": "RUNNING"}}
    }
    scheduler._finish = MagicMock()

    scheduler._observe_application(
        UUID("00000000-0000-0000-0000-000000000003"),
        "dataset-job",
        "dataset_materialization",
    )

    scheduler._finish.assert_not_called()


def test_reconcile_releases_only_eligible_jobs(
    runtime_dependencies: Dependencies,
) -> None:
    """Scheduler reconciliation retains an admission gate for blocked jobs."""
    scheduler = JobScheduler(runtime_dependencies)
    first_job_id = UUID("00000000-0000-0000-0000-000000000004")
    second_job_id = UUID("00000000-0000-0000-0000-000000000005")
    scheduler._ensure_dataset_application = MagicMock()
    scheduler._is_eligible = MagicMock(side_effect=[True, False])
    scheduler._remove_admission_gate = MagicMock()
    scheduler._observe_application = MagicMock()

    with patch.object(
        type(scheduler._repository),
        "submitted_jobs",
        return_value=[
            {"id": first_job_id, "type": "dataset_materialization"},
            {"id": second_job_id, "type": "dataset_materialization"},
        ],
    ):
        scheduler._reconcile_submitted_jobs()

    assert scheduler._ensure_dataset_application.call_count == 2
    scheduler._remove_admission_gate.assert_called_once_with(
        f"dataset-{first_job_id}"
    )
    assert scheduler._observe_application.call_args_list == [
        call(
            first_job_id, f"dataset-{first_job_id}", "dataset_materialization"
        ),
        call(
            second_job_id, f"dataset-{second_job_id}", "dataset_materialization"
        ),
    ]


@patch("ml_platform.api.services._scheduler.client.CustomObjectsApi")
def test_kubernetes_error_leaves_job_submitted_for_a_later_reconciliation(
    custom_objects: MagicMock,
    runtime_dependencies: Dependencies,
) -> None:
    """A transient Kubernetes error does not discard an API-owned job."""
    scheduler = JobScheduler(runtime_dependencies)
    job_id = UUID("00000000-0000-0000-0000-000000000006")
    custom_objects.return_value.create_namespaced_custom_object.side_effect = (
        ApiException(status=503, reason="Unavailable")
    )

    with (
        patch.object(
            type(scheduler._repository),
            "submitted_jobs",
            return_value=[{"id": job_id, "type": "dataset_materialization"}],
        ),
        patch.object(
            type(scheduler._repository),
            "dataset_job",
            return_value={
                "name": "prices",
                "version": 1,
                "dataset_type": "market_data",
                "config": {
                    "description": "Prices.",
                    "validations": [],
                    "keys": ["symbol"],
                    "partition_type": "daily",
                    "symbols": ["AAA"],
                },
                "callbacks": [],
            },
        ),
    ):
        scheduler._reconcile_submitted_jobs()

    custom_objects.return_value.create_namespaced_custom_object.assert_called_once()


@patch("ml_platform.api.services._scheduler.client.CustomObjectsApi")
def test_feature_application_is_created_with_resolved_worker_config(
    custom_objects: MagicMock, runtime_dependencies: Dependencies
) -> None:
    """Feature Spark applications carry the scheduler-resolved worker config."""
    scheduler = JobScheduler(runtime_dependencies)
    job_id = UUID("00000000-0000-0000-0000-000000000003")
    scheduler._feature_job = MagicMock(
        return_value={
            "config": {
                "name": "log_return",
                "version": 1,
                "dtype": "double",
                "frequency": "daily",
                "keys": ["symbol"],
                "validations": [],
                "datasets": [],
                "transform": "ml_platform.data_sdk.LogReturnTransform",
                "description": "Log return.",
                "parameters": {},
            },
            "callbacks": [],
        }
    )

    scheduler._ensure_feature_application(job_id, "feature-job")

    manifest = custom_objects.return_value.create_namespaced_custom_object.call_args.args[
        -1
    ]
    assert manifest["spec"]["mainApplicationFile"].endswith("_feature.py")
    config = json.loads(manifest["spec"]["arguments"][1])
    assert config["environment"] == "test"
    assert config["warehouse_dir"] == "s3a://warehouse"
    assert config["partitions"] == []


@patch("ml_platform.api.services._scheduler.client.CustomObjectsApi")
def test_label_application_is_created_with_resolved_worker_config(
    custom_objects: MagicMock, runtime_dependencies: Dependencies
) -> None:
    """Label Spark applications carry the scheduler-resolved worker config."""
    scheduler = JobScheduler(runtime_dependencies)
    job_id = UUID("00000000-0000-0000-0000-000000000004")
    scheduler._label_job = MagicMock(
        return_value={
            "config": {
                "name": "triple_barrier",
                "version": 1,
                "dtype": "int",
                "frequency": "daily",
                "keys": ["symbol"],
                "validations": [],
                "datasets": [],
                "transform": "ml_platform.data_sdk.TripleBarrierLabelTransform",
                "description": "Triple barrier.",
                "parameters": {},
            },
            "callbacks": [],
        }
    )

    scheduler._ensure_label_application(job_id, "label-job")

    manifest = custom_objects.return_value.create_namespaced_custom_object.call_args.args[
        -1
    ]
    assert manifest["spec"]["mainApplicationFile"].endswith("_label.py")
    config = json.loads(manifest["spec"]["arguments"][1])
    assert config["environment"] == "test"
    assert config["warehouse_dir"] == "s3a://warehouse"
    assert config["partitions"] == []
