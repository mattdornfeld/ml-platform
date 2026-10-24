"""Unit coverage for scheduler Kubernetes reconciliation."""

import json
import unittest
from unittest.mock import MagicMock, patch
from uuid import UUID

from ml_platform.api import Dependencies
from ml_platform.api.services import ADMISSION_GATE, JobScheduler
from ml_platform.common.types import Environment


def _dependencies() -> Dependencies:
    """Return fixed runtime dependencies for isolated scheduler tests."""
    return Dependencies(
        environment=Environment.LOCAL,
        database_url="postgresql://unused",
        kubernetes_namespace="ml-platform",
        materialization_image="ml-platform:latest",
        warehouse_dir="s3a://warehouse",
        s3_endpoint="http://minio:9000",
        minio_root_user="minio",
        minio_root_password="miniosecret",
        headlamp_url="",
        minio_console_url="",
    )


class SchedulerTest(unittest.TestCase):
    """Cover manifest construction and terminal-state mapping without Kubernetes."""

    @patch("ml_platform.api.services._scheduler.client.CustomObjectsApi")
    def test_dataset_application_is_created_admission_gated(
        self, custom_objects: MagicMock
    ) -> None:
        """The API, rather than Kueue, owns initial partition eligibility."""
        scheduler = JobScheduler(_dependencies())
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
        self.assertEqual(
            manifest["metadata"]["annotations"][
                "kueue.x-k8s.io/admission-gated-by"
            ],
            ADMISSION_GATE,
        )
        self.assertTrue(manifest["spec"]["suspend"])
        self.assertEqual(
            manifest["spec"]["arguments"].count("--config-json"), 1
        )
        self.assertNotIn(
            "--partition-callbacks-json", manifest["spec"]["arguments"]
        )
        self.assertNotIn("--dataset-type", manifest["spec"]["arguments"])
        config = json.loads(
            manifest["spec"]["arguments"][
                manifest["spec"]["arguments"].index("--config-json") + 1
            ]
        )
        self.assertEqual(config["dataset_type"], "market_data")
        self.assertEqual(config["name"], "sp500")
        self.assertEqual(config["version"], 3)
        self.assertEqual(config["environment"], "local")
        self.assertEqual(config["warehouse_dir"], "s3a://warehouse")
        self.assertEqual(config["partitions"], [])

    @patch("ml_platform.api.services._scheduler.client.CustomObjectsApi")
    def test_terminal_spark_states_finish_the_api_job(
        self, custom_objects: MagicMock
    ) -> None:
        """A Spark failure is persisted as a failed control-plane job."""
        scheduler = JobScheduler(_dependencies())
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
