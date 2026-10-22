"""Unit coverage for scheduler Kubernetes reconciliation."""

import json
import unittest
from unittest.mock import MagicMock, patch
from uuid import UUID

from ml_platform.api.scheduler import ADMISSION_GATE, JobScheduler


class SchedulerTest(unittest.TestCase):
    """Cover manifest construction and terminal-state mapping without Kubernetes."""

    @patch("ml_platform.api.scheduler.client.CustomObjectsApi")
    def test_dataset_application_is_created_admission_gated(
        self, custom_objects: MagicMock
    ) -> None:
        """The API, rather than Kueue, owns initial partition eligibility."""
        scheduler = JobScheduler("postgresql://unused")
        job_id = UUID("00000000-0000-0000-0000-000000000001")
        scheduler._dataset_job = MagicMock(
            return_value={
                "name": "sp500",
                "version": 3,
                "dataset_type": "market_data",
                "config": {
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
        self.assertEqual(config["warehouse_dir"], "s3a://warehouse")
        self.assertEqual(config["partitions"], [])

    @patch("ml_platform.api.scheduler.client.CustomObjectsApi")
    def test_terminal_spark_states_finish_the_api_job(
        self, custom_objects: MagicMock
    ) -> None:
        """A Spark failure is persisted as a failed control-plane job."""
        scheduler = JobScheduler("postgresql://unused")
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
