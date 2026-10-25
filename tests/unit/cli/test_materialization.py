"""CLI resource synchronization tests."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ml_platform.cli.materialization as materialization
from ml_platform.cli.materialization import sync_resources
from ml_platform.common.api.dtos import ResourceType


class ResourceSyncTest(unittest.TestCase):
    """Verify config-defined resource identities drive create and soft-delete calls."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        (self.root / "datasets").mkdir()
        (self.root / "features").mkdir()
        (self.root / "datasets" / "prices.yaml").write_text("""name: prices
version: 2
dataset_type: market_data
description: Test prices.
partition_type: daily
keys: [symbol]
symbols: [AAPL]
""")
        (self.root / "features" / "returns.yaml").write_text("""name: returns
version: 1
dtype: double
frequency: daily
keys: [symbol]
datasets:
  - name: prices
    version: 2
    input_window:
      lookback_partitions: 1
transform: ml_platform.data_sdk.LogReturnTransform
description: Returns
validations:
  - type: minimum_rows
    minimum_rows: 1
    reason: Each requested partition requires a row.
""")

    def test_dataset_payload_conforms_to_the_public_api_contract(self) -> None:
        """Internal dataset settings never leak into the public request DTO."""
        payload = materialization._dataset_payload(
            materialization._load_yaml(self.root / "datasets" / "prices.yaml")
        )

        self.assertNotIn("run_frequency", payload["config"])
        self.assertNotIn("num_lookback_partitions", payload["config"])
        self.assertNotIn("id", payload)

    @patch("ml_platform.cli.materialization._api_request")
    def test_dataset_sync_creates_configured_and_deletes_removed(
        self, request: object
    ) -> None:
        request.side_effect = [
            [
                {"name": "obsolete", "version": 1, "deleted_at": None},
                {
                    "name": "prices",
                    "version": 2,
                    "deleted_at": None,
                    "config": {"description": "Test prices."},
                },
            ],
            {},
        ]
        result = sync_resources("http://api", ResourceType.DATASET, self.root)
        self.assertEqual(result, {"created": [], "deleted": ["obsolete v1"]})
        self.assertEqual(
            request.call_args_list[1].args[1:3],
            ("DELETE", "/datasets/obsolete/versions/1"),
        )

    @patch("ml_platform.cli.materialization._api_request")
    def test_feature_sync_resolves_dataset_identity(
        self, request: object
    ) -> None:
        request.side_effect = [[], {}]
        result = sync_resources("http://api", ResourceType.FEATURE, self.root)
        self.assertEqual(result, {"created": ["returns v1"], "deleted": []})
        payload = request.call_args_list[1].args[3]
        self.assertEqual(
            payload["datasets"],
            [
                {
                    "name": "prices",
                    "version": 2,
                    "input_window": {"lookback_partitions": 1},
                }
            ],
        )


class HttpClientTest(unittest.TestCase):
    """Verify the CLI preserves JSON response shapes from the API."""

    @patch("ml_platform.cli.materialization.httpx2.request")
    def test_api_request_accepts_list_responses(self, request: object) -> None:
        """Resource-list endpoints return JSON arrays rather than objects."""
        response = request.return_value
        response.content = b'[{"name":"prices"}]'
        response.json.return_value = [{"name": "prices"}]

        result = materialization._api_request(
            "http://api", materialization.HttpMethod.GET, "/datasets"
        )

        self.assertEqual(result, [{"name": "prices"}])
        response.raise_for_status.assert_called_once_with()
