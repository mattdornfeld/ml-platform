"""CLI resource synchronization tests."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml_platform.cli.materialization import sync_resources


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
keys: [symbol]
symbols: [AAPL]
time_range: {start_date: 2026-01-01, end_date: 2026-01-02}
""")
        (self.root / "features" / "returns.yaml").write_text("""name: returns
version: 1
dtype: double
frequency: daily
keys: [symbol]
datasets:
  - name: prices
    version: 2
    dataset_type: market_data
    input_window:
      lookback_partitions: 1
transform: ml_platform.data_sdk.LogReturnTransform
description: Returns
validations:
  - type: minimum_rows
    minimum_rows: 1
    reason: Each requested partition requires a row.
""")

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
        result = sync_resources("http://api", "dataset", self.root)
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
        result = sync_resources("http://api", "feature", self.root)
        self.assertEqual(result, {"created": ["returns v1"], "deleted": []})
        payload = request.call_args_list[1].args[3]
        self.assertEqual(
            payload["datasets"],
            [
                {
                    "name": "prices",
                    "version": 2,
                    "dataset_type": "market_data",
                    "input_window": {"lookback_partitions": 1},
                }
            ],
        )
