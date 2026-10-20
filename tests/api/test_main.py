"""Validation tests for materialization API routes."""

import unittest

from fastapi.testclient import TestClient

from api.main import app


class PartitionMaterializationRequestTest(unittest.TestCase):
    """Verify API validation for partition request timestamps."""

    def setUp(self) -> None:
        """Create an isolated in-process API client."""
        self.client = TestClient(app)

    def test_dataset_partition_request_rejects_non_hour_timestamp(self) -> None:
        """Partition endpoints reject timestamps that are not aligned to the hour."""
        response = self.client.post(
            "/datasets/market_data/versions/1/partitions",
            json={"timestamps": ["2026-10-20T18:15:47.828Z"]},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("aligned to an exact hour", response.text)

    def test_feature_partition_request_accepts_hour_aligned_utc_timestamp(self) -> None:
        """Hour-aligned UTC timestamps reach the feature materialization stub."""
        response = self.client.post(
            "/features/trailing_volatility_1d/versions/1/partitions",
            json={"timestamps": ["2026-10-20T18:00:00Z"]},
        )

        self.assertEqual(response.status_code, 501)
