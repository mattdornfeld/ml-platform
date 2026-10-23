"""Tests for the shared public control-plane route contract."""

import unittest
from uuid import UUID

from ml_platform.common.api.routes import (
    DatasetRoutes,
    FeatureRoutes,
    HttpMethod,
    JobRoutes,
    LabelRoutes,
    PartitionRoutes,
)


class ApiRoutesTest(unittest.TestCase):
    """Verify clients and the API share stable route specifications."""

    def test_resource_routes_include_methods_and_versioned_paths(self) -> None:
        """Dataset, feature, and label routes render their resource identity."""
        self.assertEqual(DatasetRoutes.CREATE.method, HttpMethod.POST)
        self.assertEqual(
            DatasetRoutes.MATERIALIZE.path(name="sp500", version=3),
            "/datasets/sp500/versions/3/partitions",
        )
        self.assertEqual(
            FeatureRoutes.DETAIL.path(name="log_return", version=1),
            "/features/log_return/versions/1",
        )
        self.assertEqual(
            LabelRoutes.DELETE.path(name="triple_barrier", version=2),
            "/labels/triple_barrier/versions/2",
        )

    def test_job_and_completion_routes_render_public_worker_paths(self) -> None:
        """Jobs and workers use the shared API path definitions."""
        job_id = UUID("00000000-0000-0000-0000-000000000001")
        self.assertEqual(JobRoutes.DETAIL.method, HttpMethod.GET)
        self.assertEqual(
            JobRoutes.DETAIL.path(job_id=job_id), f"/jobs/{job_id}"
        )
        self.assertEqual(
            PartitionRoutes.COMPLETE.template, "/partitions/complete"
        )
