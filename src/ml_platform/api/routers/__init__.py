"""Public router composition for the FastAPI application."""

from ml_platform.api.routers._datasets import (
    create_dataset,
    delete_dataset,
    list_datasets,
    materialize_dataset_partitions,
)
from ml_platform.api.routers._datasets import (
    router as datasets_router,
)
from ml_platform.api.routers._features import (
    materialize_feature_partitions,
)
from ml_platform.api.routers._features import (
    router as features_router,
)
from ml_platform.api.routers._jobs import get_job
from ml_platform.api.routers._jobs import router as jobs_router
from ml_platform.api.routers._labels import router as labels_router
from ml_platform.api.routers._partitions import (
    complete_partitions,
)
from ml_platform.api.routers._partitions import (
    router as partitions_router,
)
from ml_platform.api.routers._ui import mount_ui
from ml_platform.api.routers._ui import router as ui_router

__all__ = [
    "complete_partitions",
    "create_dataset",
    "datasets_router",
    "delete_dataset",
    "features_router",
    "get_job",
    "jobs_router",
    "labels_router",
    "list_datasets",
    "materialize_dataset_partitions",
    "materialize_feature_partitions",
    "mount_ui",
    "partitions_router",
    "ui_router",
]
