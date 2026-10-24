"""Runtime dependencies used by the control-plane API and its scheduler."""

import os
from dataclasses import dataclass

from ml_platform.api._exceptions import InvalidRuntimeEnvironmentError
from ml_platform.common.types import Environment


@dataclass(frozen=True)
class Dependencies:
    """Typed operational dependencies configured for one API process."""

    environment: Environment
    database_url: str
    kubernetes_namespace: str
    materialization_image: str
    warehouse_dir: str
    s3_endpoint: str
    minio_root_user: str
    minio_root_password: str
    headlamp_url: str
    minio_console_url: str

    @classmethod
    def from_environment(cls) -> "Dependencies":
        """Load the API process's explicit runtime dependency configuration."""
        database_url = os.environ.get("DATABASE_URL", "")
        environment_value = os.environ.get(
            "ML_PLATFORM_ENVIRONMENT", Environment.LOCAL
        )
        try:
            environment = Environment(environment_value)
        except ValueError as error:
            raise InvalidRuntimeEnvironmentError(environment_value) from error
        return cls(
            environment=environment,
            database_url=database_url,
            kubernetes_namespace=os.environ.get(
                "KUBERNETES_NAMESPACE", "ml-platform"
            ),
            materialization_image=os.environ.get(
                "MATERIALIZATION_IMAGE", "ml-platform:latest"
            ),
            warehouse_dir=os.environ.get(
                "DATA_WAREHOUSE_DIR", "s3a://warehouse"
            ),
            s3_endpoint=os.environ.get("S3_ENDPOINT", "http://minio:9000"),
            minio_root_user=os.environ.get("MINIO_ROOT_USER", "minio"),
            minio_root_password=os.environ.get(
                "MINIO_ROOT_PASSWORD", "miniosecret"
            ),
            headlamp_url=os.environ.get("HEADLAMP_URL", ""),
            minio_console_url=os.environ.get("MINIO_CONSOLE_URL", ""),
        )
