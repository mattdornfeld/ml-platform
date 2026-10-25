"""Validated runtime dependencies for the control-plane API."""

import os

from pydantic import Field, model_validator

from ml_platform.api._exceptions import (
    DatabaseUrlNotConfiguredError,
    InvalidDependencyConfigurationError,
    InvalidRuntimeEnvironmentError,
)
from ml_platform.common.types import (
    BaseConfig,
    DatabaseUrl,
    Environment,
    HeadlampUrl,
    KubernetesNamespace,
    MaterializationImage,
    MinioAccessKey,
    MinioConsoleUrl,
    MinioSecretKey,
    S3Endpoint,
    WarehouseDirectory,
)


class Dependencies(BaseConfig):
    """Immutable operational dependencies for one API call path."""

    environment: Environment = Field(description="Runtime environment.")
    database_url: DatabaseUrl = Field(
        min_length=1, description="PostgreSQL connection URL."
    )
    kubernetes_namespace: KubernetesNamespace = Field(
        min_length=1, description="Kubernetes namespace."
    )
    materialization_image: MaterializationImage = Field(
        min_length=1, description="Spark job image."
    )
    warehouse_dir: WarehouseDirectory = Field(
        min_length=1, description="Iceberg warehouse URI."
    )
    s3_endpoint: S3Endpoint = Field(
        min_length=1, description="S3-compatible endpoint URL."
    )
    minio_root_user: MinioAccessKey = Field(
        min_length=1, description="MinIO access key."
    )
    minio_root_password: MinioSecretKey = Field(
        min_length=1, description="MinIO secret key."
    )
    headlamp_url: HeadlampUrl = Field(
        min_length=1, description="Headlamp base URL."
    )
    minio_console_url: MinioConsoleUrl = Field(
        min_length=1, description="MinIO Console base URL."
    )

    @model_validator(mode="after")
    def _validate_non_blank_values(self) -> "Dependencies":
        """Reject whitespace-only runtime values after model parsing."""
        values = {
            "database_url": self.database_url,
            "kubernetes_namespace": self.kubernetes_namespace,
            "materialization_image": self.materialization_image,
            "warehouse_dir": self.warehouse_dir,
            "s3_endpoint": self.s3_endpoint,
            "minio_root_user": self.minio_root_user,
            "minio_root_password": self.minio_root_password,
            "headlamp_url": self.headlamp_url,
            "minio_console_url": self.minio_console_url,
        }
        for dependency_name, value in values.items():
            if not value.strip():
                raise InvalidDependencyConfigurationError(dependency_name)
        return self

    @classmethod
    def from_environment(cls) -> "Dependencies":
        """Read and validate this process's runtime dependencies once."""
        try:
            environment = Environment(
                os.environ.get("ML_PLATFORM_ENVIRONMENT", Environment.LOCAL)
            )
        except ValueError as error:
            raise InvalidRuntimeEnvironmentError(
                os.environ.get("ML_PLATFORM_ENVIRONMENT", "")
            ) from error
        try:
            database_url = DatabaseUrl(os.environ["DATABASE_URL"])
        except KeyError as error:
            raise DatabaseUrlNotConfiguredError from error
        if not database_url.strip():
            raise DatabaseUrlNotConfiguredError
        return cls(
            environment=environment,
            database_url=database_url,
            kubernetes_namespace=KubernetesNamespace(
                os.environ.get("KUBERNETES_NAMESPACE", "ml-platform")
            ),
            materialization_image=MaterializationImage(
                os.environ.get("MATERIALIZATION_IMAGE", "ml-platform:latest")
            ),
            warehouse_dir=WarehouseDirectory(
                os.environ.get("DATA_WAREHOUSE_DIR", "s3a://warehouse")
            ),
            s3_endpoint=S3Endpoint(
                os.environ.get("S3_ENDPOINT", "http://minio:9000")
            ),
            minio_root_user=MinioAccessKey(
                os.environ.get("MINIO_ROOT_USER", "minio")
            ),
            minio_root_password=MinioSecretKey(
                os.environ.get("MINIO_ROOT_PASSWORD", "miniosecret")
            ),
            headlamp_url=HeadlampUrl(
                os.environ.get("HEADLAMP_URL", "http://127.0.0.1:8002")
            ),
            minio_console_url=MinioConsoleUrl(
                os.environ.get("MINIO_CONSOLE_URL", "http://127.0.0.1:9003")
            ),
        )
