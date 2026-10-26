"""End-to-end verification of the complete local ML Platform deployment."""

import time

import httpx2
import pytest
from tests.e2e.conftest import DeployedPlatform

from ml_platform.data_sdk.utils.data_explorer import load_partitioned_parquet

pytestmark = pytest.mark.e2e


def test_required_services_are_ready(
    deployed_platform: DeployedPlatform,
) -> None:
    """Every declared platform service reaches Kubernetes-ready state."""
    deployments = deployed_platform.kubectl(
        "get",
        "deployments",
        "--all-namespaces",
        "--output",
        'jsonpath={range .items[*]}{.metadata.namespace}/{.metadata.name}{"\\n"}{end}',
    ).stdout.splitlines()

    assert "ml-platform/ml-control-plane" in deployments
    assert "ml-platform/minio" in deployments
    assert "ml-platform/postgres" in deployments
    assert "spark-operator/spark-operator-controller" in deployments
    assert "kueue-system/kueue-controller-manager" in deployments
    assert "headlamp/headlamp" in deployments


type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type JsonObject = dict[str, JsonValue]


class E2eApiResponseError(Exception):
    """Raised when an end-to-end API response is not a JSON object."""


class E2eJobTimeoutError(Exception):
    """Raised when a submitted control-plane job does not finish in time."""


def _string_field(document: JsonObject, field_name: str) -> str:
    """Return one required string field from an API response object."""
    value = document[field_name]
    if not isinstance(value, str):
        raise E2eApiResponseError(
            f"Expected {field_name} to be a string, received {type(value).__name__}."
        )
    return value


def _object_array_field(
    document: JsonObject,
    field_name: str,
) -> list[JsonObject]:
    """Return one required array of API response objects."""
    value = document[field_name]
    if not isinstance(value, list):
        raise E2eApiResponseError(
            f"Expected {field_name} to be an array, received {type(value).__name__}."
        )
    objects: list[JsonObject] = []
    for item in value:
        if not isinstance(item, dict):
            raise E2eApiResponseError(
                f"Expected {field_name} to contain only response objects."
            )
        objects.append(item)
    return objects


def _api_request(
    base_url: str,
    method: str,
    path: str,
    payload: JsonObject,
) -> JsonObject:
    """Make one required request to the public control-plane API."""
    response = httpx2.request(
        method=method,
        url=f"{base_url}{path}",
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    document = response.json()
    if not isinstance(document, dict):
        raise E2eApiResponseError(
            f"Expected JSON object from {method} {path}, received {type(document).__name__}."
        )
    return document


def _job_id(response: JsonObject) -> str:
    """Extract one required submitted job identifier from an API response."""
    submitted_job = response["submitted_job"]
    if not isinstance(submitted_job, dict):
        raise E2eApiResponseError(
            "Materialization response lacks submitted_job."
        )
    return _string_field(submitted_job, "id")


def _wait_for_job(
    deployed_platform: DeployedPlatform,
    base_url: str,
    job_id: str,
) -> JsonObject:
    """Wait for one submitted materialization job to reach a terminal state."""
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        job = _api_request(base_url, "GET", f"/jobs/{job_id}", {})
        status = job.get("status")
        if status == "succeeded":
            return job
        if status in {"failed", "cancelled"}:
            application_name = _string_field(job, "spark_application_name")
            application_status = deployed_platform.spark_application_status(
                application_name
            )
            raise E2eApiResponseError(
                f"Materialization job {job_id} completed with status {status!r}: "
                f"{application_status}"
            )
        time.sleep(2)
    raise E2eJobTimeoutError(
        f"Materialization job {job_id} did not finish within five minutes."
    )


def _wait_for_failed_job(
    deployed_platform: DeployedPlatform,
    base_url: str,
    job_id: str,
) -> JsonObject:
    """Wait for one job that must fail through the real scheduler lifecycle."""
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        job = _api_request(base_url, "GET", f"/jobs/{job_id}", {})
        status = job.get("status")
        if status == "failed":
            return job
        if status in {"succeeded", "cancelled"}:
            raise E2eApiResponseError(
                f"Materialization job {job_id} unexpectedly ended as {status!r}."
            )
        time.sleep(2)
    raise E2eJobTimeoutError(
        f"Materialization job {job_id} did not fail within five minutes."
    )


def _wait_for_cancelled_job(
    base_url: str,
    job_id: str,
) -> JsonObject:
    """Wait for one job that must be cancelled by the scheduler lifecycle."""
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        job = _api_request(base_url, "GET", f"/jobs/{job_id}", {})
        status = job.get("status")
        if status == "cancelled":
            return job
        if status in {"succeeded", "failed"}:
            raise E2eApiResponseError(
                f"Materialization job {job_id} unexpectedly ended as {status!r}."
            )
        time.sleep(1)
    raise E2eJobTimeoutError(
        f"Materialization job {job_id} did not cancel within one minute."
    )


def _wait_for_spark_application(
    deployed_platform: DeployedPlatform,
    application_name: str,
) -> None:
    """Wait until the scheduler has persisted a SparkApplication resource."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        result = deployed_platform.kubectl(
            "get",
            "sparkapplication",
            application_name,
            "--namespace",
            "ml-platform",
            "--ignore-not-found",
            "--output",
            "name",
        )
        if result.stdout.strip():
            return
        time.sleep(1)
    raise E2eJobTimeoutError(
        f"SparkApplication {application_name} was not created within 30 seconds."
    )


def _wait_for_kueue_workload(deployed_platform: DeployedPlatform) -> str:
    """Return a Kueue workload created for an active SparkApplication."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        workloads = deployed_platform.kubectl(
            "get",
            "workloads",
            "--namespace",
            "ml-platform",
            "--output",
            "name",
        ).stdout.splitlines()
        if workloads:
            workload = workloads[0]
            queue_name = deployed_platform.kubectl(
                "get",
                workload,
                "--namespace",
                "ml-platform",
                "--output",
                "jsonpath={.spec.queueName}",
            ).stdout.strip()
            if queue_name == "materialization":
                return workload
        time.sleep(1)
    raise E2eJobTimeoutError("Kueue did not create a materialization workload.")


def _dataset_payload() -> JsonObject:
    """Return a small synthetic market-data dataset definition for K3s tests."""
    return {
        "name": "market_data",
        "version": 1,
        "dataset_type": "market_data",
        "config": {
            "description": "Synthetic end-to-end market data.",
            "dataset_type": "market_data",
            "keys": ["symbol"],
            "partition_type": "daily",
            "symbols": ["AAA"],
            "validations": [],
        },
    }


def _feature_payload() -> JsonObject:
    """Return a log-return feature definition over the end-to-end dataset."""
    return {
        "name": "log_return",
        "version": 1,
        "dtype": "double",
        "frequency": "daily",
        "keys": ["symbol"],
        "datasets": [
            {
                "name": "market_data",
                "version": 1,
                "input_window": {"lookback_partitions": 1},
            }
        ],
        "transform": "ml_platform.data_sdk.LogReturnTransform",
        "description": "Synthetic end-to-end log returns.",
        "parameters": {},
        "validations": [],
    }


def _broken_feature_payload() -> JsonObject:
    """Return a feature whose missing transform must fail only in a Spark worker."""
    return {
        "name": "broken_feature",
        "version": 1,
        "dtype": "double",
        "frequency": "daily",
        "keys": ["symbol"],
        "datasets": [
            {
                "name": "failure_prices",
                "version": 1,
                "input_window": {"lookback_partitions": 0},
            }
        ],
        "transform": "ml_platform.data_sdk.MissingFeatureTransform",
        "description": "Intentional worker failure for end-to-end coverage.",
        "parameters": {},
        "validations": [],
    }


def _label_payload() -> JsonObject:
    """Return a triple-barrier label definition over the end-to-end dataset."""
    return {
        "name": "triple_barrier",
        "version": 1,
        "dtype": "int",
        "frequency": "daily",
        "keys": ["symbol"],
        "datasets": [
            {
                "name": "market_data",
                "version": 1,
                "input_window": {
                    "lookback_partitions": 3,
                    "lookahead_partitions": 1,
                },
            }
        ],
        "transform": "ml_platform.data_sdk.TripleBarrierLabelTransform",
        "description": "Synthetic end-to-end triple-barrier labels.",
        "parameters": {
            "horizon_bars": 1,
            "volatility_lookback_bars": 2,
            "profit_taking_multiplier": 1.0,
            "stop_loss_multiplier": 1.0,
        },
        "validations": [],
    }


def _assert_iceberg_data(
    deployed_platform: DeployedPlatform,
    namespace: str,
    table_name: str,
) -> None:
    """Require an Iceberg table to contain physical Parquet data in MinIO."""
    result = deployed_platform.kubectl(
        "exec",
        "deployment/minio",
        "--namespace",
        "ml-platform",
        "--",
        "/bin/sh",
        "-ec",
        (
            "mc alias set local http://127.0.0.1:9000 minio miniosecret >/dev/null; "
            f"mc find local/warehouse/{namespace}/{table_name}/data -name '*.parquet'"
        ),
    )
    assert result.stdout.strip()


def test_materializes_dataset_feature_and_label_through_kubernetes(
    deployed_platform: DeployedPlatform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise API scheduling, Kueue, Spark, Iceberg, MinIO, and callbacks."""
    with deployed_platform.api_url() as api_url:
        assert (
            _api_request(api_url, "POST", "/datasets", _dataset_payload())[
                "name"
            ]
            == "market_data"
        )
        dataset_request = _api_request(
            api_url,
            "POST",
            "/datasets/market_data/versions/1/partitions",
            {
                "timestamps": [
                    "2026-01-05T00:00:00Z",
                    "2026-01-06T00:00:00Z",
                    "2026-01-07T00:00:00Z",
                    "2026-01-08T00:00:00Z",
                    "2026-01-09T00:00:00Z",
                ]
            },
        )
        assert _wait_for_kueue_workload(deployed_platform).startswith(
            "workload.kueue.x-k8s.io/"
        )
        _wait_for_job(deployed_platform, api_url, _job_id(dataset_request))
        dataset = _api_request(
            api_url,
            "GET",
            "/datasets/market_data/versions/1",
            {},
        )
        partitions = _object_array_field(dataset, "partitions")
        assert len(partitions) == 5
        assert {
            _string_field(partition, "status") for partition in partitions
        } == {"created"}
        _assert_iceberg_data(deployed_platform, "market_data", "market_data-v1")
        with deployed_platform.minio_url() as minio_url:
            monkeypatch.setenv("S3_ENDPOINT", minio_url)
            data_frame = load_partitioned_parquet(
                "s3a://warehouse/market_data/market_data-v1/data"
            )
        assert set(data_frame["symbol"]) == {"AAA"}

        assert (
            _api_request(api_url, "POST", "/features", _feature_payload())[
                "name"
            ]
            == "log_return"
        )
        feature_request = _api_request(
            api_url,
            "POST",
            "/features/log_return/versions/1/partitions",
            {"timestamps": ["2026-01-09T00:00:00Z"]},
        )
        _wait_for_job(deployed_platform, api_url, _job_id(feature_request))
        feature = _api_request(
            api_url,
            "GET",
            "/features/log_return/versions/1",
            {},
        )
        assert (
            _string_field(
                _object_array_field(feature, "partitions")[0], "status"
            )
            == "created"
        )
        _assert_iceberg_data(deployed_platform, "features", "log_return-v1")

        assert (
            _api_request(api_url, "POST", "/labels", _label_payload())["name"]
            == "triple_barrier"
        )
        label_request = _api_request(
            api_url,
            "POST",
            "/labels/triple_barrier/versions/1/partitions",
            {"timestamps": ["2026-01-08T00:00:00Z"]},
        )
        _wait_for_job(deployed_platform, api_url, _job_id(label_request))
        label = _api_request(
            api_url,
            "GET",
            "/labels/triple_barrier/versions/1",
            {},
        )
        assert (
            _string_field(_object_array_field(label, "partitions")[0], "status")
            == "created"
        )
        _assert_iceberg_data(deployed_platform, "labels", "triple_barrier-v1")


def test_failed_spark_transform_marks_job_and_partition_failed(
    deployed_platform: DeployedPlatform,
) -> None:
    """A worker exception becomes durable failed control-plane state."""
    dataset_payload = _dataset_payload()
    dataset_payload["name"] = "failure_prices"
    with deployed_platform.api_url() as api_url:
        assert (
            _api_request(api_url, "POST", "/datasets", dataset_payload)["name"]
            == "failure_prices"
        )
        dataset_request = _api_request(
            api_url,
            "POST",
            "/datasets/failure_prices/versions/1/partitions",
            {"timestamps": ["2026-01-05T00:00:00Z"]},
        )
        _wait_for_job(deployed_platform, api_url, _job_id(dataset_request))

        assert (
            _api_request(
                api_url, "POST", "/features", _broken_feature_payload()
            )["name"]
            == "broken_feature"
        )
        feature_request = _api_request(
            api_url,
            "POST",
            "/features/broken_feature/versions/1/partitions",
            {"timestamps": ["2026-01-05T00:00:00Z"]},
        )
        job = _wait_for_failed_job(
            deployed_platform,
            api_url,
            _job_id(feature_request),
        )
        assert (
            _string_field(_object_array_field(job, "partitions")[0], "status")
            == "failed"
        )
        feature = _api_request(
            api_url,
            "GET",
            "/features/broken_feature/versions/1",
            {},
        )
        assert (
            _string_field(
                _object_array_field(feature, "partitions")[0], "status"
            )
            == "failed"
        )


def test_cancelled_spark_application_marks_job_and_partition_cancelled(
    deployed_platform: DeployedPlatform,
) -> None:
    """A cancelled SparkApplication becomes durable cancelled API state."""
    dataset_payload = _dataset_payload()
    dataset_payload["name"] = "cancelled_prices"
    with deployed_platform.api_url() as api_url:
        assert (
            _api_request(api_url, "POST", "/datasets", dataset_payload)["name"]
            == "cancelled_prices"
        )
        dataset_request = _api_request(
            api_url,
            "POST",
            "/datasets/cancelled_prices/versions/1/partitions",
            {"timestamps": ["2026-01-05T00:00:00Z"]},
        )
        job_id = _job_id(dataset_request)
        application_name = f"dataset-{job_id}"
        _wait_for_spark_application(deployed_platform, application_name)
        deployed_platform.kubectl(
            "patch",
            "sparkapplication",
            application_name,
            "--namespace",
            "ml-platform",
            "--type",
            "merge",
            "--patch",
            '{"spec":{"suspend":true}}',
        )
        deployed_platform.kubectl(
            "patch",
            "sparkapplication",
            application_name,
            "--namespace",
            "ml-platform",
            "--subresource",
            "status",
            "--type",
            "merge",
            "--patch",
            '{"status":{"applicationState":{"state":"CANCELLED"}}}',
        )
        job = _wait_for_cancelled_job(api_url, job_id)
        assert (
            _string_field(_object_array_field(job, "partitions")[0], "status")
            == "cancelled"
        )
        dataset = _api_request(
            api_url,
            "GET",
            "/datasets/cancelled_prices/versions/1",
            {},
        )
        assert (
            _string_field(
                _object_array_field(dataset, "partitions")[0], "status"
            )
            == "cancelled"
        )
