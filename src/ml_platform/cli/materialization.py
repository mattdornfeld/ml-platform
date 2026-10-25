"""Submit, synchronize, and monitor ML Platform resources through the API."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import assert_never

import httpx2
from omegaconf import OmegaConf
from pydantic import ValidationError

from ml_platform.common.api.dtos import (
    DatasetCreateRequest,
    FeatureCreateRequest,
    LabelCreateRequest,
    ResourceType,
)
from ml_platform.common.api.routes import (
    DatasetRoutes,
    FeatureRoutes,
    HttpMethod,
    JobRoutes,
    LabelRoutes,
    Route,
)
from ml_platform.common.domain import (
    FeatureConfig,
    LabelConfig,
    MarketDataConfig,
)

DEFAULT_CONFIG_DIR = Path("configs")


type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type JsonObject = dict[str, JsonValue]


class CliError(Exception):
    """Base error raised by the ML Platform command-line client."""


class CliRequestError(CliError):
    """Raised when a control-plane API request cannot be completed."""


class CliResponseError(CliError):
    """Raised when a control-plane API response has an unexpected JSON shape."""


class CliConfigurationError(CliError):
    """Raised when a local CLI configuration is invalid."""


class CliTerminalJobError(CliError):
    """Raised when a monitored materialization job reaches an error state."""


def _json_value(value: object) -> JsonValue:
    """Convert an external JSON value to the CLI's recursive JSON type."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise CliResponseError(
        f"Expected JSON-compatible value, received {type(value).__name__}."
    )


def _json_object(value: JsonValue, *, context: str) -> JsonObject:
    """Require a JSON object in a named response or configuration context."""
    if not isinstance(value, dict):
        raise CliResponseError(f"{context} must be a JSON object.")
    return value


def _json_objects(value: JsonValue, *, context: str) -> list[JsonObject]:
    """Require a list of JSON objects in a named response context."""
    if not isinstance(value, list):
        raise CliResponseError(f"{context} must be a JSON array.")
    return [_json_object(item, context=context) for item in value]


def _string(value: JsonValue, *, context: str) -> str:
    """Require a JSON string in a named response or configuration context."""
    if not isinstance(value, str):
        raise CliResponseError(f"{context} must be a string.")
    return value


def _integer(value: JsonValue, *, context: str) -> int:
    """Require a JSON integer in a named response or configuration context."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise CliResponseError(f"{context} must be an integer.")
    return value


def _api_request(
    base_url: str,
    method: HttpMethod,
    path: str,
    payload: JsonObject | None = None,
) -> JsonValue:
    """Make one JSON request to the control-plane API."""
    try:
        response = httpx2.request(
            method=method,
            url=f"{base_url.rstrip('/')}{path}",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
    except httpx2.HTTPError as error:
        raise CliRequestError(f"API {method} {path} failed: {error}") from error
    if not response.content:
        return {}
    return _json_value(response.json())


def _load_yaml(path: str | Path) -> JsonObject:
    """Load one resolved YAML configuration into ordinary JSON-compatible values."""
    loaded = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True)
    if not isinstance(loaded, dict):
        raise CliConfigurationError(f"Configuration must be a mapping: {path}")
    return _json_object(_json_value(loaded), context=f"Configuration {path}")


def _timestamps(values: list[str]) -> JsonObject:
    """Build the shared API partition request body."""
    if not values:
        raise CliConfigurationError("Provide at least one --timestamp.")
    timestamps: list[JsonValue] = [*values]
    return {"timestamps": timestamps}


def _describe_job(
    base_url: str, job_id: str, poll_seconds: float | None = None
) -> JsonObject:
    """Fetch a job once, or poll it until the control plane reaches a terminal state."""
    while True:
        result = _api_request(
            base_url,
            JobRoutes.DETAIL.method,
            JobRoutes.DETAIL.path(job_id=job_id),
        )
        result = _json_object(result, context=f"Job {job_id} response")
        sys.stdout.write(json.dumps(result, default=str) + "\n")
        if poll_seconds is None or result["status"] in {
            "succeeded",
            "failed",
            "cancelled",
        }:
            return result
        time.sleep(poll_seconds)


def _dataset_payload(config: JsonObject) -> JsonObject:
    """Validate one dataset YAML mapping and convert it to the API request shape."""
    try:
        definition = MarketDataConfig.model_validate_json(
            json.dumps(config, default=str)
        )
    except ValidationError as error:
        raise CliConfigurationError(
            f"Invalid dataset configuration: {error}"
        ) from error
    values = definition.model_dump(mode="json")
    values.pop("num_lookback_partitions")
    values.pop("run_frequency")
    return DatasetCreateRequest.model_validate(
        {
            "name": values.pop("name"),
            "version": values.pop("version"),
            "dataset_type": values.pop("dataset_type"),
            "config": values,
        }
    ).model_dump(mode="json", exclude_computed_fields=True)


def _dataset_configs(config_dir: Path) -> dict[tuple[str, int], JsonObject]:
    """Load, validate, and serialize every checked-in dataset definition."""
    resources: dict[tuple[str, int], JsonObject] = {}
    for path in sorted((config_dir / "datasets").glob("*.yaml")):
        payload = _dataset_payload(_load_yaml(path))
        identity = (
            _string(payload["name"], context="dataset config name"),
            _integer(payload["version"], context="dataset config version"),
        )
        if identity in resources:
            raise CliConfigurationError(
                f"Duplicate dataset config identity: {identity[0]} v{identity[1]}"
            )
        resources[identity] = payload
    return resources


def _feature_payload(config: JsonObject) -> JsonObject:
    """Validate one feature YAML mapping and convert it to the API request shape."""
    try:
        values = FeatureConfig.model_validate_json(
            json.dumps(config, default=str)
        ).model_dump(mode="json")
    except ValidationError as error:
        raise CliConfigurationError(
            f"Invalid feature configuration: {error}"
        ) from error
    return FeatureCreateRequest.model_validate(values).model_dump(mode="json")


def _feature_configs(config_dir: Path) -> dict[tuple[str, int], JsonObject]:
    """Load and validate feature YAML definitions for API registration."""
    resources: dict[tuple[str, int], JsonObject] = {}
    for path in sorted((config_dir / "features").glob("*.yaml")):
        payload = _feature_payload(_load_yaml(path))
        identity = (
            _string(payload["name"], context="feature config name"),
            _integer(payload["version"], context="feature config version"),
        )
        if identity in resources:
            raise CliConfigurationError(
                f"Duplicate feature config identity: {identity[0]} v{identity[1]}"
            )
        resources[identity] = payload
    return resources


def _label_payload(config: JsonObject) -> JsonObject:
    """Validate one label YAML mapping and convert it to the API request shape."""
    try:
        values = LabelConfig.model_validate_json(
            json.dumps(config, default=str)
        ).model_dump(mode="json")
    except ValidationError as error:
        raise CliConfigurationError(
            f"Invalid label configuration: {error}"
        ) from error
    return LabelCreateRequest.model_validate(values).model_dump(mode="json")


def _label_configs(config_dir: Path) -> dict[tuple[str, int], JsonObject]:
    """Load and validate label YAML definitions for API registration."""
    resources: dict[tuple[str, int], JsonObject] = {}
    for path in sorted((config_dir / "labels").glob("*.yaml")):
        payload = _label_payload(_load_yaml(path))
        identity = (
            _string(payload["name"], context="label config name"),
            _integer(payload["version"], context="label config version"),
        )
        if identity in resources:
            raise CliConfigurationError(
                f"Duplicate label config identity: {identity[0]} v{identity[1]}"
            )
        resources[identity] = payload
    return resources


def _resource_configs(
    resource: ResourceType, config_dir: Path
) -> dict[tuple[str, int], JsonObject]:
    """Load checked-in definitions for one resource kind."""
    match resource:
        case ResourceType.DATASET:
            return _dataset_configs(config_dir)
        case ResourceType.FEATURE:
            return _feature_configs(config_dir)
        case ResourceType.LABEL:
            return _label_configs(config_dir)
        case _ as unsupported_resource:
            assert_never(unsupported_resource)


def _resource_sync_routes(resource: ResourceType) -> tuple[Route, Route, Route]:
    """Return list, create, and delete routes for one resource kind."""
    match resource:
        case ResourceType.DATASET:
            return (
                DatasetRoutes.LIST,
                DatasetRoutes.CREATE,
                DatasetRoutes.DELETE,
            )
        case ResourceType.FEATURE:
            return (
                FeatureRoutes.LIST,
                FeatureRoutes.CREATE,
                FeatureRoutes.DELETE,
            )
        case ResourceType.LABEL:
            return LabelRoutes.LIST, LabelRoutes.CREATE, LabelRoutes.DELETE
        case _ as unsupported_resource:
            assert_never(unsupported_resource)


def _active_resources(
    base_url: str, resource: ResourceType, list_route: Route
) -> dict[tuple[str, int], JsonObject]:
    """Load the non-deleted API definitions for one resource kind."""
    query = (
        "?statuses=active&statuses=deprecated&statuses=deleted"
        if resource is ResourceType.DATASET
        else "?include_deleted=true"
    )
    existing = _json_objects(
        _api_request(
            base_url, list_route.method, f"{list_route.template}{query}"
        ),
        context=f"{resource.value} list response",
    )
    active: dict[tuple[str, int], JsonObject] = {}
    for row in existing:
        if row.get("deleted_at") is None:
            identity = (
                _string(row["name"], context="resource name"),
                _integer(row["version"], context="resource version"),
            )
            active[identity] = row
    return active


def _update_dataset_description(
    base_url: str,
    identity: tuple[str, int],
    existing: JsonObject,
    payload: JsonObject,
) -> bool:
    """Update an existing dataset when its configured description changed."""
    existing_config = _json_object(
        existing.get("config", {}), context="existing dataset config"
    )
    desired_config = _json_object(payload["config"], context="dataset config")
    if existing_config.get("description") == desired_config.get("description"):
        return False
    _api_request(
        base_url,
        DatasetRoutes.UPDATE.method,
        DatasetRoutes.UPDATE.path(name=identity[0], version=identity[1]),
        {
            "description": _string(
                desired_config["description"], context="dataset description"
            )
        },
    )
    return True


def sync_resources(
    base_url: str, resource: ResourceType, config_dir: Path = DEFAULT_CONFIG_DIR
) -> JsonObject:
    """Reconcile API definitions with checked-in configs and soft-delete removed ones."""
    desired = _resource_configs(resource, config_dir)
    list_route, create_route, delete_route = _resource_sync_routes(resource)
    active = _active_resources(base_url, resource, list_route)
    created: list[JsonValue] = []
    updated: list[JsonValue] = []
    deleted: list[JsonValue] = []
    for identity, payload in desired.items():
        if identity not in active:
            _api_request(
                base_url, create_route.method, create_route.template, payload
            )
            created.append(f"{identity[0]} v{identity[1]}")
        elif resource is ResourceType.DATASET and _update_dataset_description(
            base_url, identity, active[identity], payload
        ):
            updated.append(f"{identity[0]} v{identity[1]}")
    for name, version in active:
        if (name, version) not in desired:
            _api_request(
                base_url,
                delete_route.method,
                delete_route.path(name=name, version=version),
            )
            deleted.append(f"{name} v{version}")
    result: JsonObject = {"created": created, "deleted": deleted}
    if updated:
        result["updated"] = updated
    return result


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser for resource operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-url",
        default=os.environ.get(
            "MATERIALIZATION_API_URL", "http://127.0.0.1:8000"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create_resources = create.add_subparsers(dest="resource", required=True)
    for resource in ResourceType:
        resource_parser = create_resources.add_parser(resource.value)
        resource_parser.add_argument("--config", required=True)
        if resource is ResourceType.DATASET:
            resource_parser.add_argument("--name", required=True)
            resource_parser.add_argument("--version", type=int, required=True)
    materialize = commands.add_parser("materialize")
    materialize_resources = materialize.add_subparsers(
        dest="resource", required=True
    )
    for resource in ResourceType:
        resource_parser = materialize_resources.add_parser(resource.value)
        resource_parser.add_argument("--name", required=True)
        resource_parser.add_argument("--version", type=int, required=True)
        resource_parser.add_argument("--timestamp", action="append", default=[])
        resource_parser.add_argument(
            "--monitor",
            action="store_true",
            help="Poll the created job until it reaches a terminal state.",
        )
        resource_parser.add_argument(
            "--poll-seconds",
            type=float,
            default=5,
            help="Seconds between --monitor status requests.",
        )
    describe = commands.add_parser("describe")
    describe_resources = describe.add_subparsers(dest="resource", required=True)
    job = describe_resources.add_parser("job")
    job.add_argument("job_id")
    for plural, resource in (
        ("datasets", ResourceType.DATASET),
        ("features", ResourceType.FEATURE),
        ("labels", ResourceType.LABEL),
    ):
        resource_commands = commands.add_parser(plural)
        resource_actions = resource_commands.add_subparsers(
            dest="action", required=True
        )
        sync = resource_actions.add_parser(
            "sync",
            help="Create checked-in definitions and soft-delete removed definitions.",
        )
        sync.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
        sync.set_defaults(sync_resource=resource)
    return parser


def _create_resource(arguments: argparse.Namespace) -> JsonValue:
    """Create one declared resource through its public API route."""
    resource = ResourceType(arguments.resource)
    match resource:
        case ResourceType.DATASET:
            payload = _dataset_payload(_load_yaml(arguments.config))
            if (
                payload["name"] != arguments.name
                or payload["version"] != arguments.version
            ):
                raise CliConfigurationError(
                    "Dataset --name and --version must match the config identity."
                )
            return _api_request(
                arguments.api_url,
                DatasetRoutes.CREATE.method,
                DatasetRoutes.CREATE.template,
                payload,
            )
        case ResourceType.FEATURE:
            return _api_request(
                arguments.api_url,
                FeatureRoutes.CREATE.method,
                FeatureRoutes.CREATE.template,
                _feature_payload(_load_yaml(arguments.config)),
            )
        case ResourceType.LABEL:
            return _api_request(
                arguments.api_url,
                LabelRoutes.CREATE.method,
                LabelRoutes.CREATE.template,
                _label_payload(_load_yaml(arguments.config)),
            )
        case _ as unsupported_resource:
            assert_never(unsupported_resource)


def _materialize_resource(arguments: argparse.Namespace) -> JsonValue:
    """Submit one resource partition request through its public API route."""
    resource = ResourceType(arguments.resource)
    match resource:
        case ResourceType.DATASET:
            route = DatasetRoutes.MATERIALIZE
        case ResourceType.FEATURE:
            route = FeatureRoutes.MATERIALIZE
        case ResourceType.LABEL:
            route = LabelRoutes.MATERIALIZE
        case _ as unsupported_resource:
            assert_never(unsupported_resource)
    return _api_request(
        arguments.api_url,
        route.method,
        route.path(name=arguments.name, version=arguments.version),
        _timestamps(arguments.timestamp),
    )


def _job_id(result: JsonValue) -> str | None:
    """Extract the optional submitted job identifier from one response."""
    response = _json_object(result, context="Materialization response")
    submitted_job = response.get("submitted_job")
    if submitted_job is None:
        return None
    job = _json_object(submitted_job, context="submitted_job")
    job_id = job.get("id")
    if not isinstance(job_id, str):
        raise CliResponseError("submitted_job.id must be a string.")
    return job_id


def main() -> None:
    """Run the materialization API command-line client."""
    try:
        arguments = _parser().parse_args()
        match arguments.command:
            case "create":
                result = _create_resource(arguments)
            case "materialize":
                result = _materialize_resource(arguments)
            case "datasets" | "features" | "labels":
                result = sync_resources(
                    arguments.api_url,
                    arguments.sync_resource,
                    arguments.config_dir,
                )
            case "describe":
                _describe_job(arguments.api_url, arguments.job_id)
                return
            case unsupported_command:
                raise CliConfigurationError(
                    f"Unsupported command {unsupported_command!r}."
                )
        sys.stdout.write(json.dumps(result, default=str) + "\n")
        if arguments.command == "materialize" and arguments.monitor:
            submitted_job_id = _job_id(result)
            if submitted_job_id is not None:
                job = _describe_job(
                    arguments.api_url,
                    submitted_job_id,
                    arguments.poll_seconds,
                )
                status = _string(job["status"], context="Job status")
                if status in {"failed", "cancelled"}:
                    raise CliTerminalJobError(
                        f"Materialization job {submitted_job_id} {status}."
                    )
    except CliError as error:
        sys.stderr.write(f"{error}\n")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
