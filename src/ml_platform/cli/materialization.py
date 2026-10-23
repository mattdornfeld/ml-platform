"""Submit, synchronize, and monitor ML Platform resources through the API."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, assert_never
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from omegaconf import OmegaConf

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
)
from ml_platform.common.domain import (
    FeatureConfig,
    LabelConfig,
    MarketDataConfig,
)

DEFAULT_CONFIG_DIR = Path("configs")


def _api_request(
    base_url: str,
    method: HttpMethod,
    path: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    """Make one JSON request to the control-plane API."""
    body = (
        None if payload is None else json.dumps(payload, default=str).encode()
    )
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            content = response.read()
            decoded = json.loads(content) if content else {}
            if not isinstance(decoded, dict):
                raise RuntimeError(
                    f"API {method} {path} did not return a JSON object."
                )
            return {str(key): value for key, value in decoded.items()}
    except HTTPError as error:
        detail = error.read().decode()
        raise RuntimeError(
            f"API {method} {path} failed with HTTP {error.code}: {detail}"
        ) from error


def _load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one resolved YAML configuration into ordinary JSON-compatible values."""
    loaded = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return {str(key): value for key, value in loaded.items()}


def _timestamps(values: list[str]) -> dict[str, list[str]]:
    """Build the shared API partition request body."""
    if not values:
        raise ValueError("Provide at least one --timestamp.")
    return {"timestamps": values}


def _describe_job(
    base_url: str, job_id: str, poll_seconds: float | None = None
) -> dict[str, Any]:
    """Fetch a job once, or poll it until the control plane reaches a terminal state."""
    while True:
        result = _api_request(
            base_url,
            JobRoutes.DETAIL.method,
            JobRoutes.DETAIL.path(job_id=job_id),
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"Job {job_id} response must be a JSON object.")
        sys.stdout.write(json.dumps(result, default=str) + "\n")
        if poll_seconds is None or result["status"] in {
            "succeeded",
            "failed",
            "cancelled",
        }:
            return result
        time.sleep(poll_seconds)


def _dataset_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Validate one dataset YAML mapping and convert it to the API request shape."""
    definition = MarketDataConfig.model_validate_json(
        json.dumps(config, default=str)
    )
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


def _dataset_configs(config_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load, validate, and serialize every checked-in dataset definition."""
    resources: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted((config_dir / "datasets").glob("*.yaml")):
        payload = _dataset_payload(_load_yaml(path))
        identity = (str(payload["name"]), int(payload["version"]))
        if identity in resources:
            raise ValueError(
                f"Duplicate dataset config identity: {identity[0]} v{identity[1]}"
            )
        resources[identity] = payload
    return resources


def _feature_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Validate one feature YAML mapping and convert it to the API request shape."""
    values = FeatureConfig.model_validate_json(
        json.dumps(config, default=str)
    ).model_dump(mode="json")
    return FeatureCreateRequest.model_validate(values).model_dump(mode="json")


def _feature_configs(config_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load and validate feature YAML definitions for API registration."""
    resources: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted((config_dir / "features").glob("*.yaml")):
        payload = _feature_payload(_load_yaml(path))
        identity = (str(payload["name"]), int(payload["version"]))
        if identity in resources:
            raise ValueError(
                f"Duplicate feature config identity: {identity[0]} v{identity[1]}"
            )
        resources[identity] = payload
    return resources


def _label_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Validate one label YAML mapping and convert it to the API request shape."""
    values = LabelConfig.model_validate_json(
        json.dumps(config, default=str)
    ).model_dump(mode="json")
    return LabelCreateRequest.model_validate(values).model_dump(mode="json")


def _label_configs(config_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load and validate label YAML definitions for API registration."""
    resources: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted((config_dir / "labels").glob("*.yaml")):
        payload = _label_payload(_load_yaml(path))
        identity = (str(payload["name"]), int(payload["version"]))
        if identity in resources:
            raise ValueError(
                f"Duplicate label config identity: {identity[0]} v{identity[1]}"
            )
        resources[identity] = payload
    return resources


def sync_resources(
    base_url: str, resource: ResourceType, config_dir: Path = DEFAULT_CONFIG_DIR
) -> dict[str, list[str]]:
    """Reconcile API resource definitions with checked-in configs by soft-deleting removed identities."""
    match resource:
        case ResourceType.DATASET:
            desired = _dataset_configs(config_dir)
            list_route = DatasetRoutes.LIST
            create_route = DatasetRoutes.CREATE
            delete_route = DatasetRoutes.DELETE
        case ResourceType.FEATURE:
            desired = _feature_configs(config_dir)
            list_route = FeatureRoutes.LIST
            create_route = FeatureRoutes.CREATE
            delete_route = FeatureRoutes.DELETE
        case ResourceType.LABEL:
            desired = _label_configs(config_dir)
            list_route = LabelRoutes.LIST
            create_route = LabelRoutes.CREATE
            delete_route = LabelRoutes.DELETE
        case _ as unsupported_resource:
            raise ValueError(f"Unsupported resource {unsupported_resource!r}.")
    existing = _api_request(
        base_url,
        list_route.method,
        (
            f"{list_route.template}?statuses=active&statuses=deprecated"
            "&statuses=deleted"
            if resource is ResourceType.DATASET
            else f"{list_route.template}?include_deleted=true"
        ),
    )
    active = {
        (row["name"], int(row["version"])): row
        for row in existing
        if row["deleted_at"] is None
    }
    created: list[str] = []
    updated: list[str] = []
    deleted: list[str] = []
    for identity, payload in desired.items():
        if identity not in active:
            _api_request(
                base_url, create_route.method, create_route.template, payload
            )
            created.append(f"{identity[0]} v{identity[1]}")
        elif resource is ResourceType.DATASET and active[identity].get(
            "config", {}
        ).get("description") != payload["config"].get("description"):
            _api_request(
                base_url,
                DatasetRoutes.UPDATE.method,
                DatasetRoutes.UPDATE.path(
                    name=identity[0], version=identity[1]
                ),
                {"description": payload["config"]["description"]},
            )
            updated.append(f"{identity[0]} v{identity[1]}")
    for name, version in active:
        if (name, version) not in desired:
            _api_request(
                base_url,
                delete_route.method,
                delete_route.path(name=name, version=version),
            )
            deleted.append(f"{name} v{version}")
    result = {"created": created, "deleted": deleted}
    if updated:
        result["updated"] = updated
    return result


def main() -> None:
    """Run the materialization API command-line client."""
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
    args = parser.parse_args()

    if args.command == "create":
        resource = ResourceType(args.resource)
        match resource:
            case ResourceType.DATASET:
                payload = _dataset_payload(_load_yaml(args.config))
                if (
                    payload["name"] != args.name
                    or payload["version"] != args.version
                ):
                    raise ValueError(
                        "Dataset --name and --version must match the config identity."
                    )
                result = _api_request(
                    args.api_url,
                    DatasetRoutes.CREATE.method,
                    DatasetRoutes.CREATE.template,
                    payload,
                )
            case ResourceType.FEATURE:
                result = _api_request(
                    args.api_url,
                    FeatureRoutes.CREATE.method,
                    FeatureRoutes.CREATE.template,
                    _feature_payload(_load_yaml(args.config)),
                )
            case ResourceType.LABEL:
                result = _api_request(
                    args.api_url,
                    LabelRoutes.CREATE.method,
                    LabelRoutes.CREATE.template,
                    _label_payload(_load_yaml(args.config)),
                )
            case _ as unsupported_resource:
                assert_never(unsupported_resource)
    elif args.command == "materialize":
        resource = ResourceType(args.resource)
        match resource:
            case ResourceType.DATASET:
                route = DatasetRoutes.MATERIALIZE
            case ResourceType.FEATURE:
                route = FeatureRoutes.MATERIALIZE
            case ResourceType.LABEL:
                route = LabelRoutes.MATERIALIZE
            case _ as unsupported_resource:
                assert_never(unsupported_resource)
        result = _api_request(
            args.api_url,
            route.method,
            route.path(name=args.name, version=args.version),
            _timestamps(args.timestamp),
        )
    elif args.command in {"datasets", "features", "labels"}:
        result = sync_resources(
            args.api_url, args.sync_resource, args.config_dir
        )
    else:
        _describe_job(args.api_url, args.job_id)
        return

    sys.stdout.write(json.dumps(result, default=str) + "\n")
    if (
        args.command == "materialize"
        and args.monitor
        and result["submitted_job"] is not None
    ):
        _describe_job(
            args.api_url, result["submitted_job"]["id"], args.poll_seconds
        )


if __name__ == "__main__":
    main()
