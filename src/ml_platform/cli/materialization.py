"""Submit, synchronize, and monitor ML Platform resources through the API."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from omegaconf import OmegaConf


DEFAULT_CONFIG_DIR = Path("configs")


def _api_request(base_url: str, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    """Make one JSON request to the control-plane API."""
    body = None if payload is None else json.dumps(payload, default=str).encode()
    request = Request(f"{base_url.rstrip('/')}{path}", data=body, method=method, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            content = response.read()
            return json.loads(content) if content else {}
    except HTTPError as error:
        detail = error.read().decode()
        raise RuntimeError(f"API {method} {path} failed with HTTP {error.code}: {detail}") from error


def _load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one resolved YAML configuration into ordinary JSON-compatible values."""
    loaded = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return loaded


def _timestamps(values: list[str]) -> dict[str, list[str]]:
    """Build the shared API partition request body."""
    if not values:
        raise ValueError("Provide at least one --timestamp.")
    return {"timestamps": values}


def _describe_job(base_url: str, job_id: str, poll_seconds: float | None = None) -> dict[str, Any]:
    """Fetch a job once, or poll it until the control plane reaches a terminal state."""
    while True:
        result = _api_request(base_url, "GET", f"/jobs/{job_id}")
        print(json.dumps(result, default=str))
        if poll_seconds is None or result["status"] in {"succeeded", "failed", "cancelled"}:
            return result
        time.sleep(poll_seconds)


def _dataset_configs(config_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load every checked-in dataset config into its API create payload."""
    resources: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted((config_dir / "datasets").glob("*.yaml")):
        config = _load_yaml(path)
        name = config.pop("name")
        version = config.pop("version")
        config.pop("dataset_id")
        identity = (str(name), int(version))
        if identity in resources:
            raise ValueError(f"Duplicate dataset config identity: {identity[0]} v{identity[1]}")
        resources[identity] = {"name": name, "version": version, "dataset_type": config.pop("dataset_type"), "config": config}
    return resources


def _feature_configs(config_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load feature configs and resolve logical dataset IDs to API dataset identities."""
    dataset_ids: dict[str, tuple[str, int]] = {}
    for path in sorted((config_dir / "datasets").glob("*.yaml")):
        config = _load_yaml(path)
        dataset_id = str(config["dataset_id"])
        if dataset_id in dataset_ids:
            raise ValueError(f"Duplicate dataset_id in configs: {dataset_id}")
        dataset_ids[dataset_id] = (str(config["name"]), int(config["version"]))
    resources: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted((config_dir / "features").glob("*.yaml")):
        config = _load_yaml(path)
        config.pop("metadata", None)
        dependencies = []
        for dependency in config["datasets"]:
            dataset_id = str(dependency["dataset_id"])
            if dataset_id not in dataset_ids:
                raise ValueError(f"Feature {config['name']} references unknown dataset_id: {dataset_id}")
            name, version = dataset_ids[dataset_id]
            dependencies.append({"name": name, "version": version, "num_lookback_partitions": dependency["num_lookback_partitions"]})
        config["datasets"] = dependencies
        identity = (str(config["name"]), int(config["version"]))
        if identity in resources:
            raise ValueError(f"Duplicate feature config identity: {identity[0]} v{identity[1]}")
        resources[identity] = config
    return resources


def sync_resources(base_url: str, resource: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> dict[str, list[str]]:
    """Reconcile API resource definitions with checked-in configs by soft-deleting removed identities."""
    desired = _dataset_configs(config_dir) if resource == "dataset" else _feature_configs(config_dir)
    existing = _api_request(base_url, "GET", f"/{resource}s?include_deleted=true")
    active = {(row["name"], int(row["version"])): row for row in existing if row["deleted_at"] is None}
    created: list[str] = []
    updated: list[str] = []
    deleted: list[str] = []
    for identity, payload in desired.items():
        if identity not in active:
            _api_request(base_url, "POST", f"/{resource}s", payload)
            created.append(f"{identity[0]} v{identity[1]}")
        elif resource == "dataset" and active[identity].get("config", {}).get("description") != payload["config"].get("description"):
            _api_request(base_url, "PATCH", f"/datasets/{identity[0]}/versions/{identity[1]}", {"description": payload["config"]["description"]})
            updated.append(f"{identity[0]} v{identity[1]}")
    for name, version in active:
        if (name, version) not in desired:
            _api_request(base_url, "DELETE", f"/{resource}s/{name}/versions/{version}")
            deleted.append(f"{name} v{version}")
    result = {"created": created, "deleted": deleted}
    if updated:
        result["updated"] = updated
    return result


def main() -> None:
    """Run the materialization API command-line client."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=os.environ.get("MATERIALIZATION_API_URL", "http://127.0.0.1:8000"))
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create_resources = create.add_subparsers(dest="resource", required=True)
    for resource in ("dataset", "feature"):
        resource_parser = create_resources.add_parser(resource)
        resource_parser.add_argument("--config", required=True)
        if resource == "dataset":
            resource_parser.add_argument("--name", required=True)
            resource_parser.add_argument("--version", type=int, required=True)
    materialize = commands.add_parser("materialize")
    materialize_resources = materialize.add_subparsers(dest="resource", required=True)
    for resource in ("dataset", "feature"):
        resource_parser = materialize_resources.add_parser(resource)
        resource_parser.add_argument("--name", required=True)
        resource_parser.add_argument("--version", type=int, required=True)
        resource_parser.add_argument("--timestamp", action="append", default=[])
        resource_parser.add_argument("--monitor", action="store_true", help="Poll the created job until it reaches a terminal state.")
        resource_parser.add_argument("--poll-seconds", type=float, default=5, help="Seconds between --monitor status requests.")
    describe = commands.add_parser("describe")
    describe_resources = describe.add_subparsers(dest="resource", required=True)
    job = describe_resources.add_parser("job")
    job.add_argument("job_id")
    for plural, resource in (("datasets", "dataset"), ("features", "feature")):
        resource_commands = commands.add_parser(plural)
        resource_actions = resource_commands.add_subparsers(dest="action", required=True)
        sync = resource_actions.add_parser("sync", help="Create checked-in definitions and soft-delete removed definitions.")
        sync.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
        sync.set_defaults(sync_resource=resource)
    args = parser.parse_args()

    if args.command == "create" and args.resource == "dataset":
        config = _load_yaml(args.config)
        result = _api_request(args.api_url, "POST", "/datasets", {"name": args.name, "version": args.version, **config})
    elif args.command == "create" and args.resource == "feature":
        config = _load_yaml(args.config)
        config.pop("metadata", None)
        result = _api_request(args.api_url, "POST", "/features", config)
    elif args.command == "materialize":
        result = _api_request(args.api_url, "POST", f"/{args.resource}s/{args.name}/versions/{args.version}/partitions", _timestamps(args.timestamp))
    elif args.command in {"datasets", "features"}:
        result = sync_resources(args.api_url, args.sync_resource, args.config_dir)
    else:
        _describe_job(args.api_url, args.job_id)
        return
    print(json.dumps(result, default=str))
    if args.command == "materialize" and args.monitor and result["job"] is not None:
        _describe_job(args.api_url, result["job"]["id"], args.poll_seconds)


if __name__ == "__main__":
    main()
