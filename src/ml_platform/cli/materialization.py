"""Submit immutable resources and monitor materialization jobs through the API."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from omegaconf import OmegaConf


def _api_request(base_url: str, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Make one JSON request to the control-plane API."""
    body = None if payload is None else json.dumps(payload, default=str).encode()
    request = Request(f"{base_url.rstrip('/')}{path}", data=body, method=method, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except HTTPError as error:
        detail = error.read().decode()
        raise RuntimeError(f"API {method} {path} failed with HTTP {error.code}: {detail}") from error


def _load_yaml(path: str) -> dict[str, Any]:
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
    else:
        _describe_job(args.api_url, args.job_id)
        return
    print(json.dumps(result, default=str))
    if args.command == "materialize" and args.monitor and result["job"] is not None:
        _describe_job(args.api_url, result["job"]["id"], args.poll_seconds)


if __name__ == "__main__":
    main()
