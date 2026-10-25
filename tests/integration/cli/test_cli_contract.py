"""Black-box HTTP contract tests for the installed ML Platform CLI."""

import json
import os
import subprocess
import sys
from collections import deque
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

REPOSITORY_ROOT = Path(__file__).parents[3]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type JsonObject = dict[str, JsonValue]


@dataclass(frozen=True)
class PlannedResponse:
    """One response emitted by the local control-plane contract fixture.

    Attributes:
        status: HTTP status code returned to the CLI.
        body: JSON document returned to the CLI.
    """

    status: int
    body: JsonValue


@dataclass(frozen=True)
class RecordedRequest:
    """One HTTP request emitted by the CLI under test.

    Attributes:
        method: HTTP method used by the CLI.
        path: Requested route and query string.
        body: JSON request body, when present.
    """

    method: str
    path: str
    body: JsonObject | None


class ScriptedControlPlane:
    """A local HTTP server with a deterministic response sequence."""

    def __init__(self, responses: Iterable[PlannedResponse] = ()) -> None:
        self._responses = deque(responses)
        self.requests: list[RecordedRequest] = []
        self._server = ThreadingHTTPServer(
            ("127.0.0.1", 0), _request_handler(self)
        )
        self._thread = Thread(
            target=self._server.serve_forever,
            daemon=True,
        )

    @property
    def url(self) -> str:
        """Return this fixture's base URL."""
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def add(self, *responses: PlannedResponse) -> None:
        """Append deterministic API responses for the next requests."""
        self._responses.extend(responses)

    def start(self) -> None:
        """Start serving requests in a background thread."""
        self._thread.start()

    def close(self) -> None:
        """Stop serving and release this fixture's local port."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def handle(self, handler: BaseHTTPRequestHandler) -> None:
        """Record a request and return its next planned response."""
        content_length = int(handler.headers.get("Content-Length", "0"))
        document = handler.rfile.read(content_length)
        body = self._body(document)
        self.requests.append(
            RecordedRequest(
                method=handler.command,
                path=handler.path,
                body=body,
            )
        )
        if not self._responses:
            response = PlannedResponse(
                status=500,
                body={"detail": "Unexpected CLI request."},
            )
        else:
            response = self._responses.popleft()
        encoded_body = json.dumps(response.body).encode()
        handler.send_response(response.status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(encoded_body)))
        handler.end_headers()
        handler.wfile.write(encoded_body)

    @staticmethod
    def _body(document: bytes) -> JsonObject | None:
        """Decode an optional JSON object request body."""
        if not document:
            return None
        decoded = json.loads(document)
        assert isinstance(decoded, dict)
        return decoded


def _request_handler(
    control_plane: ScriptedControlPlane,
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to one scripted control-plane fixture."""

    class ControlPlaneHandler(BaseHTTPRequestHandler):
        """Forward every supported request to the scripted control plane."""

        def do_DELETE(self) -> None:
            control_plane.handle(self)

        def do_GET(self) -> None:
            control_plane.handle(self)

        def do_PATCH(self) -> None:
            control_plane.handle(self)

        def do_POST(self) -> None:
            control_plane.handle(self)

        def log_message(self, message_format: str, *arguments: object) -> None:
            """Suppress fixture request logs during normal test execution."""

    return ControlPlaneHandler


@pytest.fixture
def control_plane() -> Generator[ScriptedControlPlane, None, None]:
    """Start one scripted HTTP API fixture for a single CLI contract test."""
    fixture = ScriptedControlPlane()
    fixture.start()
    try:
        yield fixture
    finally:
        fixture.close()


def _run_cli(
    control_plane: ScriptedControlPlane,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    """Invoke the public module entrypoint against a local HTTP fixture."""
    environment = os.environ | {
        "PYTHONPATH": (
            f"{SOURCE_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
        )
    }
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_platform.cli",
            "--api-url",
            control_plane.url,
            *arguments,
        ],
        check=False,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )


def _write_config(tmp_path: Path, filename: str, source: str) -> Path:
    """Write one user-facing YAML definition and return its absolute path."""
    path = tmp_path / filename
    path.write_text(source)
    return path


@pytest.mark.parametrize(
    ("resource", "source", "arguments", "path", "expected_body"),
    [
        (
            "dataset",
            """name: prices
version: 1
dataset_type: market_data
description: Hourly prices.
partition_type: daily
keys: [symbol]
symbols: [AAPL]
""",
            ("--name", "prices", "--version", "1"),
            "/datasets",
            {
                "name": "prices",
                "version": 1,
                "dataset_type": "market_data",
                "config": {
                    "description": "Hourly prices.",
                    "validations": [],
                    "partition_type": "daily",
                    "dataset_type": "market_data",
                    "keys": ["symbol"],
                    "symbols": ["AAPL"],
                },
            },
        ),
        (
            "feature",
            """name: returns
version: 1
dtype: double
frequency: daily
keys: [symbol]
datasets:
  - name: prices
    version: 1
    input_window:
      lookback_partitions: 1
transform: ml_platform.data_sdk.LogReturnTransform
description: Daily returns.
""",
            (),
            "/features",
            {
                "name": "returns",
                "version": 1,
                "dtype": "double",
                "frequency": "daily",
                "keys": ["symbol"],
                "datasets": [
                    {
                        "name": "prices",
                        "version": 1,
                        "input_window": {"lookback_partitions": 1},
                    }
                ],
                "transform": "ml_platform.data_sdk.LogReturnTransform",
                "description": "Daily returns.",
                "parameters": {},
                "validations": [],
            },
        ),
        (
            "label",
            """name: direction
version: 1
dtype: int
frequency: daily
keys: [symbol]
datasets:
  - name: prices
    version: 1
    input_window:
      lookback_partitions: 1
      lookahead_partitions: 1
transform: ml_platform.data_sdk.TripleBarrierLabelTransform
description: Direction labels.
parameters:
  horizon_bars: 24
  volatility_lookback_bars: 24
  profit_taking_multiplier: 1.0
  stop_loss_multiplier: 1.0
""",
            (),
            "/labels",
            {
                "name": "direction",
                "version": 1,
                "dtype": "int",
                "frequency": "daily",
                "keys": ["symbol"],
                "datasets": [
                    {
                        "name": "prices",
                        "version": 1,
                        "input_window": {
                            "lookback_partitions": 1,
                            "lookahead_partitions": 1,
                        },
                    }
                ],
                "transform": "ml_platform.data_sdk.TripleBarrierLabelTransform",
                "description": "Direction labels.",
                "parameters": {
                    "horizon_bars": 24,
                    "volatility_lookback_bars": 24,
                    "profit_taking_multiplier": 1.0,
                    "stop_loss_multiplier": 1.0,
                },
                "validations": [],
            },
        ),
    ],
)
def test_create_posts_each_resource_definition(
    control_plane: ScriptedControlPlane,
    tmp_path: Path,
    resource: str,
    source: str,
    arguments: tuple[str, ...],
    path: str,
    expected_body: JsonObject,
) -> None:
    """Create commands send the public resource DTO for every resource type."""
    config = _write_config(tmp_path, f"{resource}.yaml", source)
    control_plane.add(
        PlannedResponse(status=201, body={"id": f"{resource}_v1"})
    )

    result = _run_cli(
        control_plane,
        "create",
        resource,
        "--config",
        str(config),
        *arguments,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout) == {"id": f"{resource}_v1"}
    assert control_plane.requests == [
        RecordedRequest(method="POST", path=path, body=expected_body)
    ]


def test_dataset_create_rejects_mismatched_identity(
    control_plane: ScriptedControlPlane, tmp_path: Path
) -> None:
    """Dataset identity flags must agree with the declared YAML definition."""
    config = _write_config(
        tmp_path,
        "prices.yaml",
        """name: prices
version: 1
dataset_type: market_data
description: Hourly prices.
partition_type: daily
keys: [symbol]
symbols: [AAPL]
""",
    )

    result = _run_cli(
        control_plane,
        "create",
        "dataset",
        "--config",
        str(config),
        "--name",
        "other_prices",
        "--version",
        "1",
    )

    assert result.returncode == 1
    assert "must match the config identity" in result.stderr
    assert control_plane.requests == []


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("- not\n- a mapping\n", "Configuration must be a mapping"),
        (
            """name: prices
version: 1
dataset_type: market_data
description: Hourly prices.
partition_type: daily
keys: [symbol]
symbols: []
""",
            "Invalid dataset configuration",
        ),
    ],
)
def test_create_rejects_invalid_yaml_configuration(
    control_plane: ScriptedControlPlane,
    tmp_path: Path,
    source: str,
    message: str,
) -> None:
    """Malformed and schema-invalid YAML fail before the CLI sends HTTP."""
    config = _write_config(tmp_path, "invalid.yaml", source)

    result = _run_cli(
        control_plane,
        "create",
        "dataset",
        "--config",
        str(config),
        "--name",
        "prices",
        "--version",
        "1",
    )

    assert result.returncode == 1
    assert message in result.stderr
    assert control_plane.requests == []


@pytest.mark.parametrize("status", [409, 500])
def test_materialize_reports_http_errors(
    control_plane: ScriptedControlPlane, status: int
) -> None:
    """Client and server failures produce one nonzero CLI result with context."""
    control_plane.add(
        PlannedResponse(status=status, body={"detail": "blocked"})
    )

    result = _run_cli(
        control_plane,
        "materialize",
        "dataset",
        "--name",
        "prices",
        "--version",
        "1",
        "--timestamp",
        "2026-01-01T00:00:00Z",
    )

    assert result.returncode == 1
    assert (
        "API POST /datasets/prices/versions/1/partitions failed"
        in result.stderr
    )


@pytest.mark.parametrize(
    ("resource", "path"),
    [
        ("dataset", "/datasets/prices/versions/1/partitions"),
        ("feature", "/features/prices/versions/1/partitions"),
        ("label", "/labels/prices/versions/1/partitions"),
    ],
)
def test_materialize_posts_all_requested_timestamps(
    control_plane: ScriptedControlPlane, resource: str, path: str
) -> None:
    """Each resource kind submits its selected partition timestamps unchanged."""
    control_plane.add(PlannedResponse(status=202, body={"submitted_job": None}))

    result = _run_cli(
        control_plane,
        "materialize",
        resource,
        "--name",
        "prices",
        "--version",
        "1",
        "--timestamp",
        "2026-01-01T00:00:00Z",
        "--timestamp",
        "2026-01-02T00:00:00Z",
    )

    assert result.returncode == 0
    assert control_plane.requests == [
        RecordedRequest(
            method="POST",
            path=path,
            body={
                "timestamps": [
                    "2026-01-01T00:00:00Z",
                    "2026-01-02T00:00:00Z",
                ]
            },
        )
    ]


def test_materialize_monitor_polls_until_succeeded(
    control_plane: ScriptedControlPlane,
) -> None:
    """Monitoring stops immediately after the first successful terminal result."""
    control_plane.add(
        PlannedResponse(
            status=202,
            body={"submitted_job": {"id": "job-1"}},
        ),
        PlannedResponse(
            status=200, body={"id": "job-1", "status": "submitted"}
        ),
        PlannedResponse(
            status=200, body={"id": "job-1", "status": "succeeded"}
        ),
    )

    result = _run_cli(
        control_plane,
        "materialize",
        "dataset",
        "--name",
        "prices",
        "--version",
        "1",
        "--timestamp",
        "2026-01-01T00:00:00Z",
        "--monitor",
        "--poll-seconds",
        "0",
    )

    assert result.returncode == 0
    assert [request.path for request in control_plane.requests] == [
        "/datasets/prices/versions/1/partitions",
        "/jobs/job-1",
        "/jobs/job-1",
    ]
    assert [
        json.loads(line)["status"] for line in result.stdout.splitlines()[1:]
    ] == [
        "submitted",
        "succeeded",
    ]


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_materialize_monitor_fails_for_unsuccessful_terminal_jobs(
    control_plane: ScriptedControlPlane, status: str
) -> None:
    """Monitoring converts each unsuccessful terminal job state into failure."""
    control_plane.add(
        PlannedResponse(
            status=202,
            body={"submitted_job": {"id": "job-1"}},
        ),
        PlannedResponse(status=200, body={"id": "job-1", "status": status}),
    )

    result = _run_cli(
        control_plane,
        "materialize",
        "dataset",
        "--name",
        "prices",
        "--version",
        "1",
        "--timestamp",
        "2026-01-01T00:00:00Z",
        "--monitor",
        "--poll-seconds",
        "0",
    )

    assert result.returncode == 1
    assert f"Materialization job job-1 {status}." in result.stderr


def test_describe_prints_job_and_reports_missing_job(
    control_plane: ScriptedControlPlane,
) -> None:
    """Describe prints a valid job and exposes a missing-job HTTP error."""
    control_plane.add(
        PlannedResponse(status=200, body={"id": "job-1", "status": "succeeded"})
    )

    described = _run_cli(control_plane, "describe", "job", "job-1")

    assert described.returncode == 0
    assert json.loads(described.stdout) == {
        "id": "job-1",
        "status": "succeeded",
    }
    control_plane.add(PlannedResponse(status=404, body={"detail": "missing"}))

    missing = _run_cli(control_plane, "describe", "job", "job-missing")

    assert missing.returncode == 1
    assert "API GET /jobs/job-missing failed" in missing.stderr
