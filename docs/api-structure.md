# API project structure

`ml_platform.common.api` owns the public contract: typed DTOs and route
specifications. It has no dependency on FastAPI, PostgreSQL, or the data SDK,
so the CLI and workers can import the same contract.

`ml_platform.api.app` composes the FastAPI application, installs its lifecycle,
and includes every router. It is the ASGI and console-script entry point.

`ml_platform.api.dependencies` defines the typed FastAPI dependencies shared by
routers. `ml_platform.api.routers` owns HTTP concerns: endpoint declarations,
status codes, OpenAPI examples, and request/response DTO conversion. Routers
are grouped by datasets, features, labels, partitions, jobs, and the dashboard.

`ml_platform.api.repositories` contains reusable PostgreSQL queries. It does not
know about FastAPI. `ml_platform.api.services` coordinates reusable application
behavior that is not persistence, including MinIO and read-only Headlamp links.

`ml_platform.api.models` holds SQLAlchemy table mappings; `database.py` owns
connection/session construction; `migrations/` contains ordered schema changes;
and `scheduler.py` remains the authoritative job scheduler.
