"""FastAPI application composition for the ML Platform control plane."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ml_platform.api._configuration import Dependencies
from ml_platform.api._dependencies import get_dependencies
from ml_platform.api.infrastructure import upgrade_database
from ml_platform.api.routers import (
    datasets_router,
    features_router,
    jobs_router,
    labels_router,
    mount_ui,
    partitions_router,
    ui_router,
)
from ml_platform.api.services import JobScheduler


def create_app(dependencies: Dependencies) -> FastAPI:
    """Compose one API application around one validated dependency instance."""

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
        """Populate PostgreSQL then run the authoritative scheduler."""
        upgrade_database(dependencies)
        stop = asyncio.Event()
        scheduler = JobScheduler(dependencies)
        scheduler_task = asyncio.create_task(scheduler.run(stop))
        try:
            yield
        finally:
            stop.set()
            await scheduler_task

    application = FastAPI(
        title="ML Platform Materialization API",
        version="0.1.0",
        summary="Control plane for Iceberg-backed dataset and feature materialization.",
        lifespan=lifespan,
    )

    def injected_dependencies() -> Dependencies:
        """Return the one immutable dependency context for this application."""
        return dependencies

    application.dependency_overrides[get_dependencies] = injected_dependencies
    mount_ui(application)
    for router in (
        ui_router,
        datasets_router,
        features_router,
        labels_router,
        partitions_router,
        jobs_router,
    ):
        application.include_router(router)
    return application


app = create_app(Dependencies.from_environment())


def main() -> None:
    """Run the application with Uvicorn for local development."""
    import uvicorn

    uvicorn.run(
        "ml_platform.api.app:app", host="127.0.0.1", port=8000, reload=True
    )
