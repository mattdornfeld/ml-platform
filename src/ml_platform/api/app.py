"""FastAPI application composition for the ML Platform control plane."""

import asyncio
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

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


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Populate PostgreSQL then run the authoritative materialization scheduler."""
    upgrade_database()
    stop = asyncio.Event()
    scheduler = JobScheduler(os.environ["DATABASE_URL"])
    scheduler_task = asyncio.create_task(scheduler.run(stop))
    try:
        yield
    finally:
        stop.set()
        await scheduler_task


app = FastAPI(
    title="ML Platform Materialization API",
    version="0.1.0",
    summary="Control plane for Iceberg-backed dataset and feature materialization.",
    lifespan=lifespan,
)
mount_ui(app)
for router in (
    ui_router,
    datasets_router,
    features_router,
    labels_router,
    partitions_router,
    jobs_router,
):
    app.include_router(router)


def main() -> None:
    """Run the application with Uvicorn for local development."""
    import uvicorn

    uvicorn.run(
        "ml_platform.api.app:app", host="127.0.0.1", port=8000, reload=True
    )
