"""Static single-page dashboard routes."""

from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

router = APIRouter(include_in_schema=False)
UI_DIRECTORY = Path(__file__).parents[1] / "ui"


def mount_ui(application: FastAPI) -> None:
    """Attach static dashboard assets to the FastAPI application."""
    application.mount(
        "/ui/assets", StaticFiles(directory=UI_DIRECTORY), name="ui-assets"
    )


@router.get("/")
def dashboard() -> RedirectResponse:
    """Open the control-plane dashboard."""
    return RedirectResponse(url="/ui")


@router.get("/ui")
@router.get("/ui/{dashboard_path:path}")
def dashboard_ui(dashboard_path: str = "") -> FileResponse:
    """Serve the single-page dashboard at every resource browser path."""
    if dashboard_path:
        return FileResponse(UI_DIRECTORY / "index.html")
    return FileResponse(UI_DIRECTORY / "index.html")
