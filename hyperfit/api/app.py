"""FastAPI application factory.

``create_app()`` serves the JSON API under ``/api/*`` and, when a built
frontend is available (``frontend/dist``), the static web UI at ``/`` --
which is how the desktop app ships a single self-contained server.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import service

# Origins used by `npm run dev` (vite) during frontend development.
DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

DEFAULT_FRONTEND_DIST = service.REPO_ROOT / "frontend" / "dist"


def create_app(static_dir=None) -> FastAPI:
    """Build the application.

    Args:
        static_dir: Directory with the built frontend. Defaults to
            ``frontend/dist`` when it exists; pass ``False`` to disable
            static serving entirely.
    """
    app = FastAPI(title="Hyperelastic Calibration API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        return {"ok": True, "data_file": service.DATA_FILE.exists()}

    @app.get("/api/datasets")
    def datasets():
        return service.list_datasets()

    @app.get("/api/models")
    def models():
        return service.model_catalogue()

    @app.get("/api/preview")
    def preview(
        author: str = Query(..., min_length=1),
        mode: list[str] = Query(..., min_length=1),
    ):
        return service.preview_payload(author, mode)

    @app.post("/api/calibrate")
    def calibrate(payload: dict):
        return service.calibrate(payload)

    @app.post("/api/predict")
    def predict(payload: dict):
        return service.predict(payload)

    if static_dir is not False:
        dist = Path(static_dir) if static_dir else DEFAULT_FRONTEND_DIST
        if dist.is_dir() and (dist / "index.html").exists():
            app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")

    return app
