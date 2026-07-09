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

from . import service, sessions, user_data, user_models

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

    # --- user-uploaded experimental datasets ---

    @app.get("/api/user-datasets")
    def user_datasets_list():
        return user_data.list_payload()

    @app.post("/api/user-datasets")
    def user_datasets_save(payload: dict):
        return user_data.save_dataset(payload)

    @app.delete("/api/user-datasets/{dataset_id}")
    def user_datasets_delete(dataset_id: str):
        return user_data.delete_dataset(dataset_id)

    # --- user-defined models ---

    @app.get("/api/user-models")
    def user_models_list():
        return user_models.list_payload()

    @app.post("/api/user-models")
    def user_models_save(payload: dict):
        return user_models.save_model(payload)

    @app.post("/api/user-models/validate")
    def user_models_validate(payload: dict):
        return user_models.validate_payload(payload)

    @app.delete("/api/user-models/{model_id}")
    def user_models_delete(model_id: str):
        return user_models.delete_model(model_id)

    # --- saved calibration sessions (start page) ---

    @app.get("/api/sessions")
    def sessions_list():
        return sessions.list_sessions()

    @app.post("/api/sessions")
    def sessions_save(payload: dict):
        return sessions.save_session(payload)

    @app.get("/api/sessions/{session_id}")
    def sessions_get(session_id: str):
        return sessions.get_session(session_id)

    @app.post("/api/sessions/{session_id}/rename")
    def sessions_rename(session_id: str, payload: dict):
        return sessions.rename_session(session_id, payload.get("name", ""))

    @app.post("/api/sessions/{session_id}/duplicate")
    def sessions_duplicate(session_id: str):
        return sessions.duplicate_session(session_id)

    @app.delete("/api/sessions/{session_id}")
    def sessions_delete(session_id: str):
        return sessions.delete_session(session_id)

    if static_dir is not False:
        dist = Path(static_dir) if static_dir else DEFAULT_FRONTEND_DIST
        if dist.is_dir() and (dist / "index.html").exists():
            app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")

    return app
