"""Saved calibration sessions: JSON documents on the local disk.

A session captures the full workbench state (dataset selection, model
composition, solver settings, results) plus a lightweight ``summary`` used to
render the start-page cards without loading the full state.

Layout: one ``{id}.json`` per session under ``~/.hyperfit/sessions`` (override
with the ``HYPERFIT_SESSIONS_DIR`` environment variable).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from fastapi import HTTPException


def sessions_dir() -> Path:
    env = os.environ.get("HYPERFIT_SESSIONS_DIR")
    base = Path(env) if env else Path.home() / ".hyperfit" / "sessions"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _path(session_id: str) -> Path:
    # IDs are generated server-side (hex); reject anything path-like.
    if not session_id or not session_id.isalnum():
        raise HTTPException(status_code=400, detail=f"Invalid session id: {session_id!r}")
    return sessions_dir() / f"{session_id}.json"


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write(path: Path, document: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(document, ensure_ascii=False, indent=1))
    tmp.replace(path)


def list_sessions() -> dict:
    """Card list for the start page: metadata + summary, newest first."""
    items = []
    for path in sessions_dir().glob("*.json"):
        doc = _read(path)
        if not doc or "id" not in doc:
            continue
        items.append({
            "id": doc["id"],
            "name": doc.get("name", "Untitled"),
            "createdAt": doc.get("createdAt", 0),
            "updatedAt": doc.get("updatedAt", 0),
            "summary": doc.get("summary", {}),
        })
    items.sort(key=lambda item: item["updatedAt"], reverse=True)
    return {"sessions": items}


def get_session(session_id: str) -> dict:
    doc = _read(_path(session_id))
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return doc


def save_session(payload: dict) -> dict:
    """Create a new session, or overwrite when ``id`` matches an existing one."""
    state = payload.get("state")
    if not isinstance(state, dict) or not state:
        raise HTTPException(status_code=400, detail="Session state is required.")

    now = time.time()
    session_id = payload.get("id")
    existing = None
    if session_id:
        existing = _read(_path(session_id))
    if not session_id or existing is None:
        session_id = uuid.uuid4().hex[:12]

    document = {
        "version": 1,
        "id": session_id,
        "name": str(payload.get("name") or "Untitled calibration"),
        "createdAt": existing.get("createdAt", now) if existing else now,
        "updatedAt": now,
        "summary": payload.get("summary") or {},
        "state": state,
    }
    _write(_path(session_id), document)
    return {"id": session_id, "updatedAt": now}


def rename_session(session_id: str, name: str) -> dict:
    doc = get_session(session_id)
    doc["name"] = str(name or doc["name"])
    doc["updatedAt"] = time.time()
    _write(_path(session_id), doc)
    return {"id": session_id, "name": doc["name"], "updatedAt": doc["updatedAt"]}


def duplicate_session(session_id: str) -> dict:
    doc = get_session(session_id)
    copy_id = uuid.uuid4().hex[:12]
    now = time.time()
    copy = dict(doc, id=copy_id, name=f"{doc.get('name', 'Untitled')} · Copy", createdAt=now, updatedAt=now)
    _write(_path(copy_id), copy)
    return {"id": copy_id}


def delete_session(session_id: str) -> dict:
    path = _path(session_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    path.unlink()
    return {"deleted": session_id}
