"""User-uploaded experimental datasets.

Uploads are grouped by *material*: each material behaves exactly like a
built-in literature source (an "author" with one mode per uploaded test), so
the preview / calibration / prediction pipeline consumes user data without
special cases downstream.

Layout: one ``{id}.json`` per uploaded test under ``~/.hyperfit/datasets``
(override with ``HYPERFIT_DATASETS_DIR``). Author keys are ``user:{material}``.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from pathlib import Path

from fastapi import HTTPException

USER_AUTHOR_PREFIX = "user:"

#: Mode families a user can upload, with expected column counts.
UPLOAD_FAMILIES = {
    "UT": (2, 2),
    "UC": (2, 2),
    "ET": (2, 2),
    "PS": (2, 2),
    "SS": (2, 2),
    "BT": (3, 4),
}

MAX_POINTS = 5000


def is_user_author(author) -> bool:
    return isinstance(author, str) and author.startswith(USER_AUTHOR_PREFIX)


def material_of(author: str) -> str:
    return author[len(USER_AUTHOR_PREFIX):]


def datasets_dir() -> Path:
    env = os.environ.get("HYPERFIT_DATASETS_DIR")
    base = Path(env) if env else Path.home() / ".hyperfit" / "datasets"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _path(dataset_id: str) -> Path:
    if not dataset_id or not dataset_id.isalnum():
        raise HTTPException(status_code=400, detail=f"Invalid dataset id: {dataset_id!r}")
    return datasets_dir() / f"{dataset_id}.json"


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _all_documents() -> list:
    docs = []
    for path in datasets_dir().glob("*.json"):
        doc = _read(path)
        if doc and "id" in doc:
            docs.append(doc)
    docs.sort(key=lambda doc: doc.get("createdAt", 0))
    return docs


def materials() -> dict:
    """Uploaded tests grouped by material name, in upload order."""
    grouped = {}
    for doc in _all_documents():
        grouped.setdefault(doc["material"], []).append(doc)
    return grouped


def get_dataset(material: str, mode_key: str):
    for doc in materials().get(material, []):
        if doc["modeKey"] == mode_key:
            return doc
    raise HTTPException(status_code=404, detail=f"User dataset not found: {material}/{mode_key}")


def _validate_points(family: str, points) -> list:
    min_cols, max_cols = UPLOAD_FAMILIES[family]
    if not isinstance(points, list) or len(points) < 2:
        raise HTTPException(status_code=400, detail="At least 2 data points are required.")
    if len(points) > MAX_POINTS:
        raise HTTPException(status_code=400, detail=f"Too many points (max {MAX_POINTS}).")

    width = None
    cleaned = []
    for index, row in enumerate(points):
        if not isinstance(row, (list, tuple)):
            raise HTTPException(status_code=400, detail=f"Row {index + 1} is not a list of numbers.")
        if width is None:
            width = len(row)
            if not (min_cols <= width <= max_cols):
                expected = f"{min_cols}" if min_cols == max_cols else f"{min_cols}-{max_cols}"
                raise HTTPException(status_code=400, detail=f"{family} data needs {expected} columns, got {width}.")
        elif len(row) != width:
            raise HTTPException(status_code=400, detail=f"Row {index + 1} has {len(row)} columns, expected {width}.")
        values = []
        for value in row:
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"Row {index + 1} contains a non-numeric value.") from None
            if not math.isfinite(number):
                raise HTTPException(status_code=400, detail=f"Row {index + 1} contains a non-finite value.")
            values.append(number)
        # Stretches must be positive (columns before the stress columns);
        # shear strain gamma in SS may be any sign.
        stretch_cols = 2 if family == "BT" else (0 if family == "SS" else 1)
        for column in range(stretch_cols):
            if values[column] <= 0:
                raise HTTPException(status_code=400, detail=f"Row {index + 1}: stretch must be positive.")
        cleaned.append(values)
    return cleaned


def _next_mode_key(material: str, family: str) -> str:
    taken = {doc["modeKey"] for doc in materials().get(material, [])}
    if family not in taken:
        return family
    suffix = 2
    while f"{family}_{suffix}" in taken:
        suffix += 1
    return f"{family}_{suffix}"


def save_dataset(payload: dict) -> dict:
    material = str(payload.get("material") or "").strip()
    if not material:
        raise HTTPException(status_code=400, detail="Material name is required.")
    if len(material) > 60:
        raise HTTPException(status_code=400, detail="Material name is too long (max 60 characters).")

    family = payload.get("family")
    if family not in UPLOAD_FAMILIES:
        raise HTTPException(status_code=400, detail=f"Unsupported loading mode: {family}")

    stress_type = payload.get("stressType", "PK1")
    if stress_type not in ("PK1", "cauchy"):
        raise HTTPException(status_code=400, detail=f"Unsupported stress type: {stress_type}")

    points = _validate_points(family, payload.get("points"))

    dataset_id = uuid.uuid4().hex[:12]
    document = {
        "version": 1,
        "id": dataset_id,
        "material": material,
        "family": family,
        "modeKey": _next_mode_key(material, family),
        "stressType": stress_type,
        "fileName": str(payload.get("fileName") or "")[:120],
        "createdAt": time.time(),
        "points": points,
    }
    path = _path(dataset_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(document, ensure_ascii=False))
    tmp.replace(path)
    return {
        "id": dataset_id,
        "author": USER_AUTHOR_PREFIX + material,
        "modeKey": document["modeKey"],
    }


def delete_dataset(dataset_id: str) -> dict:
    path = _path(dataset_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"User dataset not found: {dataset_id}")
    path.unlink()
    return {"deleted": dataset_id}


def list_payload() -> dict:
    """Flat listing for the management UI."""
    items = [
        {
            "id": doc["id"],
            "material": doc["material"],
            "author": USER_AUTHOR_PREFIX + doc["material"],
            "modeKey": doc["modeKey"],
            "family": doc["family"],
            "stressType": doc["stressType"],
            "fileName": doc.get("fileName", ""),
            "points": len(doc.get("points", [])),
            "createdAt": doc.get("createdAt", 0),
        }
        for doc in _all_documents()
    ]
    return {"datasets": items}
