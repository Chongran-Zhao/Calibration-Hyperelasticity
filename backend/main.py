from pathlib import Path
import re

import h5py
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "data.h5"

MODE_LABELS = {
    "UT": "Uniaxial Tension",
    "UC": "Uniaxial Compression",
    "ET": "Equibiaxial Tension",
    "PS": "Pure Shear",
    "SS": "Simple Shear",
    "CSS": "Compound Simple Shear",
    "BT": "Biaxial Tension",
}


app = FastAPI(title="Hyperelastic Calibration API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _mode_family(mode_raw: str) -> str:
    for prefix in ("CSS", "BT", "UT", "UC", "ET", "PS", "SS"):
        if mode_raw.startswith(prefix):
            return prefix
    return mode_raw


def _mode_label(mode_raw: str) -> str:
    family = _mode_family(mode_raw)
    base = MODE_LABELS.get(family, mode_raw)
    if mode_raw == family:
        return base
    suffix = mode_raw[len(family):].strip("_")
    suffix = re.sub(r"(?<=\d)d(?=\d)", ".", suffix.replace("_", " "))
    return f"{base}, {suffix}" if suffix else base


def _as_text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _stress_series(mode: str, stress_tensor: np.ndarray) -> np.ndarray:
    if mode in ("SS", "CSS"):
        return stress_tensor[:, 0, 1]
    return stress_tensor[:, 0, 0]


def _axis_labels(mode: str, stress_type: str) -> tuple[str, str]:
    if mode in ("SS", "CSS"):
        return "Shear strain gamma (-)", "Shear stress P12"
    stress_label = "Cauchy stress sigma11" if stress_type == "cauchy" else "Nominal stress P11"
    return "Stretch lambda (-)", stress_label


def _read_mode_preview(h5, author: str, mode: str) -> dict:
    if author not in h5 or mode not in h5[author]:
        raise HTTPException(status_code=404, detail=f"Dataset mode not found: {author}/{mode}")

    group = h5[author][mode]
    family = _mode_family(mode)
    stress_type = group.attrs.get("stress_type", "PK1")
    if isinstance(stress_type, bytes):
        stress_type = stress_type.decode("utf-8")

    stretch = group["stretch"][()] if "stretch" in group else group["F"][:, 0, 0]
    stress = _stress_series(family, group["stress"][()])
    points = [
        {"x": float(x), "y": float(y)}
        for x, y in zip(np.asarray(stretch).ravel(), np.asarray(stress).ravel())
    ]
    return {
        "mode": mode,
        "modeFamily": family,
        "modeLabel": _mode_label(mode),
        "stressType": stress_type,
        "points": points,
    }


@app.get("/api/health")
def health():
    return {"ok": True, "data_file": DATA_FILE.exists()}


@app.get("/api/datasets")
def datasets():
    if not DATA_FILE.exists():
        raise HTTPException(status_code=500, detail="data/data.h5 not found")

    authors = []
    with h5py.File(DATA_FILE, "r") as h5:
        for author in sorted(h5.keys()):
            modes = []
            for mode_raw in sorted(h5[author].keys()):
                group = h5[author][mode_raw]
                stress_type = _as_text(group.attrs.get("stress_type", "PK1"))
                modes.append(
                    {
                        "key": mode_raw,
                        "family": _mode_family(mode_raw),
                        "label": _mode_label(mode_raw),
                        "points": int(group["stretch"].shape[0]) if "stretch" in group else int(group["F"].shape[0]),
                        "stressType": stress_type,
                    }
                )
            authors.append({"author": author, "modes": modes})
    return {"authors": authors}


@app.get("/api/preview")
def preview(
    author: str = Query(..., min_length=1),
    mode: list[str] = Query(..., min_length=1),
):
    if not DATA_FILE.exists():
        raise HTTPException(status_code=500, detail="data/data.h5 not found")

    with h5py.File(DATA_FILE, "r") as h5:
        series = [_read_mode_preview(h5, author, item) for item in mode]

    rows = sum(len(item["points"]) for item in series)
    stress_types = sorted({item["stressType"] for item in series})
    families = [item["modeFamily"] for item in series]
    primary_family = families[0] if families else ""
    primary_stress = stress_types[0] if len(stress_types) == 1 else "mixed"
    x_label, y_label = _axis_labels(primary_family, primary_stress)

    return {
        "author": author,
        "modes": mode,
        "modeFamilies": families,
        "stressTypes": stress_types,
        "series": series,
        "points": series[0]["points"] if series else [],
        "metadata": {
            "rows": rows,
            "source": f"{author}/{', '.join(mode)}",
            "selectedMode": ", ".join(item["modeLabel"] for item in series),
            "stressType": "Mixed" if len(stress_types) > 1 else ("Cauchy" if primary_stress == "cauchy" else "First PK"),
            "setCount": len(series),
        },
        "axes": {"x": x_label, "y": y_label},
    }
