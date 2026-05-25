from pathlib import Path
import re
import sys

import h5py
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "data.h5"
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generalized_strains import STRAIN_CONFIGS, STRAIN_FORMULAS
from material_models import MaterialModels

MODE_LABELS = {
    "UT": "Uniaxial Tension",
    "UC": "Uniaxial Compression",
    "ET": "Equibiaxial Tension",
    "PS": "Pure Shear",
    "SS": "Simple Shear",
    "CSS": "Compound Simple Shear",
    "BT": "Biaxial Tension",
}

MODEL_META = {
    "NeoHookean": {
        "name": "Neo-Hookean",
        "reference": "Rivlin (1948); Mooney (1940)",
    },
    "MooneyRivlin": {
        "name": "Mooney-Rivlin",
        "reference": "Mooney (1940); Rivlin (1948)",
    },
    "Yeoh": {
        "name": "Yeoh",
        "reference": "Yeoh (1993)",
    },
    "ArrudaBoyce": {
        "name": "Arruda-Boyce",
        "reference": "Arruda and Boyce (1993)",
    },
    "ZhanGaussian": {
        "name": "Zhan-Gaussian",
        "reference": "Zhan et al. (2023)",
    },
    "ZhanNonGaussian": {
        "name": "Zhan-Non-Gaussian",
        "reference": "Zhan et al. (2023)",
    },
    "Ogden": {
        "name": "Ogden",
        "reference": "Ogden (1972)",
    },
    "ModifiedOgden": {
        "name": "Modified-Ogden",
        "reference": "Budday et al. (2017); Ogden (1972)",
    },
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


def _json_bound(bound) -> list[float | None]:
    if not bound:
        return [None, None]
    lower, upper = bound
    return [None if lower is None else float(lower), None if upper is None else float(upper)]


def _model_payload(model_func, name: str | None = None, extra: dict | None = None) -> dict:
    params = getattr(model_func, "param_names", [])
    guesses = getattr(model_func, "initial_guess", [])
    bounds = getattr(model_func, "bounds", [])
    key = name or model_func.__name__
    meta = MODEL_META.get(key, {})
    payload = {
        "key": key,
        "name": meta.get("name", key),
        "type": getattr(model_func, "model_type", "unknown"),
        "category": getattr(model_func, "category", "unknown"),
        "reference": meta.get("reference", ""),
        "formula": getattr(model_func, "formula", ""),
        "strainFormula": getattr(model_func, "strain_formula", ""),
        "parameters": [
            {
                "name": param,
                "initial": None if index >= len(guesses) else float(guesses[index]),
                "bounds": _json_bound(bounds[index] if index < len(bounds) else None),
            }
            for index, param in enumerate(params)
        ],
    }
    if extra:
        payload.update(extra)
    return payload


def _strain_options() -> list[dict]:
    options = []
    for strain_name, config in STRAIN_CONFIGS.items():
        bounds = config.get("bounds", [])
        options.append(
            {
                "key": strain_name,
                "name": strain_name,
                "formula": STRAIN_FORMULAS.get(strain_name, ""),
                "parameters": [
                    {
                        "name": param,
                        "initial": None if index >= len(config.get("defaults", [])) else float(config["defaults"][index]),
                        "bounds": _json_bound(bounds[index] if index < len(bounds) else None),
                    }
                    for index, param in enumerate(config.get("params", []))
                ],
            }
        )
    return options


def _hill_payload() -> dict:
    strains = _strain_options()
    return {
        "key": "Hill",
        "name": "Hill",
        "type": "stretch_based",
        "category": "phenomenological",
        "reference": "Hill (1978); Seth (1964)",
        "formula": r"\Psi = \sum_{k=1}^{n_t} \mu_k \sum_{i=1}^{3} E_k(\lambda_i)^2",
        "strainFormula": strains[0]["formula"] if strains else "",
        "parameters": [],
        "configurable": {
            "kind": "hill",
            "minTerms": 1,
            "maxTerms": 5,
            "strains": strains,
        },
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


@app.get("/api/models")
def models():
    base_names = [
        "NeoHookean",
        "MooneyRivlin",
        "Yeoh",
        "ArrudaBoyce",
        "ZhanGaussian",
        "ZhanNonGaussian",
        "Ogden",
        "ModifiedOgden",
    ]
    model_items = [
        _model_payload(
            getattr(MaterialModels, name),
            name,
            {
                "configurable": {
                    "kind": "ogden",
                    "minTerms": 1,
                    "maxTerms": 5,
                },
            } if name == "Ogden" else None,
        )
        for name in base_names
    ]
    model_items.append(_hill_payload())
    categories = sorted({item["category"] for item in model_items})
    types = sorted({item["type"] for item in model_items})
    return {
        "models": model_items,
        "categories": categories,
        "types": types,
        "strains": _strain_options(),
    }


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
