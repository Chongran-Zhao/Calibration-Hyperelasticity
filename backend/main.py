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
from kinematics import Kinematics
from material_models import MaterialModels
from optimization import MaterialOptimizer
from parallel_springs import ParallelNetwork
from utils import get_deformation_gradient, get_stress_components, load_experimental_data_h5

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
    if mode == "BT":
        stress_label = "Cauchy stress sigma11" if stress_type == "cauchy" else "Nominal stress P11"
        return "Variable stretch lambda_1 (-)", stress_label
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
    stretch_secondary = group["stretch_secondary"][()] if "stretch_secondary" in group else None
    stress_tensor = group["stress"][()]
    stress = _stress_series(family, stress_tensor)
    stress_secondary = stress_tensor[:, 1, 1] if family in ("BT", "PS") else None

    stretch_secondary_values = None
    fixed_secondary = None
    if stretch_secondary is not None:
        stretch_secondary_values = np.asarray(stretch_secondary).ravel()
        if stretch_secondary_values.size and np.allclose(stretch_secondary_values, stretch_secondary_values[0]):
            fixed_secondary = float(stretch_secondary_values[0])

    points = [
        {
            "x": float(x),
            "y": float(y),
            **({"x2": float(stretch_secondary_values[index])} if stretch_secondary_values is not None else {}),
            **({"y2": float(np.asarray(stress_secondary).ravel()[index])} if stress_secondary is not None else {}),
        }
        for index, (x, y) in enumerate(zip(np.asarray(stretch).ravel(), np.asarray(stress).ravel()))
    ]
    return {
        "mode": mode,
        "modeFamily": family,
        "modeLabel": _mode_label(mode),
        "stressType": stress_type,
        "component": "P11",
        "variableStretch": "lambda_1" if family == "BT" else ("gamma" if family in ("SS", "CSS") else "lambda"),
        "fixedStretch": fixed_secondary,
        "fixedStretchLabel": "lambda_2" if family == "BT" and fixed_secondary is not None else None,
        "points": points,
    }


def _json_bound(bound) -> list[float | None]:
    if not bound:
        return [None, None]
    lower, upper = bound
    return [None if lower is None else float(lower), None if upper is None else float(upper)]


def _safe_prefix(value: str) -> str:
    prefix = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")
    return prefix or "branch"


def _model_function(model_key: str, model_config: dict | None = None):
    config = model_config or {}
    if model_key == "Ogden":
        return MaterialModels.create_ogden_model(int(config.get("termCount", 1)))
    if model_key == "Hill":
        terms = config.get("terms") or [{"strain": "Seth-Hill"}]
        if len(terms) != 1:
            raise HTTPException(status_code=400, detail="Backend calibration currently supports one Hill term per branch.")
        return MaterialModels.create_hill_model(terms[0].get("strain", "Seth-Hill"))
    if not hasattr(MaterialModels, model_key):
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model_key}")
    return getattr(MaterialModels, model_key)


def _build_solver_from_payload(branches: list[dict]) -> tuple[Kinematics, list[float], list[tuple], list[dict]]:
    active = [branch for branch in branches if branch.get("enabled", True)]
    if not active:
        raise HTTPException(status_code=400, detail="At least one active branch is required.")

    network = ParallelNetwork()
    mappings = []
    for branch in active:
        model_key = branch.get("modelKey")
        model_func = _model_function(model_key, branch.get("modelConfig"))
        prefix = _safe_prefix(branch.get("id") or branch.get("name") or model_key)
        network.add_model(model_func, prefix)
        for local_name in getattr(model_func, "param_names", []):
            mappings.append(
                {
                    "branchId": branch.get("id"),
                    "name": local_name,
                    "key": f"{branch.get('id')}-{local_name}",
                    "solverKey": f"{prefix}_{local_name}",
                    "initial": branch.get("parameters", {}).get(local_name),
                }
            )

    initial_by_solver_key = {
        item["solverKey"]: item["initial"]
        for item in mappings
        if item["initial"] is not None
    }
    initial_guess = []
    for solver_key, fallback in zip(network.param_names, network.initial_guess):
        value = initial_by_solver_key.get(solver_key, fallback)
        try:
            initial_guess.append(float(value))
        except (TypeError, ValueError):
            initial_guess.append(float(fallback))

    solver = Kinematics(network, network.param_names)
    return solver, initial_guess, network.bounds, mappings


def _params_payload(mappings: list[dict], params: np.ndarray) -> list[dict]:
    by_solver_key = {mapping["solverKey"]: mapping for mapping in mappings}
    payload = []
    for solver_key, value in zip(by_solver_key.keys(), params):
        mapping = by_solver_key[solver_key]
        payload.append(
            {
                "branchId": mapping["branchId"],
                "name": mapping["name"],
                "key": mapping["key"],
                "solverKey": solver_key,
                "value": float(value),
            }
        )
    return payload


def _prediction_curves(h5, author: str, modes: list[str], solver: Kinematics, params: dict) -> list[dict]:
    curves = []
    for mode in modes:
        item = _read_mode_preview(h5, author, mode)
        points = sorted(item["points"], key=lambda point: point["x"])
        if not points:
            curves.append({**item, "key": mode, "family": item["modeFamily"], "points": []})
            continue

        x_values = np.linspace(points[0]["x"], points[-1]["x"], max(32, min(160, len(points) * 12)))
        family = item["modeFamily"]
        fixed_stretch = item.get("fixedStretch")
        model_points = []
        for x in x_values:
            if family == "BT":
                lam2 = fixed_stretch if fixed_stretch is not None else points[0].get("x2", 1.0)
                F = get_deformation_gradient((float(x), float(lam2)), "BT")
            elif family in ("SS", "CSS"):
                F = get_deformation_gradient(float(x), family)
            else:
                F = get_deformation_gradient(float(x), family)
            stress_tensor = (
                solver.get_Cauchy_stress(F, params)
                if item["stressType"] == "cauchy"
                else solver.get_1st_PK_stress(F, params)
            )
            components = get_stress_components(stress_tensor, family)
            point = {"x": float(x), "y": float(components[0])}
            if len(components) > 1:
                point["x2"] = float(lam2 if family == "BT" else x)
                point["y2"] = float(components[1])
            model_points.append(point)

        curves.append(
            {
                "key": mode,
                "family": family,
                "label": item["modeLabel"],
                "fixedStretch": item.get("fixedStretch"),
                "fixedStretchLabel": item.get("fixedStretchLabel"),
                "points": model_points,
            }
        )
    return curves


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


@app.post("/api/calibrate")
def calibrate(payload: dict):
    if not DATA_FILE.exists():
        raise HTTPException(status_code=500, detail="data/data.h5 not found")

    author = payload.get("author")
    modes = payload.get("modes") or []
    branches = payload.get("branches") or []
    solver_settings = payload.get("solver") or {}
    if not author or not modes:
        raise HTTPException(status_code=400, detail="Author and at least one mode are required.")

    solver, initial_guess, bounds, mappings = _build_solver_from_payload(branches)
    datasets = load_experimental_data_h5(
        [{"author": author, "mode": mode} for mode in modes],
        str(DATA_FILE),
        announce=False,
    )
    optimizer = MaterialOptimizer(solver, datasets)

    result = optimizer.fit(
        initial_guess,
        bounds,
        method=solver_settings.get("method", "L-BFGS-B"),
        max_iter=int(float(solver_settings.get("maxIter", 500))),
        r2_target=float(solver_settings.get("r2Target", 0.995)),
        abs_tol=float(solver_settings.get("absTol", 1e-6)),
        rel_tol=float(solver_settings.get("relTol", 1e-4)),
        max_loss=float(solver_settings.get("maxLoss", 0.05)),
    )
    params_array = np.asarray(result.x, dtype=float)
    params_dict = dict(zip(solver.param_names_ordered, params_array))

    with h5py.File(DATA_FILE, "r") as h5:
        curves = _prediction_curves(h5, author, modes, solver, params_dict)

    return {
        "success": bool(result.success),
        "message": str(result.message),
        "author": author,
        "modes": modes,
        "initialLoss": float(result.initial_loss),
        "loss": float(result.fun),
        "r2": float(result.r2_total),
        "r2Average": float(result.r2_avg),
        "iterations": int(result.nit),
        "parameters": _params_payload(mappings, params_array),
        "prediction": {
            "author": author,
            "modes": modes,
            "curves": curves,
        },
    }
