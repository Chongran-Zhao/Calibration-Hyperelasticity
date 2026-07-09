"""Data access and calibration orchestration behind the API routes."""

from __future__ import annotations

import os
import re
from pathlib import Path

import h5py
import numpy as np
from fastapi import HTTPException

from ..datasets import dataset_from_columns, load_experimental_data_h5
from ..evaluation import predict_curve
from ..kinematics import Kinematics
from ..models import MaterialModels
from ..network import ParallelNetwork
from ..optimizer import MaterialOptimizer
from ..paths import resource_root
from ..strains import STRAIN_CONFIGS, STRAIN_FORMULAS
from . import meta, user_data, user_models

REPO_ROOT = resource_root()


def resolve_data_file() -> Path:
    """Locate ``data.h5``: env override, repo checkout, or working directory."""
    env = os.environ.get("CALIBRATION_DATA_FILE")
    if env:
        return Path(env)
    candidates = [
        REPO_ROOT / "data" / "data.h5",
        Path.cwd() / "data" / "data.h5",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DATA_FILE = resolve_data_file()
HIDDEN_DATA_FAMILIES = {"CSS"}


def require_data_file():
    if not DATA_FILE.exists():
        raise HTTPException(status_code=500, detail="data/data.h5 not found")


def reject_hidden_modes(modes):
    hidden = [mode for mode in modes if meta.mode_family(mode) in HIDDEN_DATA_FAMILIES]
    if hidden:
        raise HTTPException(status_code=404, detail="CSS data are not available in the app.")


def resolve_source_meta(author: str) -> dict:
    if user_data.is_user_author(author):
        return {"name": user_data.material_of(author), "reference": "User data", "url": ""}
    return meta.source_meta(author)


# --- dataset reading ---------------------------------------------------------

def read_mode_preview(h5, author: str, mode: str) -> dict:
    if author not in h5 or mode not in h5[author]:
        raise HTTPException(status_code=404, detail=f"Dataset mode not found: {author}/{mode}")

    group = h5[author][mode]
    family = meta.mode_family(mode)
    stress_type = meta.as_text(group.attrs.get("stress_type", "PK1"))

    stretch = group["stretch"][()] if "stretch" in group else group["F"][:, 0, 0]
    stretch_secondary = group["stretch_secondary"][()] if "stretch_secondary" in group else None
    stress_tensor = group["stress"][()]
    if family in ("SS", "CSS"):
        stress = stress_tensor[:, 0, 1]
    else:
        stress = stress_tensor[:, 0, 0]
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
        "modeLabel": meta.mode_label(mode),
        "modeShortLabel": meta.mode_short_label(mode),
        **meta.mode_ui_meta(mode),
        "tensorExpressions": meta.mode_tensor_expressions(family, mode, fixed_secondary),
        "axisSymbols": meta.axis_symbols(family, stress_type),
        "stressType": stress_type,
        "stressDisplay": meta.stress_display(stress_type),
        "component": "P11",
        "variableStretch": "lambda_1" if family == "BT" else ("gamma" if family in ("SS", "CSS") else "lambda"),
        "fixedStretch": fixed_secondary,
        "fixedStretchLabel": "lambda_2" if family == "BT" and fixed_secondary is not None else None,
        "points": points,
    }


def list_datasets() -> dict:
    require_data_file()
    authors = []
    with h5py.File(DATA_FILE, "r") as h5:
        for author in sorted(h5.keys()):
            modes = []
            for mode_raw in sorted(h5[author].keys()):
                family = meta.mode_family(mode_raw)
                if family in HIDDEN_DATA_FAMILIES:
                    continue
                group = h5[author][mode_raw]
                stress_type = meta.as_text(group.attrs.get("stress_type", "PK1"))
                modes.append(
                    {
                        "key": mode_raw,
                        "family": family,
                        "label": meta.mode_label(mode_raw),
                        "shortLabel": meta.mode_short_label(mode_raw),
                        **meta.mode_ui_meta(mode_raw),
                        "points": int(group["stretch"].shape[0]) if "stretch" in group else int(group["F"].shape[0]),
                        "stressType": stress_type,
                        "stressDisplay": meta.stress_display(stress_type),
                    }
                )
            if not modes:
                continue
            entry = {"author": author, "modes": modes}
            entry.update(meta.source_meta(author))
            authors.append(entry)

    # User-uploaded materials behave as additional sources.
    for material, docs in user_data.materials().items():
        modes = [
            {
                "key": doc["modeKey"],
                "family": doc["family"],
                "label": meta.mode_label(doc["modeKey"]),
                "shortLabel": meta.mode_label(doc["modeKey"]),
                "points": len(doc.get("points", [])),
                "stressType": doc["stressType"],
                "stressDisplay": meta.stress_display(doc["stressType"]),
                "datasetId": doc["id"],
                "isUserData": True,
            }
            for doc in docs
        ]
        authors.append({
            "author": user_data.USER_AUTHOR_PREFIX + material,
            "modes": modes,
            "name": material,
            "reference": "User data",
            "url": "",
            "isUserData": True,
        })
    return {"authors": authors}


def user_mode_preview(author: str, mode_key: str) -> dict:
    """Preview payload item for a user-uploaded dataset (same shape as
    :func:`read_mode_preview`)."""
    doc = user_data.get_dataset(user_data.material_of(author), mode_key)
    family = doc["family"]
    stress_type = doc["stressType"]
    pts = np.asarray(doc["points"], dtype=float)

    if family == "BT":
        stretch = pts[:, 0]
        stretch_secondary = pts[:, 1]
        stress = pts[:, 2]
        stress_secondary = pts[:, 3] if pts.shape[1] == 4 else None
    else:
        stretch = pts[:, 0]
        stress = pts[:, 1]
        stretch_secondary = None
        stress_secondary = None

    fixed_secondary = None
    if stretch_secondary is not None and stretch_secondary.size and np.allclose(stretch_secondary, stretch_secondary[0]):
        fixed_secondary = float(stretch_secondary[0])

    points = [
        {
            "x": float(x),
            "y": float(y),
            **({"x2": float(stretch_secondary[index])} if stretch_secondary is not None else {}),
            **({"y2": float(stress_secondary[index])} if stress_secondary is not None else {}),
        }
        for index, (x, y) in enumerate(zip(stretch, stress))
    ]
    return {
        "mode": mode_key,
        "modeFamily": family,
        "modeLabel": meta.mode_label(mode_key),
        "modeShortLabel": meta.mode_label(mode_key),
        "tensorExpressions": meta.mode_tensor_expressions(family, mode_key, fixed_secondary),
        "axisSymbols": meta.axis_symbols(family, stress_type),
        "stressType": stress_type,
        "stressDisplay": meta.stress_display(stress_type),
        "component": "P11",
        "variableStretch": "lambda_1" if family == "BT" else ("gamma" if family in ("SS", "CSS") else "lambda"),
        "fixedStretch": fixed_secondary,
        "fixedStretchLabel": "lambda_2" if family == "BT" and fixed_secondary is not None else None,
        "points": points,
    }


def mode_previews(author: str, modes: list) -> list:
    """Preview items for any source: built-in HDF5 or user uploads."""
    if user_data.is_user_author(author):
        return [user_mode_preview(author, mode) for mode in modes]
    require_data_file()
    reject_hidden_modes(modes)
    with h5py.File(DATA_FILE, "r") as h5:
        return [read_mode_preview(h5, author, mode) for mode in modes]


def load_calibration_datasets(author: str, modes: list) -> list:
    """Optimizer-ready dataset entries for any source."""
    if user_data.is_user_author(author):
        material = user_data.material_of(author)
        entries = []
        for mode in modes:
            doc = user_data.get_dataset(material, mode)
            entries.append(
                dataset_from_columns(
                    material,
                    doc["modeKey"],
                    doc["points"],
                    stress_type=doc["stressType"],
                    tag=f"{material}_{doc['modeKey']}",
                )
            )
        return entries
    require_data_file()
    reject_hidden_modes(modes)
    return load_experimental_data_h5(
        [{"author": author, "mode": mode} for mode in modes],
        str(DATA_FILE),
    )


def preview_payload(author: str, modes: list) -> dict:
    series = mode_previews(author, modes)

    rows = sum(len(item["points"]) for item in series)
    stress_types = sorted({item["stressType"] for item in series})
    families = [item["modeFamily"] for item in series]
    primary_family = families[0] if families else ""
    primary_stress = stress_types[0] if len(stress_types) == 1 else "mixed"
    source = resolve_source_meta(author)
    x_label, y_label = meta.axis_labels(primary_family, primary_stress)

    return {
        "author": author,
        "source": source,
        "modes": modes,
        "modeFamilies": families,
        "stressTypes": stress_types,
        "series": series,
        "points": series[0]["points"] if series else [],
        "metadata": {
            "rows": rows,
            "source": source["name"],
            "sourceReference": source["reference"],
            "sourceUrl": source["url"],
            "selectedMode": ", ".join(item["modeLabel"] for item in series),
            "selectedModeShort": ", ".join(item["modeShortLabel"] for item in series),
            "stressType": "Mixed" if len(stress_types) > 1 else ("Cauchy" if primary_stress == "cauchy" else "First PK"),
            "stressDisplay": meta.stress_display(primary_stress),
            "setCount": len(series),
        },
        "axes": {"x": x_label, "y": y_label},
    }


# --- model catalogue -----------------------------------------------------------

def json_bound(bound) -> list:
    if not bound:
        return [None, None]
    lower, upper = bound
    return [None if lower is None else float(lower), None if upper is None else float(upper)]


def model_payload(model_func, name=None, extra=None) -> dict:
    params = getattr(model_func, "param_names", [])
    guesses = getattr(model_func, "initial_guess", [])
    bounds = getattr(model_func, "bounds", [])
    key = name or model_func.__name__
    model_meta = meta.MODEL_META.get(key, {})
    payload = {
        "key": key,
        "name": model_meta.get("name", key),
        "type": getattr(model_func, "model_type", "unknown"),
        "category": getattr(model_func, "category", "unknown"),
        "reference": model_meta.get("reference", ""),
        "referenceUrl": model_meta.get("referenceUrl", ""),
        "formula": getattr(model_func, "formula", ""),
        "strainFormula": getattr(model_func, "strain_formula", ""),
        "parameters": [
            {
                "name": param,
                "initial": None if index >= len(guesses) else float(guesses[index]),
                "bounds": json_bound(bounds[index] if index < len(bounds) else None),
            }
            for index, param in enumerate(params)
        ],
    }
    if extra:
        payload.update(extra)
    return payload


def strain_options() -> list:
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
                        "bounds": json_bound(bounds[index] if index < len(bounds) else None),
                    }
                    for index, param in enumerate(config.get("params", []))
                ],
            }
        )
    return options


def hill_payload() -> dict:
    strains = strain_options()
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


def model_catalogue() -> dict:
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
        model_payload(
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
    model_items.append(hill_payload())

    # User-defined models join the catalogue as first-class entries.
    for doc in user_models.all_models():
        try:
            model_func = user_models.build_model_function(doc)
        except Exception:  # noqa: BLE001 - a broken stored model must not kill the catalogue
            continue
        model_items.append({
            "key": user_models.USER_MODEL_PREFIX + doc["id"],
            "name": doc["name"],
            "type": model_func.model_type,
            "category": "user",
            "reference": doc.get("notes", "") or "User-defined model",
            "referenceUrl": "",
            "formula": model_func.formula,
            "strainFormula": "",
            "isUserModel": True,
            "parameters": [
                {
                    "name": item["name"],
                    "initial": item.get("initial"),
                    "bounds": [item.get("lower"), item.get("upper")],
                }
                for item in doc.get("params", [])
            ],
        })

    return {
        "models": model_items,
        "categories": sorted({item["category"] for item in model_items}),
        "types": sorted({item["type"] for item in model_items}),
        "strains": strain_options(),
    }


# --- solver construction ---------------------------------------------------------

def safe_prefix(value: str) -> str:
    prefix = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")
    return prefix or "branch"


def model_function(model_key, model_config=None):
    config = model_config or {}
    if user_models.is_user_model_key(model_key):
        return user_models.build_model_function(user_models.get_model(user_models.model_id_of(model_key)))
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


def build_solver_from_payload(branches: list):
    """Assemble the ParallelNetwork solver plus parameter bookkeeping."""
    active = [branch for branch in branches if branch.get("enabled", True)]
    if not active:
        raise HTTPException(status_code=400, detail="At least one active branch is required.")

    network = ParallelNetwork()
    mappings = []
    for branch in active:
        model_key = branch.get("modelKey")
        model_func = model_function(model_key, branch.get("modelConfig"))
        prefix = safe_prefix(branch.get("id") or branch.get("name") or model_key)
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


def params_payload(mappings: list, params) -> list:
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


# --- prediction curves --------------------------------------------------------------

def prediction_curves(author: str, modes: list, solver: Kinematics, params: dict) -> list:
    curves = []
    for item, mode in zip(mode_previews(author, modes), modes):
        points = sorted(item["points"], key=lambda point: point["x"])
        if not points:
            curves.append({**item, "key": mode, "family": item["modeFamily"], "points": []})
            continue

        x_values = np.linspace(points[0]["x"], points[-1]["x"], max(32, min(160, len(points) * 12)))
        family = item["modeFamily"]
        fixed_stretch = item.get("fixedStretch")

        if family == "BT":
            lam2 = fixed_stretch if fixed_stretch is not None else points[0].get("x2", 1.0)
            grid = [(float(x), float(lam2)) for x in x_values]
        else:
            lam2 = None
            grid = [float(x) for x in x_values]

        comps = predict_curve(solver, family, grid, params, item["stressType"])
        model_points = []
        for index, x in enumerate(x_values):
            point = {"x": float(x), "y": float(comps[0][index])}
            if len(comps) > 1:
                point["x2"] = float(lam2 if family == "BT" else x)
                point["y2"] = float(comps[1][index])
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


# --- top-level operations --------------------------------------------------------------

def calibrate(payload: dict) -> dict:
    author = payload.get("author")
    modes = payload.get("modes") or []
    branches = payload.get("branches") or []
    solver_settings = payload.get("solver") or {}
    if not author or not modes:
        raise HTTPException(status_code=400, detail="Author and at least one mode are required.")

    solver, initial_guess, bounds, mappings = build_solver_from_payload(branches)
    datasets = load_calibration_datasets(author, modes)
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
    curves = prediction_curves(author, modes, solver, params_dict)

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
        "parameters": params_payload(mappings, params_array),
        "prediction": {
            "author": author,
            "modes": modes,
            "curves": curves,
        },
    }


def predict(payload: dict) -> dict:
    author = payload.get("author")
    modes = payload.get("modes") or []
    branches = payload.get("branches") or []
    if not author or not modes:
        raise HTTPException(status_code=400, detail="Author and at least one prediction mode are required.")

    solver, params_array, _bounds, _mappings = build_solver_from_payload(branches)
    params_dict = dict(zip(solver.param_names_ordered, np.asarray(params_array, dtype=float)))
    curves = prediction_curves(author, modes, solver, params_dict)

    return {
        "author": author,
        "modes": modes,
        "curves": curves,
    }
