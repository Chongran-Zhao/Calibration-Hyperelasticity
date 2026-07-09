"""User-defined hyperelastic models from strain-energy expressions.

Design: *the energy is the model*. A user supplies the strain-energy density
Psi as a math expression — either invariant-based ``Psi(I1, I2)`` or
stretch-based ``Psi(lambda_1, lambda_2, lambda_3)`` — plus parameter initial
values and bounds. The expression is parsed with a restricted sympy parser,
physics-checked, stored as a small JSON document, and materialised on demand
as a tagged energy function that plugs into the exact same symbolic pipeline
(:class:`hyperfit.kinematics.Kinematics`) as the built-in models: automatic
differentiation, pressure elimination, vectorised evaluation, parallel
networks and sessions all work unchanged.

Layout: one ``{id}.json`` per model under ``~/.hyperfit/models`` (override
with ``HYPERFIT_MODELS_DIR``). Catalogue keys are ``user-model:{id}``.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from pathlib import Path

import numpy as np
import sympy as sp
from fastapi import HTTPException
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

USER_MODEL_PREFIX = "user-model:"

MAX_EXPRESSION_LENGTH = 600
MAX_OPS = 300
MAX_PARAMS = 12

#: Whitelisted functions and constants available inside expressions.
SAFE_FUNCTIONS = {
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "asinh": sp.asinh,
    "acosh": sp.acosh,
    "atanh": sp.atanh,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "atan": sp.atan,
    "pi": sp.pi,
    "E": sp.E,
}

KINEMATIC_SYMBOLS = {
    "invariant": ("I1", "I2"),
    "stretch": ("lambda_1", "lambda_2", "lambda_3"),
}

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def is_user_model_key(model_key) -> bool:
    return isinstance(model_key, str) and model_key.startswith(USER_MODEL_PREFIX)


def model_id_of(model_key: str) -> str:
    return model_key[len(USER_MODEL_PREFIX):]


def models_dir() -> Path:
    env = os.environ.get("HYPERFIT_MODELS_DIR")
    base = Path(env) if env else Path.home() / ".hyperfit" / "models"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _path(model_id: str) -> Path:
    if not model_id or not model_id.isalnum():
        raise HTTPException(status_code=400, detail=f"Invalid model id: {model_id!r}")
    return models_dir() / f"{model_id}.json"


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def all_models() -> list:
    docs = [doc for doc in (_read(p) for p in models_dir().glob("*.json")) if doc and "id" in doc]
    docs.sort(key=lambda doc: doc.get("createdAt", 0))
    return docs


def get_model(model_id: str) -> dict:
    doc = _read(_path(model_id))
    if doc is None:
        raise HTTPException(status_code=404, detail=f"User model not found: {model_id}")
    return doc


# --- expression parsing -------------------------------------------------------

def parse_energy(kind: str, expression: str):
    """Parse an energy expression safely.

    Returns ``(expr, kinematic_symbols, parameter_names)``; raises
    ``HTTPException`` with a readable message on any problem.
    """
    if kind not in KINEMATIC_SYMBOLS:
        raise HTTPException(status_code=400, detail=f"Unknown model kind: {kind}")
    text = str(expression or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Expression is empty.")
    if len(text) > MAX_EXPRESSION_LENGTH:
        raise HTTPException(status_code=400, detail=f"Expression too long (max {MAX_EXPRESSION_LENGTH} characters).")
    if "__" in text:
        raise HTTPException(status_code=400, detail="Expression contains a forbidden token.")
    if re.search(r"[\[\]{}@$&|<>=!;\\]", text):
        raise HTTPException(status_code=400, detail="Expression contains unsupported characters.")

    kin_names = KINEMATIC_SYMBOLS[kind]
    local_dict = {name: sp.Symbol(name) for name in kin_names}

    # Friendly error for calls to non-whitelisted functions.
    for called in set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)):
        if called not in SAFE_FUNCTIONS:
            allowed = ", ".join(sorted(k for k, v in SAFE_FUNCTIONS.items() if not isinstance(v, sp.Basic)))
            raise HTTPException(status_code=400, detail=f"Unknown function '{called}'. Allowed functions: {allowed}.")

    # Pre-create parameter symbols for every identifier that is neither a
    # kinematic variable nor a whitelisted function, so nothing falls through
    # to sympy's global namespace.
    for token in set(_IDENTIFIER.findall(text)):
        if token in local_dict or token in SAFE_FUNCTIONS:
            continue
        local_dict[token] = sp.Symbol(token)

    try:
        expr = parse_expr(
            text,
            local_dict={**SAFE_FUNCTIONS, **local_dict},
            # Only the constructors sympy's own tokenizer emits — everything a
            # user writes resolves through local_dict or becomes a Symbol.
            global_dict={"Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational, "Symbol": sp.Symbol},
            transformations=standard_transformations,
            evaluate=True,
        )
    except Exception as exc:  # noqa: BLE001 - report parser errors verbatim
        raise HTTPException(status_code=400, detail=f"Could not parse expression: {exc}") from None

    if expr.atoms(AppliedUndef):
        unknown = ", ".join(sorted({f.func.__name__ for f in expr.atoms(AppliedUndef)}))
        raise HTTPException(status_code=400, detail=f"Unknown function(s): {unknown}. Allowed: {', '.join(sorted(k for k, v in SAFE_FUNCTIONS.items() if not isinstance(v, (sp.Basic,))))}.")
    if not expr.free_symbols:
        raise HTTPException(status_code=400, detail="Expression has no symbols — it must depend on the kinematic variables.")
    if sp.count_ops(expr) > MAX_OPS:
        raise HTTPException(status_code=400, detail=f"Expression too complex (max {MAX_OPS} operations).")

    kin_syms = tuple(local_dict[name] for name in kin_names)
    used_kinematic = expr.free_symbols & set(kin_syms)
    if not used_kinematic:
        raise HTTPException(
            status_code=400,
            detail=f"Expression must use at least one kinematic variable ({', '.join(kin_names)}).",
        )

    param_syms = expr.free_symbols - set(kin_syms)
    if len(param_syms) > MAX_PARAMS:
        raise HTTPException(status_code=400, detail=f"Too many parameters (max {MAX_PARAMS}).")
    if not param_syms:
        raise HTTPException(status_code=400, detail="Expression has no material parameters to calibrate.")

    # Order parameters by first appearance in the text for a stable UI.
    ordered = sorted(param_syms, key=lambda symbol: text.find(symbol.name) if text.find(symbol.name) >= 0 else 1e9)
    return expr, kin_syms, [symbol.name for symbol in ordered]


# --- model materialisation ------------------------------------------------------

def build_model_function(doc: dict):
    """Create a tagged energy function (register_model-compatible) from a doc."""
    kind = doc["kind"]
    expr, kin_syms, detected = parse_energy(kind, doc["expression"])
    params_meta = doc.get("params") or [{"name": name} for name in detected]
    names = [item["name"] for item in params_meta]
    param_symbols = {name: sp.Symbol(name) for name in names}

    if kind == "invariant":
        I1s, I2s = kin_syms

        def energy(I1, I2, params):
            mapping = {I1s: I1, I2s: I2}
            mapping.update({param_symbols[name]: params[name] for name in names})
            return expr.xreplace(mapping)

        energy.model_type = "invariant_based"
    else:
        l1s, l2s, l3s = kin_syms

        def energy(lambda_1, lambda_2, lambda_3, params):
            mapping = {l1s: lambda_1, l2s: lambda_2, l3s: lambda_3}
            mapping.update({param_symbols[name]: params[name] for name in names})
            return expr.xreplace(mapping)

        energy.model_type = "stretch_based"

    energy.__name__ = f"UserModel_{doc.get('id', 'draft')}"
    energy.category = "user"
    energy.formula = rf"\Psi = {sp.latex(expr)}"
    energy.param_names = names
    energy.initial_guess = [float(item.get("initial", 1.0)) for item in params_meta]
    energy.bounds = [
        (
            None if item.get("lower") is None else float(item["lower"]),
            None if item.get("upper") is None else float(item["upper"]),
        )
        for item in params_meta
    ]
    return energy


# --- physics checks ---------------------------------------------------------------

def physics_report(doc: dict) -> dict:
    """Run definition-time sanity checks with the given initial parameters."""
    from ..kinematics import Kinematics
    from ..mechanics import get_deformation_gradient

    warnings = []
    errors = []

    try:
        model = build_model_function(doc)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        return {"errors": [f"Could not build the model: {exc}"], "warnings": []}

    params = dict(zip(model.param_names, model.initial_guess))

    # 1. Energy normalisation at the reference state.
    expr, kin_syms, _ = parse_energy(doc["kind"], doc["expression"])
    reference = {symbol: (3 if symbol.name == "I1" or symbol.name == "I2" else 1) for symbol in kin_syms}
    reference.update({sp.Symbol(name): value for name, value in params.items()})
    try:
        psi_ref = complex(sp.N(expr.xreplace(reference)))
        if not math.isfinite(psi_ref.real) or abs(psi_ref.imag) > 1e-12:
            warnings.append("Ψ is not finite/real at the undeformed state with the initial parameters.")
        elif abs(psi_ref.real) > 1e-6:
            warnings.append(f"Ψ(undeformed) = {psi_ref.real:.3g} ≠ 0 — consider normalising the energy.")
    except Exception:  # noqa: BLE001
        warnings.append("Could not evaluate Ψ at the undeformed state.")

    # 2/3. Symbolic differentiation + finite uniaxial response over a stretch sweep.
    try:
        solver = Kinematics(model, model.param_names)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Symbolic differentiation failed: {exc}")
        return {"errors": errors, "warnings": warnings}

    probe = np.concatenate([np.linspace(0.55, 0.95, 5), np.linspace(1.05, 3.0, 9)])
    stresses = []
    with np.errstate(all="ignore"):
        for lam in probe:
            try:
                P = solver.get_1st_PK_stress(get_deformation_gradient(float(lam), "UT"), params)
                stresses.append(float(P[0, 0]))
            except Exception:  # noqa: BLE001
                stresses.append(float("nan"))
    bad = [f"{lam:.2f}" for lam, value in zip(probe, stresses) if not math.isfinite(value)]
    if bad:
        errors.append(
            f"Uniaxial stress is not finite at stretch λ = {', '.join(bad[:4])}{'…' if len(bad) > 4 else ''} "
            "with the initial parameters — check for divisions by zero, logs of negative arguments, or overflow."
        )
    else:
        tension = stresses[-9:]
        if all(value <= 0 for value in tension):
            warnings.append("Uniaxial tension stress is non-positive over λ ∈ [1.05, 3] with the initial parameters — check parameter signs.")

    return {"errors": errors, "warnings": warnings}


# --- validation / CRUD --------------------------------------------------------------

def validate_payload(payload: dict) -> dict:
    """Parse-and-report for the live preview (no persistence)."""
    kind = payload.get("kind", "invariant")
    expression = payload.get("expression", "")
    expr, _kin, detected = parse_energy(kind, expression)

    report = {
        "ok": True,
        "latex": rf"\Psi = {sp.latex(expr)}",
        "parameters": detected,
        "errors": [],
        "warnings": [],
    }

    params = payload.get("params")
    if params:
        by_name = {item.get("name"): item for item in params if isinstance(item, dict)}
        doc = {
            "kind": kind,
            "expression": expression,
            "params": [
                {
                    "name": name,
                    "initial": by_name.get(name, {}).get("initial", 1.0),
                    "lower": by_name.get(name, {}).get("lower"),
                    "upper": by_name.get(name, {}).get("upper"),
                }
                for name in detected
            ],
        }
        checks = physics_report(doc)
        report["errors"] = checks["errors"]
        report["warnings"] = checks["warnings"]
        report["ok"] = not checks["errors"]
    return report


def _clean_params(detected: list, params) -> list:
    by_name = {item.get("name"): item for item in (params or []) if isinstance(item, dict)}
    cleaned = []
    for name in detected:
        item = by_name.get(name, {})
        try:
            initial = float(item.get("initial", 1.0))
        except (TypeError, ValueError):
            initial = 1.0
        def _bound(value):
            if value is None or value == "":
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        lower = _bound(item.get("lower"))
        upper = _bound(item.get("upper"))
        if lower is not None and upper is not None and lower > upper:
            raise HTTPException(status_code=400, detail=f"Parameter {name}: lower bound exceeds upper bound.")
        if not math.isfinite(initial):
            raise HTTPException(status_code=400, detail=f"Parameter {name}: initial value must be finite.")
        cleaned.append({"name": name, "initial": initial, "lower": lower, "upper": upper})
    return cleaned


def save_model(payload: dict) -> dict:
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Model name is required.")
    if len(name) > 60:
        raise HTTPException(status_code=400, detail="Model name is too long (max 60 characters).")

    kind = payload.get("kind", "invariant")
    expression = str(payload.get("expression") or "").strip()
    _expr, _kin, detected = parse_energy(kind, expression)
    params = _clean_params(detected, payload.get("params"))

    now = time.time()
    model_id = payload.get("id")
    existing = _read(_path(model_id)) if model_id else None
    if not model_id or existing is None:
        model_id = uuid.uuid4().hex[:12]

    document = {
        "version": 1,
        "id": model_id,
        "name": name,
        "kind": kind,
        "expression": expression,
        "params": params,
        "notes": str(payload.get("notes") or "")[:500],
        "createdAt": existing.get("createdAt", now) if existing else now,
        "updatedAt": now,
    }

    checks = physics_report(document)
    if checks["errors"]:
        raise HTTPException(status_code=400, detail=" ".join(checks["errors"]))

    path = _path(model_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(document, ensure_ascii=False, indent=1))
    tmp.replace(path)
    return {
        "id": model_id,
        "key": USER_MODEL_PREFIX + model_id,
        "warnings": checks["warnings"],
    }


def delete_model(model_id: str) -> dict:
    path = _path(model_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"User model not found: {model_id}")
    path.unlink()
    return {"deleted": model_id}


def list_payload() -> dict:
    items = []
    for doc in all_models():
        try:
            expr, _kin, _names = parse_energy(doc["kind"], doc["expression"])
            latex = rf"\Psi = {sp.latex(expr)}"
        except HTTPException:
            latex = ""
        items.append({
            "id": doc["id"],
            "key": USER_MODEL_PREFIX + doc["id"],
            "name": doc["name"],
            "kind": doc["kind"],
            "expression": doc["expression"],
            "latex": latex,
            "params": doc.get("params", []),
            "notes": doc.get("notes", ""),
            "createdAt": doc.get("createdAt", 0),
        })
    return {"models": items}
