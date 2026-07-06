"""Capture a numerical baseline of the calibration core.

Runs stress predictions, objective evaluations and representative fits for
every model family against packaged datasets, and writes the numbers to JSON.
Used to prove numerical equivalence across refactors:

    python3 tests/capture_baseline.py /path/to/baseline.json

The script works against either the legacy flat ``src/`` layout or the
refactored ``hyperfit`` package, whichever is importable.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from hyperfit.models import MaterialModels
    from hyperfit.kinematics import Kinematics
    from hyperfit.optimizer import MaterialOptimizer
    from hyperfit.network import ParallelNetwork
    from hyperfit.datasets import load_experimental_data_h5

    FLAVOR = "hyperfit"
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))
    from material_models import MaterialModels
    from kinematics import Kinematics
    from optimization import MaterialOptimizer
    from parallel_springs import ParallelNetwork
    from utils import load_experimental_data_h5

    FLAVOR = "legacy-src"

DATA_H5 = str(ROOT / "data" / "data.h5")

STRAINS = [
    "Seth-Hill",
    "Hencky",
    "Curnier-Rakotomanana",
    "Curnier-Zysset",
    "Darijani-Naghdabadi",
]


def build_models():
    """Return list of (name, model_func) covering every model family."""
    models = [
        ("NeoHookean", MaterialModels.NeoHookean),
        ("MooneyRivlin", MaterialModels.MooneyRivlin),
        ("Yeoh", MaterialModels.Yeoh),
        ("ArrudaBoyce", MaterialModels.ArrudaBoyce),
        ("ModifiedOgden", MaterialModels.ModifiedOgden),
        ("Ogden3", MaterialModels.create_ogden_model(3)),
        ("ZhanGaussian", MaterialModels.ZhanGaussian),
        ("ZhanNonGaussian", MaterialModels.ZhanNonGaussian),
    ]
    for strain in STRAINS:
        models.append((f"Hill_{strain}", MaterialModels.create_hill_model(strain)))

    net_a = ParallelNetwork()
    net_a.add_model(MaterialModels.NeoHookean, "matrix")
    net_a.add_model(MaterialModels.ArrudaBoyce, "chain")
    models.append(("Network_NH_AB", net_a))

    net_b = ParallelNetwork()
    net_b.add_model(MaterialModels.ZhanGaussian, "zhan")
    net_b.add_model(MaterialModels.NeoHookean, "nh")
    models.append(("Network_Zhan_NH", net_b))
    return models


DATASET_GROUPS = {
    "treloar": [
        {"author": "Treloar_1944", "mode": "UT"},
        {"author": "Treloar_1944", "mode": "ET"},
        {"author": "Treloar_1944", "mode": "PS"},
    ],
    "jones_bt": [
        {"author": "Jones_1975", "mode": "BT_lambda_1d502"},
        {"author": "Jones_1975", "mode": "UT"},
    ],
    "budday": [
        {"author": "Budday_2017", "mode": "CSS_corona_radiata_lambda_0d90"},
        {"author": "Budday_2017", "mode": "SS_corona_radiata"},
        {"author": "Budday_2017", "mode": "UT_C_corona_radiata"},
    ],
    "katashima": [
        {"author": "Katashima_2012", "mode": "BT"},
        {"author": "Katashima_2012", "mode": "PS"},
    ],
    "kawabata_et": [
        {"author": "Kawabata_1981", "mode": "ET"},
    ],
    "james_ut": [
        {"author": "James_1975", "mode": "UT"},
    ],
}


def param_vectors(model):
    """Initial guess plus deterministic perturbations clipped into bounds."""
    guess = np.asarray(model.initial_guess, dtype=float)
    bounds = list(model.bounds)
    vectors = {"initial": guess}
    for tag, factor in (("scaled_down", 0.93), ("scaled_up", 1.07)):
        vec = guess * factor
        for j, b in enumerate(bounds):
            lo = -np.inf if b is None or b[0] is None else b[0]
            hi = np.inf if b is None or b[1] is None else b[1]
            vec[j] = float(np.clip(vec[j], lo, hi))
        vectors[tag] = vec
    return vectors


def clean(value):
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [clean(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def capture():
    out = {"flavor": FLAVOR, "objective": {}, "stress": {}, "fits": {}, "timing": {}}
    models = build_models()
    groups = {
        name: load_experimental_data_h5(cfgs, DATA_H5, announce=False)
        for name, cfgs in DATASET_GROUPS.items()
    }

    # --- Objective values + stress predictions at fixed parameter vectors ---
    for model_name, model in models:
        solver = Kinematics(model, model.param_names)
        for group_name, datasets in groups.items():
            optimizer = MaterialOptimizer(solver, datasets)
            key = f"{model_name}::{group_name}"
            entry = {}
            for tag, vec in param_vectors(model).items():
                t0 = time.perf_counter()
                loss = float(optimizer._objective_function(vec))
                dt = time.perf_counter() - t0
                residuals = optimizer._residuals(vec)
                r2_total, r2_avg = optimizer.compute_r2(vec)
                entry[tag] = {
                    "params": clean(vec),
                    "objective": loss,
                    "residual_norm": float(np.linalg.norm(residuals)),
                    "r2_total": float(r2_total),
                    "r2_avg": float(r2_avg),
                }
                if tag == "initial":
                    out["timing"].setdefault(key, dt)
            out["objective"][key] = entry

        # Full stress tensors on one representative deformation set
        datasets = groups["treloar"] if model_name != "Hill_Curnier-Zysset" else groups["budday"]
        params = dict(zip(model.param_names, param_vectors(model)["initial"]))
        tensors = {"PK1": [], "cauchy": []}
        for ds in datasets:
            for F in np.asarray(ds["F_list"])[::5]:
                tensors["PK1"].append(clean(solver.get_1st_PK_stress(F, params)))
                tensors["cauchy"].append(clean(solver.get_Cauchy_stress(F, params)))
        out["stress"][model_name] = tensors

    # --- Representative fits ---
    fit_specs = [
        ("NeoHookean_treloarUT_lbfgsb", MaterialModels.NeoHookean, groups["treloar"][:1], "L-BFGS-B", 300),
        ("ArrudaBoyce_treloar_lbfgsb", MaterialModels.ArrudaBoyce, groups["treloar"], "L-BFGS-B", 300),
        ("Ogden3_treloar_lbfgsb", MaterialModels.create_ogden_model(3), groups["treloar"], "L-BFGS-B", 300),
        ("HillSethHill_kawabata_trf", MaterialModels.create_hill_model("Seth-Hill"), groups["kawabata_et"], "trf", 200),
        ("MooneyRivlin_jones_trf", MaterialModels.MooneyRivlin, groups["jones_bt"], "trf", 200),
        ("ZhanNonGaussian_james_lbfgsb", MaterialModels.ZhanNonGaussian, groups["james_ut"], "L-BFGS-B", 100),
    ]
    net = ParallelNetwork()
    net.add_model(MaterialModels.NeoHookean, "matrix")
    net.add_model(MaterialModels.ArrudaBoyce, "chain")
    fit_specs.append(("Network_treloarUT_lbfgsb", net, groups["treloar"][:1], "L-BFGS-B", 150))

    for fit_name, model, datasets, method, max_iter in fit_specs:
        solver = Kinematics(model, model.param_names)
        optimizer = MaterialOptimizer(solver, datasets)
        t0 = time.perf_counter()
        result = optimizer.fit(
            list(model.initial_guess),
            list(model.bounds),
            method=method,
            max_iter=max_iter,
            r2_target=0.995,
            abs_tol=1e-6,
            rel_tol=1e-4,
            max_loss=0.05,
        )
        out["fits"][fit_name] = {
            "x": clean(np.asarray(result.x)),
            "loss": float(result.fun),
            "r2_total": float(result.r2_total),
            "r2_avg": float(result.r2_avg),
            "success": bool(result.success),
            "seconds": round(time.perf_counter() - t0, 3),
        }

    return out


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests" / "baseline.json"
    result = capture()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(f"\n[{FLAVOR}] baseline written to {target}")
