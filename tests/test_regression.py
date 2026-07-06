"""Numerical regression test against the pre-refactor baseline.

``baseline.json`` was captured from the original flat ``src/`` implementation
(see ``capture_baseline.py``). This test recomputes every fingerprint with the
current package and compares:

* objective values / residual norms / R^2 at fixed parameter vectors and
  full stress tensors: rtol 1e-9 (pointwise evaluations are ULP-identical up
  to numpy's SIMD-vs-libm difference in transcendental functions);
* calibrated parameters and losses of full fits: rtol 1e-5 (iterative
  amplification of the same ULP noise; same minimum, same quality).

Run directly (``python3 tests/test_regression.py``) or via pytest.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture_baseline import capture  # noqa: E402

BASELINE = Path(__file__).parent / "baseline.json"

POINTWISE_RTOL = 1e-9
FIT_RTOL = 1e-5


def _check(path, reference, current, rtol, failures):
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.shape != cur.shape:
        failures.append(f"{path}: shape {ref.shape} != {cur.shape}")
        return
    ok = np.isclose(ref, cur, rtol=rtol, atol=1e-12, equal_nan=True)
    if not np.all(ok):
        worst = np.max(np.abs(ref - cur) / np.maximum(np.abs(ref), 1e-300))
        failures.append(f"{path}: max rel diff {worst:.3e} (rtol {rtol})")


def compare_to_baseline():
    reference = json.loads(BASELINE.read_text())
    current = capture()
    failures = []

    for key, ref_entry in reference["objective"].items():
        cur_entry = current["objective"].get(key)
        if cur_entry is None:
            failures.append(f"objective/{key}: missing")
            continue
        for tag, ref_vals in ref_entry.items():
            for field in ("objective", "residual_norm", "r2_total", "r2_avg"):
                _check(
                    f"objective/{key}/{tag}/{field}",
                    ref_vals[field],
                    cur_entry[tag][field],
                    POINTWISE_RTOL,
                    failures,
                )

    for model, ref_tensors in reference["stress"].items():
        cur_tensors = current["stress"].get(model)
        if cur_tensors is None:
            failures.append(f"stress/{model}: missing")
            continue
        for kind in ("PK1", "cauchy"):
            _check(f"stress/{model}/{kind}", ref_tensors[kind], cur_tensors[kind], POINTWISE_RTOL, failures)

    for fit, ref_fit in reference["fits"].items():
        cur_fit = current["fits"].get(fit)
        if cur_fit is None:
            failures.append(f"fits/{fit}: missing")
            continue
        _check(f"fits/{fit}/x", ref_fit["x"], cur_fit["x"], FIT_RTOL, failures)
        _check(f"fits/{fit}/loss", ref_fit["loss"], cur_fit["loss"], FIT_RTOL, failures)
        _check(f"fits/{fit}/r2_total", ref_fit["r2_total"], cur_fit["r2_total"], FIT_RTOL, failures)

    return failures


def test_regression():
    failures = compare_to_baseline()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    problems = compare_to_baseline()
    if problems:
        print(f"REGRESSION: {len(problems)} mismatches")
        for p in problems:
            print(" ", p)
        sys.exit(1)
    print("Regression check passed: current implementation matches the baseline.")
