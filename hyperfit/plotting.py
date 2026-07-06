"""Matplotlib comparison plots of experimental data vs model predictions."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

from .evaluation import experimental_values, predict_curve, predict_observables

logger = logging.getLogger(__name__)

_RC_PARAMS = {
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "font.size": 12,
}

_MARKERS = {"UT": "o", "ET": "s", "PS": "^", "SS": "v", "CSS": "x", "BT": "D"}
_COLORS = {"UT": "blue", "ET": "red", "PS": "green", "SS": "teal", "CSS": "teal", "BT": "purple"}


def calculate_r2(exp_stress, model_stress):
    """Coefficient of determination R^2 = 1 - SS_res / SS_tot."""
    exp_flat = np.asarray(exp_stress).ravel()
    model_flat = np.asarray(model_stress).ravel()
    ss_res = np.sum((exp_flat - model_flat) ** 2)
    ss_tot = np.sum((exp_flat - np.mean(exp_flat)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return 1 - (ss_res / ss_tot)


def _scatter(x, y, label, mode, marker=None):
    plt.scatter(
        x,
        y,
        label=label,
        marker=marker or _MARKERS.get(mode, "o"),
        facecolors="none",
        edgecolors=_COLORS.get(mode, "black"),
        s=60,
        zorder=2,
    )


def _line(x, y, mode, linestyle="-"):
    plt.plot(x, y, color=_COLORS.get(mode, "black"), linestyle=linestyle, linewidth=2, zorder=1)


def plot_comparison(experimental_data, kinematics_solver, fitted_params, title="Model Fit Results", save_path=None):
    """Plot experimental points against model curves, with per-dataset R^2.

    Args:
        experimental_data: List of dataset dicts.
        kinematics_solver: :class:`hyperfit.kinematics.Kinematics` instance.
        fitted_params: Dict of calibrated parameters.
        title: Figure title.
        save_path: Save the figure there instead of showing it interactively.
    """
    with plt.rc_context(_RC_PARAMS):
        plt.figure(figsize=(10, 7))

        for dataset in experimental_data:
            mode = dataset["mode"]
            stress_type = dataset.get("stress_type", "PK1")
            exp_stretch = dataset["stretch"]
            exp_values = experimental_values(dataset)
            bt_diff = dataset.get("bt_component") == "diff" and mode == "BT"
            component = dataset.get("component")

            model_at_exp = predict_observables(kinematics_solver, dataset, fitted_params)
            r2 = calculate_r2(exp_values, model_at_exp)

            stress_label = "Cauchy" if stress_type == "cauchy" else "Nominal"
            label = f"Exp: {dataset['tag']} ({stress_label}, $R^2={r2:.3f}$)"
            sigma = stress_type == "cauchy"

            # Experimental scatter.
            if bt_diff:
                comp_label = r"$\sigma_{11}-\sigma_{22}$" if sigma else r"$P_{11}-P_{22}$"
                _scatter(exp_stretch, exp_values, f"{label} {comp_label}", mode)
            elif component in ("11", "22"):
                sub = component
                comp_label = rf"$\sigma_{{{sub}}}$" if sigma else rf"$P_{{{sub}}}$"
                _scatter(exp_stretch, exp_values, f"{label} {comp_label}", mode)
            elif np.ndim(exp_values) == 1:
                _scatter(exp_stretch, exp_values, label, mode)
            else:
                _scatter(exp_stretch, exp_values[:, 0], f"{label} P11", mode)
                _scatter(exp_stretch, exp_values[:, 1], f"{label} P22", mode, marker="^")

            # Model curve. Per-component sources use the experimental grid
            # (their two components live on different stretch grids).
            if component is not None:
                _line(exp_stretch, model_at_exp, mode)
                continue

            min_lam, max_lam = np.min(exp_stretch), np.max(exp_stretch)
            if mode in ("SS", "CSS"):
                smooth = np.linspace(min_lam, max_lam, 100)
                grid = smooth
            else:
                smooth = np.linspace(1.0, max_lam * 1.05, 100)
                grid = smooth
            if mode == "BT":
                lam2 = float(dataset["stretch_secondary"][0])
                grid = [(lam, lam2) for lam in smooth]

            comps = predict_curve(kinematics_solver, mode, grid, fitted_params, stress_type)
            if bt_diff:
                _line(smooth, comps[0] - comps[1], mode)
            else:
                _line(smooth, comps[0], mode)
                if len(comps) > 1:
                    _line(smooth, comps[1], mode, linestyle="--")

        modes = {d["mode"] for d in experimental_data}
        if modes.issubset({"SS", "CSS"}):
            plt.xlabel(r"Shear strain $\gamma$ [-]")
            plt.ylabel(r"Shear stress $P_{12}$ [MPa]")
        else:
            plt.xlabel(r"Stretch $\lambda$ [-]")
            plt.ylabel("Nominal Stress $P$ [MPa]")
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info("Plot saved to %s", save_path)
            plt.close()
        else:
            plt.show()
