"""Domain-specific text formatting helpers for GUI labels and plots."""

from app.catalog import MODE_DISPLAY_MAP


def format_mode_label(mode_key):
    if mode_key.startswith("BT_lambda_"):
        lam2 = mode_key.replace("BT_lambda_", "").replace("d", ".")
        return f"Biaxial Tension, λ₂={lam2}"
    if mode_key.startswith("UT_T_"):
        return "Uniaxial Tension"
    if mode_key.startswith("UT_C_"):
        return "Uniaxial Compression"
    if mode_key.startswith("CSS_") and "_lambda_" in mode_key:
        lam = mode_key.split("_lambda_")[-1].replace("d", ".")
        return f"Simple shear under λ₁ = {lam}"
    if mode_key.startswith("UT_"):
        return "Uniaxial (Tension/Compression)"
    if mode_key.startswith("SS_"):
        return "Simple Shear"
    return MODE_DISPLAY_MAP.get(mode_key, mode_key)


def format_component_label(component_key):
    return component_key.replace("_", " ").title()


def extract_component_from_mode(mode_key, prefixes):
    for prefix in prefixes:
        if prefix == "CSS_" and mode_key.startswith(prefix) and "_lambda_" in mode_key:
            return mode_key[len(prefix):].split("_lambda_")[0]
        if mode_key.startswith(prefix):
            return mode_key[len(prefix):]
    return None


def get_stress_type_label(stress_type):
    return "Cauchy stress" if stress_type == "cauchy" else "Nominal stress"


def get_bt_component_labels(stress_type):
    if stress_type == "cauchy":
        return r"$\sigma_{11}$", r"$\sigma_{22}$"
    return r"$P_{11}$", r"$P_{22}$"


def get_component_label(stress_type, component):
    comp_11, comp_22 = get_bt_component_labels(stress_type)
    if component == "22":
        return comp_22
    if component == "11":
        return comp_11
    return get_uniaxial_component_label(stress_type)


def get_bt_diff_label(stress_type):
    if stress_type == "cauchy":
        return r"$\sigma_{11}-\sigma_{22}$"
    return r"$P_{11}-P_{22}$"


def get_uniaxial_component_label(stress_type):
    return r"$\sigma_{11}$" if stress_type == "cauchy" else r"$P_{11}$"


def get_shear_component_label(stress_type):
    return r"$\sigma_{12}$" if stress_type == "cauchy" else r"$P_{12}$"


def get_mode_xlabel(mode):
    return r"$\gamma$" if mode == "SS" else r"$\lambda_1$"


def choose_xlabel_for_modes(modes):
    if modes and all(m in ("SS", "CSS") for m in modes):
        return r"$\gamma$"
    return r"$\lambda_1$"


def parse_lambda2(mode_raw):
    if mode_raw.startswith("BT_lambda_"):
        return mode_raw.replace("BT_lambda_", "").replace("d", ".")
    return None
