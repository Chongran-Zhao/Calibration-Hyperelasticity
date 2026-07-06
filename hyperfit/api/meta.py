"""Presentation metadata: labels, references and LaTeX for the web UI."""

from __future__ import annotations

import re

MODE_LABELS = {
    "UT": "Uniaxial Tension",
    "UC": "Uniaxial Compression",
    "ET": "Equibiaxial Tension",
    "PS": "Pure Shear",
    "SS": "Simple Shear",
    "CSS": "Compound Simple Shear",
    "BT": "Biaxial Tension",
}

SOURCE_META = {
    "Budday_2017": {
        "name": "Budday et al. 2017",
        "reference": "Budday et al. (2017)",
        "url": "https://scholar.google.com/scholar?q=Budday+2017+mechanical+properties+brain+tissue",
    },
    "James_1975": {
        "name": "James et al. 1975",
        "reference": "James, Green and Simpson (1975)",
        "url": "https://scholar.google.com/scholar?q=James+Green+Simpson+1975+strain+energy+functions+rubber",
    },
    "Jones_1975": {
        "name": "Jones and Treloar 1975",
        "reference": "Jones and Treloar (1975)",
        "url": "https://scholar.google.com/scholar?q=Jones+Treloar+1975+biaxial+tension+rubber",
    },
    "Katashima_2012": {
        "name": "Katashima et al. 2012",
        "reference": "Katashima et al. (2012)",
        "url": "https://scholar.google.com/scholar?q=Katashima+2012+biaxial+stretching+polymer+gel",
    },
    "Kawabata_1981": {
        "name": "Kawabata et al. 1981",
        "reference": "Kawabata et al. (1981)",
        "url": "https://scholar.google.com/scholar?q=Kawabata+1981+biaxial+tensile+properties+rubber",
    },
    "Kawamura_2001": {
        "name": "Kawamura et al. 2001",
        "reference": "Kawamura et al. (2001)",
        "url": "https://scholar.google.com/scholar?q=Kawamura+2001+biaxial+tension+elastomer",
    },
    "Meunier_2008": {
        "name": "Meunier et al. 2008",
        "reference": "Meunier et al. (2008)",
        "url": "https://scholar.google.com/scholar?q=Meunier+2008+mechanical+properties+skin+biaxial",
    },
    "Treloar_1944": {
        "name": "Treloar 1944",
        "reference": "Treloar (1944)",
        "url": "https://scholar.google.com/scholar?q=Treloar+1944+stress+strain+data+rubber",
    },
}

MODEL_META = {
    "NeoHookean": {
        "name": "Neo-Hookean",
        "reference": "Rivlin (1948); Mooney (1940)",
        "referenceUrl": "https://doi.org/10.1098/rsta.1948.0024",
    },
    "MooneyRivlin": {
        "name": "Mooney-Rivlin",
        "reference": "Mooney (1940); Rivlin (1948)",
        "referenceUrl": "https://doi.org/10.1063/1.1712836",
    },
    "Yeoh": {
        "name": "Yeoh",
        "reference": "Yeoh (1993)",
        "referenceUrl": "https://doi.org/10.5254/1.3538343",
    },
    "ArrudaBoyce": {
        "name": "Arruda-Boyce",
        "reference": "Arruda and Boyce (1993)",
        "referenceUrl": "https://doi.org/10.1016/0022-5096(93)90013-6",
    },
    "ZhanGaussian": {
        "name": "Zhan-Gaussian",
        "reference": "Zhan et al. (2023)",
        "referenceUrl": "https://doi.org/10.1016/j.jmps.2022.105156",
    },
    "ZhanNonGaussian": {
        "name": "Zhan-Non-Gaussian",
        "reference": "Zhan et al. (2023)",
        "referenceUrl": "https://doi.org/10.1016/j.jmps.2022.105156",
    },
    "Ogden": {
        "name": "Ogden",
        "reference": "Ogden (1972)",
        "referenceUrl": "https://doi.org/10.1098/rspa.1972.0026",
    },
    "ModifiedOgden": {
        "name": "Modified-Ogden",
        "reference": "Budday et al. (2017); Ogden (1972)",
        "referenceUrl": "https://doi.org/10.1007/s10237-016-0855-y",
    },
}


def as_text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def mode_family(mode_raw: str) -> str:
    for prefix in ("CSS", "BT", "UT", "UC", "ET", "PS", "SS"):
        if mode_raw.startswith(prefix):
            return prefix
    return mode_raw


def _decimal_suffix(suffix: str) -> str:
    """Convert the file-name encoding ``1d502`` to ``1.502``."""
    return re.sub(r"(?<=\d)d(?=\d)", ".", suffix.replace("_", " "))


def mode_label(mode_raw: str) -> str:
    family = mode_family(mode_raw)
    base = MODE_LABELS.get(family, mode_raw)
    if mode_raw == family:
        return base
    suffix = _decimal_suffix(mode_raw[len(family):].strip("_"))
    return f"{base}, {suffix}" if suffix else base


def source_meta(author: str) -> dict:
    fallback = author.replace("_", " ")
    return SOURCE_META.get(author, {"name": fallback, "reference": fallback, "url": ""})


def _title_words(value: str) -> str:
    return " ".join(part.capitalize() for part in value.split("_"))


def budday_mode_meta(mode_raw: str) -> dict | None:
    """UI metadata for the Budday brain-tissue naming scheme."""
    if mode_raw.startswith("CSS_"):
        match = re.match(r"CSS_(.+)_lambda_(.+)$", mode_raw)
        if not match:
            return None
        region, lam = match.groups()
        lam = re.sub(r"(?<=\d)d(?=\d)", ".", lam)
        return {
            "tissueRegion": region,
            "tissueRegionLabel": _title_words(region),
            "loadingLabel": "Compound Simple Shear",
            "shortLabel": f"CSS, lambda = {lam}",
        }
    if mode_raw.startswith("SS_"):
        region = mode_raw[len("SS_"):]
        return {
            "tissueRegion": region,
            "tissueRegionLabel": _title_words(region),
            "loadingLabel": "Simple Shear",
            "shortLabel": "Simple Shear",
        }
    if mode_raw.startswith("UT_C_") or mode_raw.startswith("UT_T_"):
        kind = "Compression" if mode_raw.startswith("UT_C_") else "Tension"
        region = mode_raw[len("UT_C_"):] if kind == "Compression" else mode_raw[len("UT_T_"):]
        return {
            "tissueRegion": region,
            "tissueRegionLabel": _title_words(region),
            "loadingLabel": f"Uniaxial {kind}",
            "shortLabel": f"Uniaxial {kind}",
        }
    return None


def mode_ui_meta(mode_raw: str) -> dict:
    return budday_mode_meta(mode_raw) or {}


def mode_short_label(mode_raw: str) -> str:
    family = mode_family(mode_raw)
    suffix = _decimal_suffix(mode_raw[len(family):].strip("_"))
    budday = budday_mode_meta(mode_raw)
    if budday:
        return budday["shortLabel"]
    if family == "BT":
        match = re.search(r"lambda\s+([0-9.]+)", suffix)
        return f"BT, fixed lambda2 = {match.group(1)}" if match else "BT"
    if family == "CSS":
        tissue = suffix.split("lambda")[0].strip()
        match = re.search(r"lambda\s+([0-9.]+)", suffix)
        parts = ["CSS"]
        if tissue:
            parts.append(tissue.title())
        if match:
            parts.append(f"lambda = {match.group(1)}")
        return ", ".join(parts)
    return MODE_LABELS.get(family, mode_raw)


def stress_display(stress_type: str) -> dict:
    if stress_type == "cauchy":
        return {
            "label": "Cauchy stress",
            "symbol": r"\boldsymbol{\sigma}",
            "plain": "Cauchy stress σ",
        }
    return {
        "label": "First Piola-Kirchhoff stress",
        "symbol": r"\boldsymbol{P}",
        "plain": "First Piola-Kirchhoff stress P",
    }


def axis_labels(mode: str, stress_type: str) -> tuple[str, str]:
    if mode in ("SS", "CSS"):
        return "Shear strain γ [-]", "Shear stress P₁₂ [MPa]"
    if mode == "BT":
        stress_label = "Cauchy stress σ₁₁ [MPa]" if stress_type == "cauchy" else "Nominal stress P₁₁ [MPa]"
        return "Variable stretch λ₁ [-]", stress_label
    stress_label = "Cauchy stress σ₁₁ [MPa]" if stress_type == "cauchy" else "Nominal stress P₁₁ [MPa]"
    return "Stretch λ [-]", stress_label


def axis_symbols(mode: str, stress_type: str) -> dict:
    stress_symbol = r"\sigma_{11}" if stress_type == "cauchy" else r"P_{11}"
    if mode in ("SS", "CSS"):
        return {"x": r"\gamma", "y": r"P_{12}"}
    if mode == "BT":
        return {"x": r"\lambda_1", "y": stress_symbol}
    return {"x": r"\lambda", "y": stress_symbol}


def mode_tensor_expressions(family: str, mode_raw: str, fixed_stretch: float | None = None) -> dict:
    """LaTeX for the deformation gradient and PK1 structure of a mode."""
    if family in ("UT", "UC"):
        return {
            "deformationGradient": r"\boldsymbol{F}=\begin{bmatrix}\lambda&0&0\\0&\lambda^{-1/2}&0\\0&0&\lambda^{-1/2}\end{bmatrix}",
            "firstPkStress": r"\boldsymbol{P}=\begin{bmatrix}P_{11}&0&0\\0&0&0\\0&0&0\end{bmatrix}",
            "component": r"P_{11}",
        }
    if family == "ET":
        return {
            "deformationGradient": r"\boldsymbol{F}=\begin{bmatrix}\lambda&0&0\\0&\lambda&0\\0&0&\lambda^{-2}\end{bmatrix}",
            "firstPkStress": r"\boldsymbol{P}=\begin{bmatrix}P_{11}&0&0\\0&P_{22}&0\\0&0&0\end{bmatrix}",
            "component": r"P_{11}",
        }
    if family == "PS":
        return {
            "deformationGradient": r"\boldsymbol{F}=\begin{bmatrix}\lambda&0&0\\0&1&0\\0&0&\lambda^{-1}\end{bmatrix}",
            "firstPkStress": r"\boldsymbol{P}=\begin{bmatrix}P_{11}&0&0\\0&P_{22}&0\\0&0&0\end{bmatrix}",
            "component": r"P_{11}",
        }
    if family == "SS":
        return {
            "deformationGradient": r"\boldsymbol{F}=\begin{bmatrix}1&\gamma&0\\0&1&0\\0&0&1\end{bmatrix}",
            "firstPkStress": r"\boldsymbol{P}=\begin{bmatrix}0&P_{12}&0\\0&0&0\\0&0&0\end{bmatrix}",
            "component": r"P_{12}",
        }
    if family == "CSS":
        lam = None
        match = re.search(r"lambda_(.+)$", mode_raw)
        if match:
            lam = re.sub(r"(?<=\d)d(?=\d)", ".", match.group(1))
        lam_symbol = lam or r"\lambda"
        return {
            "deformationGradient": rf"\boldsymbol{{F}}=\begin{{bmatrix}}{lam_symbol}&\gamma&0\\0&{lam_symbol}^{{-1/2}}&0\\0&0&{lam_symbol}^{{-1/2}}\end{{bmatrix}}",
            "firstPkStress": r"\boldsymbol{P}=\begin{bmatrix}0&P_{12}&0\\0&0&0\\0&0&0\end{bmatrix}",
            "component": r"P_{12}",
        }
    if family == "BT":
        lam2 = f"{fixed_stretch:.3g}" if fixed_stretch is not None else r"\lambda_2"
        return {
            "deformationGradient": rf"\boldsymbol{{F}}=\begin{{bmatrix}}\lambda_1&0&0\\0&{lam2}&0\\0&0&(\lambda_1 {lam2})^{{-1}}\end{{bmatrix}}",
            "firstPkStress": r"\boldsymbol{P}=\begin{bmatrix}P_{11}&0&0\\0&P_{22}&0\\0&0&0\end{bmatrix}",
            "component": r"P_{11},\,P_{22}",
        }
    return {
        "deformationGradient": r"\boldsymbol{F}",
        "firstPkStress": r"\boldsymbol{P}",
        "component": r"P",
    }
