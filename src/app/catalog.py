"""Shared labels, references, and display names for the desktop GUI."""

MODE_DISPLAY_MAP = {
    "UT": "Uniaxial Tension",
    "UC": "Uniaxial Compression",
    "ET": "Equibiaxial Tension",
    "PS": "Pure Shear",
    "SS": "Simple Shear",
    "CSS": "Compound Simple Shear",
    "BT": "Biaxial Tension",
}

COMPONENT_AUTHOR_CONFIG = {
    "Budday_2017": {
        "component_label": "Region",
        "mode_prefixes": ("SS_", "UT_T_", "UT_C_", "CSS_"),
        "mode_labels": {
            "SS_": "Simple Shear",
            "UT_T_": "Uniaxial Tension",
            "UT_C_": "Uniaxial Compression",
            "CSS_": "Compound Simple Shear",
        },
    }
}

DATASET_REFERENCES = {
    "Treloar_1944": ("Treloar 1944, Rubber Chemistry and Technology", "https://doi.org/10.5254/1.3546701"),
    "Kawabata_1981": ("Kawabata et al. 1981, Macromolecules", "https://doi.org/10.1021/ma50002a032"),
    "Meunier_2008": ("Meunier et al. 2008, Polymer Testing", "https://doi.org/10.1016/j.polymertesting.2008.05.011"),
    "James_1975": ("James et al. 1975, J. Appl. Polym. Sci.", "https://doi.org/10.1002/app.1975.070190723"),
    "Jones_1975": ("Jones & Treloar 1975, J. Phys. D", "https://doi.org/10.1088/0022-3727/8/11/007"),
    "Kawamura_2001": ("Kawamura et al. 2001, Macromolecules", "https://doi.org/10.1021/ma002165y"),
    "Katashima_2012": ("Katashima et al. 2012, Soft Matter", "https://doi.org/10.1039/c2sm25340b"),
    "Budday_2017": ("Budday et al. 2017, Acta Biomaterialia", "https://doi.org/10.1016/j.actbio.2016.10.036"),
}

DATASET_N_GUESS = {
    "Treloar_1944": 10.0,
    "Jones_1975": 6.0,
    "James_1975": 10.0,
    "Kawamura_2001": 8.0,
    "Katashima_2012": 8.0,
    "Kawabata_1981": 6.0,
    "Meunier_2008": 6.0,
}

MODEL_REFERENCES = {
    "NeoHookean": ("Treloar 1943, Rubber Chemistry and Technology", "https://doi.org/10.5254/1.3540158"),
    "MooneyRivlin": ("Mooney 1940, Journal of Applied Physics", "https://doi.org/10.1063/1.1712836"),
    "Yeoh": ("Yeoh 1993, Rubber Chemistry and Technology", "https://doi.org/10.5254/1.3538343"),
    "ArrudaBoyce": ("Arruda & Boyce 1993, J. Mech. Phys. Solids", "https://doi.org/10.1016/0022-5096(93)90013-6"),
    "Ogden": ("Ogden 1972, Proc. R. Soc. A", "https://doi.org/10.1098/rspa.1972.0026"),
    "ModifiedOgden": ("Budday et al. 2017, Acta Biomaterialia", "https://doi.org/10.1016/j.actbio.2016.10.036"),
    "Hill": (
        "Liu, Guan, Zhao, Luo 2024, Computer Methods in Applied Mechanics and Engineering 430:117248",
        "https://doi.org/10.1016/j.cma.2024.117248",
    ),
    "ZhanGaussian": ("Zhan et al. 2022, J. Mech. Phys. Solids", "https://www.sciencedirect.com/science/article/abs/pii/S0022509622003325"),
    "ZhanNonGaussian": ("Zhan et al. 2022, J. Mech. Phys. Solids", "https://www.sciencedirect.com/science/article/abs/pii/S0022509622003325"),
}

MODEL_DISPLAY_NAMES = {
    "ZhanGaussian": "Zhan (Gaussian)",
    "ZhanNonGaussian": "Zhan (non-Gaussian)",
    "ModifiedOgden": "Modified Ogden",
}

MODEL_DISPLAY_LOOKUP = {v: k for k, v in MODEL_DISPLAY_NAMES.items()}
