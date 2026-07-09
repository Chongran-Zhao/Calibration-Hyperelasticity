"""Experimental dataset loading (HDF5 and plain-text).

Each loaded dataset is a dict with the keys the rest of the package relies on:

``author, tag, mode, mode_raw, stress_type, stretch, stress_exp, F_list``
plus optionally ``stretch_secondary``, ``bt_component`` (``"diff"`` when the
experiment reports sigma11 - sigma22) and ``component`` (``"11"``/``"22"``
when a source is split into per-component entries).
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .mechanics import get_deformation_gradient

try:
    import h5py
except ImportError:  # pragma: no cover - h5py is a hard dep in practice
    h5py = None

logger = logging.getLogger(__name__)


class DataLoadError(RuntimeError):
    """Raised when no experimental dataset could be loaded."""


def _parse_mode(mode_raw):
    for family in ("BT", "UT", "UC", "CSS", "SS", "ET", "PS"):
        if mode_raw.startswith(family):
            return family
    return mode_raw


def _text_data_path(cfg, data_root):
    return os.path.join(data_root, cfg["author"], f"{cfg['mode']}.txt")


def _source_quirks(author, mode):
    """Per-source measurement conventions that the files do not encode.

    Jones & Treloar (1975) biaxial data reports the Cauchy stress difference
    sigma11 - sigma22, which is the pressure-independent observable.
    """
    if author == "Jones_1975" and mode == "BT":
        return "cauchy", "diff"
    return None, None


def _load_text_dataset(cfg, data_root):
    author = cfg["author"]
    mode_raw = cfg["mode"]
    mode = _parse_mode(mode_raw)

    file_path = _text_data_path(cfg, data_root)
    if not os.path.exists(file_path):
        logger.warning("Dataset file not found: %s", file_path)
        return None

    raw_data = np.loadtxt(file_path)

    stress_type = "PK1"
    quirk_stress, bt_component = _source_quirks(author, mode)
    if quirk_stress:
        stress_type = quirk_stress

    entry = dataset_from_columns(author, mode_raw, raw_data, stress_type=stress_type)
    if bt_component:
        entry["bt_component"] = bt_component
    return entry


def dataset_from_columns(author, mode_raw, raw_data, stress_type="PK1", tag=None):
    """Build a dataset entry from a raw numeric column array.

    Column conventions (shared by text files and user uploads):
    ``UT/UC/ET/PS``: stretch, stress; ``SS``: gamma, P12;
    ``BT``: lambda1, lambda2, P11 [, P22].
    """
    mode = _parse_mode(mode_raw)
    raw_data = np.asarray(raw_data, dtype=float)
    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(1, -1)

    if mode == "BT":
        if raw_data.shape[1] not in (3, 4):
            raise ValueError(f"BT data must have 3 or 4 columns, got {raw_data.shape[1]}")
        stretch_list = raw_data[:, 0]
        stretch_secondary = raw_data[:, 1]
        if raw_data.shape[1] == 4:
            stress_exp_list = np.column_stack([raw_data[:, 2], raw_data[:, 3]])
        else:
            stress_exp_list = raw_data[:, 2]

        f_tensors = [
            get_deformation_gradient((lam1, lam2), mode)
            for lam1, lam2 in zip(stretch_list, stretch_secondary)
        ]
        return {
            "author": author,
            "tag": tag or f"{author}_{mode_raw}",
            "mode": mode,
            "mode_raw": mode_raw,
            "stress_type": stress_type,
            "stretch": stretch_list,
            "stretch_secondary": stretch_secondary,
            "stress_exp": stress_exp_list,
            "F_list": np.array(f_tensors),
        }

    if raw_data.shape[1] < 2:
        raise ValueError(f"{mode} data must have 2 columns, got {raw_data.shape[1]}")
    stretch_list = raw_data[:, 0]
    f_tensors = [get_deformation_gradient(lam, mode) for lam in stretch_list]
    return {
        "author": author,
        "tag": tag or f"{author}_{mode_raw}",
        "mode": mode,
        "mode_raw": mode_raw,
        "stress_type": stress_type,
        "stretch": stretch_list,
        "stress_exp": raw_data[:, 1],
        "F_list": np.array(f_tensors),
    }


def _load_h5_dataset(cfg, h5f):
    author = cfg["author"]
    mode_raw = cfg["mode"]
    mode = _parse_mode(mode_raw)

    if author not in h5f or mode_raw not in h5f[author]:
        logger.warning("HDF5 group not found: %s/%s", author, mode_raw)
        return None

    grp = h5f[author][mode_raw]
    F_list = grp["F"][()]
    stress_tensor = grp["stress"][()]
    stress_type = grp.attrs.get("stress_type", "PK1")
    if isinstance(stress_type, bytes):
        stress_type = stress_type.decode("utf-8")
    quirk_stress, bt_component = _source_quirks(author, mode)
    if quirk_stress:
        stress_type = quirk_stress

    stretch = grp["stretch"][()] if "stretch" in grp else F_list[:, 0, 0]
    stretch_secondary = grp["stretch_secondary"][()] if "stretch_secondary" in grp else None

    # Katashima (2012) reports P11 and P22 on different stretch grids, so the
    # source is split into two per-component entries.
    if author == "Katashima_2012" and mode in ("BT", "PS") and stretch_secondary is not None:
        entries = []
        for component, lam_list, stress_list in (
            ("11", stretch, stress_tensor[:, 0, 0]),
            ("22", stretch_secondary, stress_tensor[:, 1, 1]),
        ):
            f_tensors = []
            for lam in lam_list:
                if mode == "BT":
                    lam2 = 0.5 + 0.5 * lam
                    f_tensors.append(get_deformation_gradient((lam, lam2), "BT"))
                else:
                    f_tensors.append(get_deformation_gradient(lam, "PS"))
            entry = {
                "author": author,
                "tag": f"{author}_{mode_raw}_P{component}",
                "mode": mode,
                "mode_raw": mode_raw,
                "stress_type": stress_type,
                "stretch": lam_list,
                "stress_exp": stress_list,
                "F_list": np.array(f_tensors),
                "component": component,
            }
            if bt_component:
                entry["bt_component"] = bt_component
            entries.append(entry)
        return entries

    if mode in ("SS", "CSS"):
        stress_exp = stress_tensor[:, 0, 1]
    elif mode == "BT":
        stress_exp = np.column_stack([stress_tensor[:, 0, 0], stress_tensor[:, 1, 1]])
    elif mode == "PS" and np.any(np.abs(stress_tensor[:, 1, 1]) > 1e-12):
        stress_exp = np.column_stack([stress_tensor[:, 0, 0], stress_tensor[:, 1, 1]])
    else:
        stress_exp = stress_tensor[:, 0, 0]

    entry = {
        "author": author,
        "tag": f"{author}_{mode_raw}",
        "mode": mode,
        "mode_raw": mode_raw,
        "stress_type": stress_type,
        "stretch": stretch,
        "stress_exp": stress_exp,
        "F_list": F_list,
    }
    if stretch_secondary is not None:
        entry["stretch_secondary"] = stretch_secondary
    if bt_component:
        entry["bt_component"] = bt_component
    return entry


def _collect(configs, loader):
    all_tests = []
    for cfg in configs:
        entry = loader(cfg)
        if entry:
            if isinstance(entry, list):
                all_tests.extend(entry)
            else:
                all_tests.append(entry)
    return all_tests


def load_experimental_data(configs):
    """Load datasets from ``data/data.h5`` (preferred) or per-author text files.

    The data root defaults to ``data`` relative to the working directory and
    can be overridden with the ``CALIBRATION_DATA_DIR`` environment variable.

    Raises:
        DataLoadError: If none of the requested configurations could be loaded.
    """
    logger.info("Loading %d dataset configuration(s)", len(configs))
    data_root = os.environ.get("CALIBRATION_DATA_DIR", "data")
    data_h5_path = os.path.join(data_root, "data.h5")

    if h5py and os.path.exists(data_h5_path):
        return load_experimental_data_h5(configs, data_h5_path)

    all_tests = _collect(configs, lambda cfg: _load_text_dataset(cfg, data_root))
    if not all_tests:
        raise DataLoadError(f"No valid data loaded from '{data_root}'.")
    logger.info("Loaded %d datasets", len(all_tests))
    return all_tests


def load_experimental_data_h5(configs, data_h5_path, announce=None):
    """Load datasets from a specific HDF5 file.

    Raises:
        DataLoadError: If none of the requested configurations could be loaded.
    """
    if h5py is None:
        raise DataLoadError("h5py is not installed; cannot read HDF5 datasets.")
    with h5py.File(data_h5_path, "r") as h5f:
        all_tests = _collect(configs, lambda cfg: _load_h5_dataset(cfg, h5f))

    if not all_tests:
        raise DataLoadError(f"No valid data loaded from '{data_h5_path}'.")
    logger.info("Loaded %d datasets from %s", len(all_tests), data_h5_path)
    return all_tests
