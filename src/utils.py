import numpy as np
import os
import sys

try:
    import h5py
except Exception:
    h5py = None

def get_deformation_gradient(stretch, mode):
    """
    Constructs the deformation gradient tensor F based on stretch and mode.
    """
    if mode in ('UT', 'UC'):
        # Uniaxial Tension: diag(lambda, lambda^-0.5, lambda^-0.5)
        F = np.diag([stretch, stretch**-0.5, stretch**-0.5])
    elif mode == 'ET':
        # Equibiaxial Tension: diag(lambda, lambda, lambda^-2)
        F = np.diag([stretch, stretch, stretch**-2.0])
    elif mode == 'PS':
        # Pure Shear: diag(lambda, 1.0, lambda^-1)
        F = np.diag([stretch, 1.0, stretch**-1.0])
    elif mode == 'SS':
        # Simple Shear: gamma in x-y plane
        F = np.array([
            [1.0, stretch, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
    elif mode == 'CSS':
        # Compound Simple Shear: axial stretch (lambda_1) + shear gamma in x-y plane
        if isinstance(stretch, (tuple, list, np.ndarray)) and len(stretch) == 2:
            gamma, lam1 = float(stretch[0]), float(stretch[1])
        else:
            gamma = float(stretch)
            lam1 = 1.0
        lam2 = lam1**-0.5
        F = np.array([
            [lam1, gamma, 0.0],
            [0.0, lam2, 0.0],
            [0.0, 0.0, lam2],
        ])
    elif mode == 'BT':
        # Biaxial Tension: diag(lambda1, lambda2, (lambda1*lambda2)^-1)
        lam1, lam2 = stretch
        F = np.diag([lam1, lam2, (lam1 * lam2)**-1.0])
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return F

def get_stress_components(P_tensor, mode):
    """
    Extract stress components from the P tensor based on the experimental mode.
    Returns a list of components in the expected order.
    """
    if mode in ('UT', 'UC'):
        # Uniaxial Tension in X: P_11
        return [P_tensor[0, 0]]
    if mode == 'ET':
        # Equibiaxial Tension in X-Y: P_11 (or P_22)
        return [P_tensor[0, 0]]
    if mode == 'PS':
        # Pure Shear (Wide strip clamped in Y, pulled in X): P_11 and P_22 if needed
        return [P_tensor[0, 0], P_tensor[1, 1]]
    if mode == 'SS':
        # Simple Shear: P_12
        return [P_tensor[0, 1]]
    if mode == 'CSS':
        # Compound Simple Shear: P_12
        return [P_tensor[0, 1]]
    if mode == 'BT':
        # Biaxial Tension: P_11 and P_22
        return [P_tensor[0, 0], P_tensor[1, 1]]
    raise ValueError(f"Unsupported mode: {mode}")

def inv_Langevin_Kroger(x):
    """
    Inverse Langevin approximation by Martin Kröger (JNNFM, 2015).
    Eqn. 14 in the reference paper.
    Calculates L^(-1)(x).
    
    Args:
        x: Input value (scalar or symbol).
    """
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    
    top = 15.0 - (6.0 * x2 + x4 - 2.0 * x6)
    bot = 5.0 * (1.0 - x2)
    
    return x * top / bot

def _parse_mode(mode_raw):
    if mode_raw.startswith('BT'):
        return 'BT'
    if mode_raw.startswith('UT'):
        return 'UT'
    if mode_raw.startswith('UC'):
        return 'UC'
    if mode_raw.startswith('CSS'):
        return 'CSS'
    if mode_raw.startswith('SS'):
        return 'SS'
    return mode_raw

def _text_data_path(cfg, data_root):
    return os.path.join(data_root, cfg['author'], f"{cfg['mode']}.txt")

def _load_text_dataset(cfg, data_root):
    author = cfg['author']
    mode_raw = cfg['mode']
    mode = _parse_mode(mode_raw)

    file_path = _text_data_path(cfg, data_root)
    if not os.path.exists(file_path):
        print(f"  Warning: File not found at {file_path}")
        return None

    raw_data = np.loadtxt(file_path)
    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(1, -1)

    stress_type = 'PK1'
    bt_component = None
    if author == 'Jones_1975' and mode == 'BT':
        stress_type = 'cauchy'
        bt_component = 'diff'

    if mode == 'BT':
        if raw_data.shape[1] not in (3, 4):
            raise ValueError(f"BT data must have 3 or 4 columns: {file_path}")
        stretch_list = raw_data[:, 0]
        stretch_secondary = raw_data[:, 1]
        stress_primary = raw_data[:, 2]
        if raw_data.shape[1] == 4:
            stress_secondary = raw_data[:, 3]
            stress_exp_list = np.column_stack([stress_primary, stress_secondary])
        else:
            stress_exp_list = stress_primary

        f_tensors = []
        for lam1, lam2 in zip(stretch_list, stretch_secondary):
            F = get_deformation_gradient((lam1, lam2), mode)
            f_tensors.append(F)

        entry = {
            'author': author,
            'tag': f"{author}_{mode_raw}",
            'mode': mode,
            'mode_raw': mode_raw,
            'stress_type': stress_type,
            'stretch': stretch_list,
            'stretch_secondary': stretch_secondary,
            'stress_exp': stress_exp_list,
            'F_list': np.array(f_tensors)
        }
        if bt_component:
            entry['bt_component'] = bt_component
        return entry

    stretch_list = raw_data[:, 0]
    stress_exp_list = raw_data[:, 1]

    f_tensors = []
    for lam in stretch_list:
        F = get_deformation_gradient(lam, mode)
        f_tensors.append(F)

    entry = {
        'author': author,
        'tag': f"{author}_{mode_raw}",
        'mode': mode,
        'mode_raw': mode_raw,
        'stress_type': stress_type,
        'stretch': stretch_list,
        'stress_exp': stress_exp_list,
        'F_list': np.array(f_tensors)
    }
    if bt_component:
        entry['bt_component'] = bt_component
    return entry

def _load_h5_dataset(cfg, h5f):
    author = cfg['author']
    mode_raw = cfg['mode']
    mode = _parse_mode(mode_raw)

    if author not in h5f or mode_raw not in h5f[author]:
        print(f"  Warning: HDF5 group not found for {author}/{mode_raw}")
        return None

    grp = h5f[author][mode_raw]
    F_list = grp["F"][()]
    stress_tensor = grp["stress"][()]
    stress_type = grp.attrs.get("stress_type", "PK1")
    if isinstance(stress_type, bytes):
        stress_type = stress_type.decode("utf-8")
    bt_component = None
    if author == "Jones_1975" and mode == "BT":
        stress_type = "cauchy"
        bt_component = "diff"

    if "stretch" in grp:
        stretch = grp["stretch"][()]
    else:
        stretch = F_list[:, 0, 0]
    if "stretch_secondary" in grp:
        stretch_secondary = grp["stretch_secondary"][()]
    else:
        stretch_secondary = None

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
                    F = get_deformation_gradient((lam, lam2), "BT")
                else:
                    F = get_deformation_gradient(lam, "PS")
                f_tensors.append(F)
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
            entries.append(entry)
        if bt_component:
            for entry in entries:
                entry["bt_component"] = bt_component
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

def load_experimental_data(configs):
    """
    Load experimental data based on a list of configurations.
    Prints status messages directly.
    """
    print(f"\n[Data Loading] Processing {len(configs)} configuration(s)...")
    data_root = os.environ.get("CALIBRATION_DATA_DIR", "data")
    data_h5_path = os.path.join(data_root, "data.h5")

    all_tests = []
    use_h5 = h5py and os.path.exists(data_h5_path)
    if use_h5:
        with h5py.File(data_h5_path, "r") as h5f:
            for cfg in configs:
                entry = _load_h5_dataset(cfg, h5f)
                if entry:
                    if isinstance(entry, list):
                        all_tests.extend(entry)
                    else:
                        all_tests.append(entry)
    else:
        for cfg in configs:
            entry = _load_text_dataset(cfg, data_root)
            if entry:
                if isinstance(entry, list):
                    all_tests.extend(entry)
                else:
                    all_tests.append(entry)

    # Check results
    if not all_tests:
        print("  Error: No valid data loaded. Please check 'data/' folder.")
        sys.exit(1)  # Exit program if no data
    else:
        print(f"  Success: Loaded {len(all_tests)} datasets.")
    return all_tests

def load_experimental_data_h5(configs, data_h5_path):
    print(f"  Using HDF5 dataset: {data_h5_path}")
    all_tests = []

    with h5py.File(data_h5_path, "r") as h5f:
        for cfg in configs:
            author = cfg['author']
            mode_raw = cfg['mode']
            mode = _parse_mode(mode_raw)

            if author not in h5f or mode_raw not in h5f[author]:
                print(f"  Warning: HDF5 group not found for {author}/{mode_raw}")
                continue

            grp = h5f[author][mode_raw]
            F_list = grp["F"][()]
            stress_tensor = grp["stress"][()]
            stress_type = grp.attrs.get("stress_type", "PK1")
            if isinstance(stress_type, bytes):
                stress_type = stress_type.decode("utf-8")
            bt_component = None
            if author == "Jones_1975" and mode == "BT":
                stress_type = "cauchy"
                bt_component = "diff"

            if "stretch" in grp:
                stretch = grp["stretch"][()]
            else:
                stretch = F_list[:, 0, 0]
            if "stretch_secondary" in grp:
                stretch_secondary = grp["stretch_secondary"][()]
            else:
                stretch_secondary = None

            if mode in ("SS", "CSS"):
                stress_exp = stress_tensor[:, 0, 1]
            elif mode == "BT":
                stress_exp = np.column_stack([stress_tensor[:, 0, 0], stress_tensor[:, 1, 1]])
            elif mode == "PS" and np.any(np.abs(stress_tensor[:, 1, 1]) > 1e-12):
                stress_exp = np.column_stack([stress_tensor[:, 0, 0], stress_tensor[:, 1, 1]])
            else:
                stress_exp = stress_tensor[:, 0, 0]

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
                            F = get_deformation_gradient((lam, lam2), "BT")
                        else:
                            F = get_deformation_gradient(lam, "PS")
                        f_tensors.append(F)
                    entry = {
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
                all_tests.extend(entries)
            else:
                entry = {
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
                all_tests.append(entry)

    if not all_tests:
        print("  Error: No valid data loaded from HDF5. Please check 'data/data.h5'.")
        sys.exit(1)
    else:
        print(f"  Success: Loaded {len(all_tests)} datasets.")

    return all_tests

# EOF
