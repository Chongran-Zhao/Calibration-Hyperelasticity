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
    if mode == 'UT':
        # Uniaxial Tension: diag(lambda, lambda^-0.5, lambda^-0.5)
        F = np.diag([stretch, stretch**-0.5, stretch**-0.5])
    elif mode == 'ET':
        # Equibiaxial Tension: diag(lambda, lambda, lambda^-2)
        F = np.diag([stretch, stretch, stretch**-2.0])
    elif mode == 'PS':
        # Pure Shear: diag(lambda, 1.0, lambda^-1)
        F = np.diag([stretch, 1.0, stretch**-1.0])
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
    if mode == 'UT':
        # Uniaxial Tension in X: P_11
        return [P_tensor[0, 0]]
    if mode == 'ET':
        # Equibiaxial Tension in X-Y: P_11 (or P_22)
        return [P_tensor[0, 0]]
    if mode == 'PS':
        # Pure Shear (Wide strip clamped in Y, pulled in X): P_11
        return [P_tensor[0, 0]]
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
    return mode_raw

def load_experimental_data(configs):
    """
    Load experimental data based on a list of configurations.
    Prints status messages directly.
    """
    print(f"\n[Data Loading] Processing {len(configs)} configuration(s)...")
    data_root = os.environ.get("CALIBRATION_DATA_DIR", "data")
    data_h5_path = os.path.join(data_root, "data.h5")
    if h5py and os.path.exists(data_h5_path):
        return load_experimental_data_h5(configs, data_h5_path)
    all_tests = []
    
    for cfg in configs:
        author = cfg['author']
        mode_raw = cfg['mode']
        mode = _parse_mode(mode_raw)
        
        file_path = os.path.join(data_root, author, f"{mode_raw}.txt")
        
        if not os.path.exists(file_path):
            print(f"  Warning: File not found at {file_path}")
            continue
            
        raw_data = np.loadtxt(file_path)
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)

        stress_type = 'PK1'
        if author == 'Jones_1975' and mode == 'BT':
            stress_type = 'cauchy'

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

            all_tests.append({
                'tag': f"{author}_{mode_raw}",
                'mode': mode,
                'mode_raw': mode_raw,
                'stress_type': stress_type,
                'stretch': stretch_list,
                'stretch_secondary': stretch_secondary,
                'stress_exp': stress_exp_list,
                'F_list': np.array(f_tensors)
            })
        else:
            stretch_list = raw_data[:, 0]
            stress_exp_list = raw_data[:, 1]

            f_tensors = []
            for lam in stretch_list:
                F = get_deformation_gradient(lam, mode)
                f_tensors.append(F)

            all_tests.append({
                'tag': f"{author}_{mode_raw}",
                'mode': mode,
                'mode_raw': mode_raw,
                'stress_type': stress_type,
                'stretch': stretch_list,
                'stress_exp': stress_exp_list,
                'F_list': np.array(f_tensors)
            })
        
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

            if "stretch" in grp:
                stretch = grp["stretch"][()]
            else:
                stretch = F_list[:, 0, 0]
            if "stretch_secondary" in grp:
                stretch_secondary = grp["stretch_secondary"][()]
            else:
                stretch_secondary = None

            if mode == "BT":
                stress_exp = np.column_stack([stress_tensor[:, 0, 0], stress_tensor[:, 1, 1]])
            else:
                stress_exp = stress_tensor[:, 0, 0]

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
            all_tests.append(entry)

    if not all_tests:
        print("  Error: No valid data loaded from HDF5. Please check 'data/data.h5'.")
        sys.exit(1)
    else:
        print(f"  Success: Loaded {len(all_tests)} datasets.")

    return all_tests

# EOF
