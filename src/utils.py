import numpy as np
import os
import sys

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
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return F

def get_stress_component(P_tensor, mode):
    """
    Extract the relevant scalar stress component from the P tensor 
    based on the experimental mode.
    """
    if mode == 'UT': 
        # Uniaxial Tension in X: P_11
        return P_tensor[0, 0]
    elif mode == 'ET':
        # Equibiaxial Tension in X-Y: P_11 (or P_22)
        return P_tensor[0, 0]
    elif mode == 'PS':
        # Pure Shear (Wide strip clamped in Y, pulled in X): P_11
        return P_tensor[0, 0]
    else:
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

def load_experimental_data(configs):
    """
    Load experimental data based on a list of configurations.
    Prints status messages directly.
    """
    print(f"\n[Data Loading] Processing {len(configs)} configuration(s)...")
    all_tests = []
    
    for cfg in configs:
        author = cfg['author']
        mode = cfg['mode']
        
        file_path = os.path.join("data", author, f"{mode}.txt")
        
        if not os.path.exists(file_path):
            print(f"  Warning: File not found at {file_path}")
            continue
            
        raw_data = np.loadtxt(file_path)
        stretch_list = raw_data[:, 0]
        stress_exp_list = raw_data[:, 1]
        
        f_tensors = []
        for lam in stretch_list:
            F = get_deformation_gradient(lam, mode)
            f_tensors.append(F)
            
        all_tests.append({
            'tag': f"{author}_{mode}",
            'mode': mode,
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

# EOF
