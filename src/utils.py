import numpy as np
import os
import sys

def get_deformation_gradient(stretch, mode):
    """
    Constructs the deformation gradient tensor F based on stretch and mode.
    """
    if mode == 'UT':
        F = np.diag([stretch, stretch**-0.5, stretch**-0.5])
    elif mode == 'ET':
        F = np.diag([stretch, stretch, stretch**-2.0])
    elif mode == 'PS':
        F = np.diag([stretch, 1.0, stretch**-1.0])
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return F

def get_stress_component(P_tensor, mode):
    """
    Extract the relevant scalar stress component.
    """
    if mode == 'UT': 
        return P_tensor[0, 0]
    elif mode == 'ET':
        return P_tensor[0, 0]
    elif mode == 'PS':
        return P_tensor[0, 0]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

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
