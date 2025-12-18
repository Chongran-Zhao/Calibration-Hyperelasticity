import numpy as np
import os

def load_experimental_data(configs):
    """
    Load experimental data based on a list of configurations.
    
    Args:
        configs: List of dicts, e.g., [{'author': 'Treloar_1944', 'mode': 'UT'}, ...]
    
    Returns:
        A list of dictionaries containing processed tensors and metadata.
    """
    all_tests = []
    
    for cfg in configs:
        author = cfg['author']
        mode = cfg['mode']
        
        # Construct path based on your tree structure
        file_path = os.path.join("data", author, f"{mode}.txt")
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found at {file_path}")
            continue
            
        # Load data: Column 0 = Stretch, Column 1 = First PK Stress (P)
        raw_data = np.loadtxt(file_path)
        stretch_list = raw_data[:, 0]
        stress_exp_list = raw_data[:, 1]
        
        # Pre-construct Deformation Gradient (F) tensors for each data point
        f_tensors = []
        for lambda_ in stretch_list:
            if mode == 'UT':
                # Uniaxial Tension: diag(lambda, lambda^-0.5, lambda^-0.5)
                F = np.diag([lambda_, lambda_**-0.5, lambda_**-0.5])
            elif mode == 'ET':
                # Equibiaxial Tension: diag(lambda, lambda, lambda^-2)
                F = np.diag([lambda_, lambda_, lambda_**-2.0])
            elif mode == 'PS':
                # Pure Shear: diag(lambda, 1.0, lambda^-1)
                F = np.diag([lambda_, 1.0, lambda_**-1.0])
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            f_tensors.append(F)
            
        all_tests.append({
            'tag': f"{author}_{mode}",
            'mode': mode,
            'stretch': stretch_list,
            'stress_exp': stress_exp_list,
            'F_list': np.array(f_tensors)
        })
        
    return all_tests

# EOF
