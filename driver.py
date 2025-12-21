import sys
import os
import json
import numpy as np

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))

from utils import load_experimental_data
from material_models import MaterialModels, print_model_info
from generalized_strains import STRAIN_CONFIGS
from parallel_springs import ParallelNetwork
from kinematics import Kinematics
from optimization import MaterialOptimizer
from plotting import plot_comparison

# ==========================================
# Helper Functions
# ==========================================

def get_available_datasets():
    data_dir = os.path.join(current_dir, 'data')
    datasets = {}
    if not os.path.exists(data_dir): return {}
    for author in os.listdir(data_dir):
        author_path = os.path.join(data_dir, author)
        if os.path.isdir(author_path):
            modes = [f.replace('.txt', '') for f in os.listdir(author_path) if f.endswith('.txt')]
            if modes: datasets[author] = sorted(modes)
    return datasets

def user_select_data(datasets):
    print("\n" + "="*50)
    print(" 1. Select Experimental Data Source")
    print("="*50)
    authors = list(datasets.keys())
    for i, author in enumerate(authors):
        print(f"{i + 1}. {author} (Modes: {', '.join(datasets[author])})")
    
    while True:
        try:
            sel_idx = int(input("\nSelect dataset number: ")) - 1
            if 0 <= sel_idx < len(authors):
                selected_author = authors[sel_idx]
                break
            print("Invalid selection.")
        except ValueError:
            print("Please enter a number.")
    
    print(f"\nAvailable Modes: {datasets[selected_author]}")
    mode_input = input("Select modes (e.g. '1, 3' or 'a' for all): ").strip()
    
    available_modes = datasets[selected_author]
    if mode_input.lower() == 'a':
        selected_modes = available_modes
    else:
        try:
            indices = [int(x)-1 for x in mode_input.split(',')]
            selected_modes = [available_modes[i] for i in indices if 0 <= i < len(available_modes)]
        except ValueError:
             print("Invalid input, defaulting to all.")
             selected_modes = available_modes
        
    return [{'author': selected_author, 'mode': m} for m in selected_modes]

def list_and_select_base_model():
    """Helper to show base model list and return selection."""
    all_models = []
    for attr_name in dir(MaterialModels):
        attr = getattr(MaterialModels, attr_name)
        if hasattr(attr, 'model_type') and hasattr(attr, 'category'):
            if hasattr(attr, 'param_names') and attr.param_names:
                all_models.append((attr_name, attr))
    all_models.append(("Hill", None))

    print("\n  --- Available Base Models ---")
    for i, (name, func) in enumerate(all_models):
        if name == "Hill":
            print(f"  {i+1}. {name:<15} [Configurable]")
        else:
            print(f"  {i+1}. {name:<15} [{func.category}, {func.model_type}]")
    
    choice = int(input("\n  Select model number: ")) - 1
    model_name, model_func = all_models[choice]
    
    # Configuration for Hill
    if model_name == "Hill":
        print("  > Configuring Hill Model...")
        strain_names = list(STRAIN_CONFIGS.keys())
        for i, s_name in enumerate(strain_names):
            print(f"    {i+1}. {s_name}")
        s_choice = int(input("    Select strain: ")) - 1
        selected_strain = strain_names[s_choice]
        model_func = MaterialModels.create_hill_model(selected_strain)
        model_name = f"Hill_{selected_strain}"
        
    return model_name, model_func

def user_construct_model():
    print("\n" + "="*50)
    print(" 2. Construct Material Model Strategy")
    print("="*50)
    print("1. Single Model")
    print("2. Parallel Network (Multi-Spring)")
    
    strategy = input("\nSelect Strategy (1 or 2): ").strip()
    
    if strategy == '1':
        # --- Single Model Path ---
        return list_and_select_base_model()
        
    elif strategy == '2':
        # --- Parallel Network Path ---
        network = ParallelNetwork()
        print("\n>>> Building Parallel Network <<<")
        
        branch_count = 0
        while True:
            branch_count += 1
            print(f"\n[Adding Branch #{branch_count}]")
            
            try:
                model_name, model_func = list_and_select_base_model()
            except (ValueError, IndexError):
                print("Invalid selection, try again.")
                continue

            default_name = f"{model_name}_{branch_count}"
            user_name = input(f"  Name for this branch (default: {default_name}): ").strip()
            if not user_name: user_name = default_name
            
            network.add_model(model_func, user_name)
            
            cont = input("\nAdd another branch? (y/n): ").strip().lower()
            if cont != 'y':
                break
        
        return "ParallelNetwork", network
    else:
        print("Invalid strategy. Defaulting to Single Model.")
        return list_and_select_base_model()

def user_select_optimizer():
    """Interactive selection for optimization method."""
    print("\n" + "="*50)
    print(" 3. Select Optimization Method")
    print("="*50)
    methods = [
        ("L-BFGS-B", "Gradient-based, handles bounds, fast (Default)"),
        ("Nelder-Mead", "Derivative-free, robust for noisy/complex landscapes"),
        ("Powell", "Derivative-free, conjugate direction method"),
        ("TNC", "Truncated Newton, good for large constraints"),
        ("SLSQP", "Sequential Least Squares Programming, strong constraints"),
        ("Differential Evolution", "Global search, slow but escapes local minima")
    ]
    
    for i, (name, desc) in enumerate(methods):
        print(f"{i+1}. {name:<22} : {desc}")
        
    choice = input(f"\nSelect method (1-{len(methods)}, default 1): ").strip()
    if not choice:
        return "L-BFGS-B"
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(methods):
            return methods[idx][0]
    except ValueError:
        pass
    
    print("Invalid choice, defaulting to L-BFGS-B.")
    return "L-BFGS-B"

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    OUTPUT_DIR = "output"
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    print(f"--- Hyperelastic Calibration v3.1 (Optimization Options) ---")

    # 1. Data Selection
    datasets = get_available_datasets()
    if not datasets:
        print("No data found in data/ folder.")
        sys.exit(1)
    data_configs = user_select_data(datasets)
    experimental_data = load_experimental_data(data_configs)

    # 2. Model Construction (Single or Parallel)
    model_name, model_obj = user_construct_model()
    
    # 3. Optimization Method Selection
    opt_method = user_select_optimizer()

    # 4. Setup Config
    if isinstance(model_obj, ParallelNetwork):
        param_names = model_obj.param_names
        initial_guess = model_obj.initial_guess
        bounds = model_obj.bounds
        print(f"\n[Network Constructed]")
        print(f"  Formula: {model_obj.formula}")
    else:
        print_model_info(model_obj)
        param_names = getattr(model_obj, 'param_names', [])
        initial_guess = getattr(model_obj, 'initial_guess', [])
        bounds = getattr(model_obj, 'bounds', None)

    print(f"\n[Configuration]")
    print(f"  Params: {param_names}")
    print(f"  Initial Guess: {initial_guess}")

    # Arruda-Boyce Safety Check
    for i, p_name in enumerate(param_names):
        if p_name == 'N' or p_name.endswith('_N'):
            max_I1 = 0.0
            for d in experimental_data:
                lam = np.max(d['stretch'])
                curr = lam**2 + 2/lam if d['mode']=='UT' else (2*lam**2 + 1/lam**4 if d['mode']=='ET' else lam**2 + 1 + 1/lam**2)
                max_I1 = max(max_I1, curr)
            
            min_N = max_I1 / 3.0
            if initial_guess[i] <= min_N:
                initial_guess[i] = min_N * 1.5
                print(f"  > [Safety] Adjusted '{p_name}' to {initial_guess[i]:.2f}")
            if bounds:
                bounds[i] = (min_N * 1.01, None)

    # 5. Optimization
    solver = Kinematics(model_obj, param_names)
    optimizer = MaterialOptimizer(solver, experimental_data)
    
    # Pass the selected method to fit()
    result = optimizer.fit(initial_guess, bounds, method=opt_method)

    if result.success:
        print("\nSUCCESS!")
        final_params = dict(zip(param_names, result.x))
        print(f"Params: {final_params}")
        
        plot_comparison(
            experimental_data, 
            solver, 
            final_params,
            title=f"{model_name} Fit ({opt_method})",
            save_path=os.path.join(OUTPUT_DIR, "fit_result.png")
        )
    else:
        print("\nOptimization Failed.")
        print(result.message)