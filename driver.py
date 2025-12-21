import sys
import os
import json
import numpy as np

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))

from utils import load_experimental_data
from material_models import MaterialModels, print_model_info
from generalized_strains import STRAIN_CONFIGS # Import strain configs
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

def user_select_model():
    print("\n" + "="*50)
    print(" 2. Select Material Model")
    print("="*50)
    
    # 1. Get all standard models
    all_models = []
    for attr_name in dir(MaterialModels):
        attr = getattr(MaterialModels, attr_name)
        if hasattr(attr, 'model_type') and hasattr(attr, 'category'):
            if hasattr(attr, 'param_names') and attr.param_names:
                all_models.append((attr_name, attr))
    
    # Add generic "Hill" entry explicitly to the list
    all_models.append(("Hill", None)) # Placeholder for generic Hill

    # Filter Logic
    print("Filter by Category?")
    print("0. Show All")
    print("1. Phenomenological Only")
    print("2. Micromechanical Only")
    filter_choice = input("Choice (0-2): ").strip()
    
    filtered_models = []
    for name, func in all_models:
        if name == "Hill": # Always show Hill if Phenomenological or All
            if filter_choice in ['0', '1']:
                filtered_models.append((name, func))
            continue
            
        if filter_choice == '1' and func.category != 'phenomenological': continue
        if filter_choice == '2' and func.category != 'micromechanical': continue
        filtered_models.append((name, func))

    # Display Models
    print("\n--- Available Models ---")
    for i, (name, func) in enumerate(filtered_models):
        if name == "Hill":
            print(f"{i+1}. {name:<15} [phenomenological, stretch_based] (Configurable)")
        else:
            print(f"{i+1}. {name:<15} [{func.category}, {func.model_type}]")
        
    choice = int(input("\nSelect model number: ")) - 1
    model_name, model_func = filtered_models[choice]
    
    # --- SPECIAL HANDLING FOR HILL MODEL ---
    if model_name == "Hill":
        print("\n  >>> Hill Model Configuration <<<")
        print("  Select Generalized Strain Measure:")
        strain_names = list(STRAIN_CONFIGS.keys())
        for i, s_name in enumerate(strain_names):
            print(f"  {i+1}. {s_name}")
        
        s_choice = int(input("\n  Select strain number: ")) - 1
        selected_strain = strain_names[s_choice]
        
        # Use Factory to create the specific function
        print(f"  > Constructing Hill Model with '{selected_strain}' strain...")
        model_func = MaterialModels.create_hill_model(selected_strain)
        model_name = f"Hill_{selected_strain}"

    return model_name, model_func

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    OUTPUT_DIR = "output"
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    print(f"--- Hyperelastic Calibration v2 ---")

    # 1. Data Selection
    datasets = get_available_datasets()
    if not datasets:
        print("No data found in data/ folder.")
        sys.exit(1)
        
    data_configs = user_select_data(datasets)
    experimental_data = load_experimental_data(data_configs)

    # 2. Model Selection (With Hill Sub-menu)
    model_name, model_function = user_select_model()
    
    # Show Info
    print_model_info(model_function)

    # 3. Setup Config
    param_names = getattr(model_function, 'param_names', [])
    initial_guess = getattr(model_function, 'initial_guess', [])
    bounds = getattr(model_function, 'bounds', None)

    print(f"\n[Configuration Loaded]")
    print(f"  Params: {param_names}")
    print(f"  Default Guess: {initial_guess}")

    # Arruda-Boyce Safety Check
    if "ArrudaBoyce" in model_name:
        max_I1 = 0.0
        for d in experimental_data:
            lam = np.max(d['stretch'])
            curr = lam**2 + 2/lam if d['mode']=='UT' else (2*lam**2 + 1/lam**4 if d['mode']=='ET' else lam**2 + 1 + 1/lam**2)
            max_I1 = max(max_I1, curr)
        
        min_N = max_I1 / 3.0
        # N is usually the last parameter
        if initial_guess[-1] <= min_N:
            initial_guess[-1] = min_N * 1.2
            print(f"  > [Auto-Adjust] N guess increased to {initial_guess[-1]:.2f}")
        bounds = [(0.01, None), (min_N * 1.01, None)]

    # 4. Optimization
    solver = Kinematics(model_function, param_names)
    optimizer = MaterialOptimizer(solver, experimental_data)
    
    result = optimizer.fit(initial_guess, bounds)

    if result.success:
        print("\nSUCCESS!")
        print(f"Params: {dict(zip(param_names, result.x))}")
        
        plot_comparison(
            experimental_data, 
            solver, 
            dict(zip(param_names, result.x)),
            title=f"{model_name} Fit",
            save_path=os.path.join(OUTPUT_DIR, "fit_result.png")
        )
    else:
        print("\nOptimization Failed.")
        print(result.message)