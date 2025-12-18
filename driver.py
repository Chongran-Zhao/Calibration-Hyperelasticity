import sys
import os
import json
import numpy as np 

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))

from utils import load_experimental_data
from material_models import InvariantModels, print_model_formula
from kinematics import Kinematics
from optimization import MaterialOptimizer
from plotting import plot_comparison

# --- 1. Setup ---
OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"--- Hyperelastic Material Fitting ---")

# --- 2. Load Data ---
data_configs = [
    {'author': 'Treloar_1944', 'mode': 'UT'},
    {'author': 'Treloar_1944', 'mode': 'ET'},
    {'author': 'Treloar_1944', 'mode': 'PS'}
]
experimental_data = load_experimental_data(data_configs)

# --- 3. Initialize Model ---
print("\n[Model Initialization] Arruda-Boyce (8-Chain / Kröger Approx)")
model_function = InvariantModels.ArrudaBoyce
param_names = ['mu', 'N'] 

# --- SAFETY CHECK FOR ARRUDA-BOYCE ---
max_I1 = 0.0
for dataset in experimental_data:
    stretch = dataset['stretch']
    mode = dataset['mode']
    lam_max = np.max(stretch)
    if mode == 'UT':
        current_I1 = lam_max**2 + 2.0/lam_max
    elif mode == 'ET':
        current_I1 = 2.0*lam_max**2 + 1.0/lam_max**4
    elif mode == 'PS':
        current_I1 = lam_max**2 + 1.0 + 1.0/lam_max**2
    
    if current_I1 > max_I1:
        max_I1 = current_I1

print(f"  > Data Check: Maximum I1 in data is approx {max_I1:.2f}")
min_required_N = max_I1 / 3.0
print(f"  > Constraint: Parameter N must be > {min_required_N:.2f} to avoid singularity.")

safe_N_guess = max(9.0, min_required_N * 1.2) 
initial_guess = [0.4, safe_N_guess]

bounds = [(0.01, None), (min_required_N * 1.01, None)]

solver = Kinematics(model_function, param_names, model_type='invariant')
print_model_formula(model_function.__name__)

# --- 4. Optimization ---
optimizer = MaterialOptimizer(solver, experimental_data)

print(f"Initial Guess (Auto-adjusted): {dict(zip(param_names, initial_guess))}")

# Run fit
result = optimizer.fit(initial_guess, bounds)

# --- 5. Results & Plotting ---
if result.success:
    print("\n--- Optimization Successful ---")
    optimized_params = dict(zip(param_names, result.x))
    print(f"Parameters: {optimized_params}")
    
    # 1. Update console output label
    print(f"Final Loss (Sum of 1-R^2): {result.fun:.6f}")
    
    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, "fitted_parameters.json")
    model_name = model_function.__name__
    with open(json_path, 'w') as f:
        # 3. Update JSON key
        json.dump({
            "model": model_name, 
            "params": optimized_params, 
            "loss_value": result.fun,  # Changed from 'cost_value'
            "metric": "Sum of (1 - R^2)"
        }, f, indent=4)
    print(f"Data saved to: {json_path}")
    
    # Save Plot
    plot_path = os.path.join(OUTPUT_DIR, "fitting_result.png")
    
    # 2. Update Plot Title
    plot_comparison(
        experimental_data, 
        solver, 
        optimized_params, 
        title=f"{model_name} Model Fit (Loss: {result.fun:.4f})", # Changed from Cost
        save_path=plot_path
    )
else:
    print("\nOptimization Failed.")
    print(result.message)
