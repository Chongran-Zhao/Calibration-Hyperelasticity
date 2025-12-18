import sys
import os
import json

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))

from utils import load_experimental_data
from material_models import InvariantModels
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
# SWITCHING TO ARRUDA-BOYCE
print("\n[Model Initialization] Arruda-Boyce (8-Chain)")
model_function = InvariantModels.ArrudaBoyce
param_names = ['mu', 'lambda_m']

solver = Kinematics(model_function, param_names, model_type='invariant')

# --- NEW: Print Model Info ---
solver.print_model_info()

# --- 4. Optimization ---
optimizer = MaterialOptimizer(solver, experimental_data)

# Initial Guess for Arruda-Boyce
# mu: Initial stiffness (approx 0.3 - 0.5 MPa for soft rubber)
# lambda_m: Locking stretch limit (approx 3.0 - 7.0)
initial_guess = [0.4, 3.0] 

# Bounds
# mu > 0
# lambda_m > 1.0 (Must be stretchable)
bounds = [(0.01, None), (1.01, None)]

print(f"Initial Guess: {dict(zip(param_names, initial_guess))}")

# Run fit (Progress bar will appear here)
result = optimizer.fit(initial_guess, bounds)

# --- 5. Results & Plotting ---
if result.success:
    print("\n--- Optimization Successful ---")
    optimized_params = dict(zip(param_names, result.x))
    print(f"Parameters: {optimized_params}")
    print(f"Loss (SSE): {result.fun:.6f}")
    
    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, "fitted_parameters.json")
    with open(json_path, 'w') as f:
        json.dump({"model": "ArrudaBoyce", "params": optimized_params, "sse": result.fun}, f, indent=4)
    print(f"Data saved to: {json_path}")
    
    # Save Plot
    plot_path = os.path.join(OUTPUT_DIR, "fitting_result.png")
    plot_comparison(
        experimental_data, 
        solver, 
        optimized_params, 
        title=f"Arruda-Boyce Model Fit (SSE: {result.fun:.2f})",
        save_path=plot_path
    )
else:
    print("\nOptimization Failed.")
    print(result.message)
