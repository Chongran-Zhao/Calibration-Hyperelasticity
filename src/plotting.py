import numpy as np
import matplotlib.pyplot as plt
from utils import get_deformation_gradient, get_stress_component

# Global Configuration
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"  # Better LaTeX math support
plt.rcParams["font.size"] = 12

def plot_comparison(experimental_data, kinematics_solver, fitted_params, title="Model Fit Results"):
    """
    Plots experimental data against model predictions.
    
    Args:
        experimental_data: List of dicts (from load_experimental_data).
        kinematics_solver: Instance of Kinematics class.
        fitted_params: Dictionary of optimized parameters (e.g. {'C1': 0.5}).
        title: Plot title.
    """
    plt.figure(figsize=(10, 7))
    
    # Styles for different modes to distinguish them visually
    markers = {'UT': 'o', 'ET': 's', 'PS': '^'}
    colors = {'UT': 'blue', 'ET': 'red', 'PS': 'green'}
    
    for dataset in experimental_data:
        mode = dataset['mode']
        tag = dataset['tag']
        exp_stretch = dataset['stretch']
        exp_stress = dataset['stress_exp']
        
        # 1. Plot Experimental Data (Scatter)
        plt.scatter(
            exp_stretch, 
            exp_stress, 
            label=f"Exp: {tag}", 
            marker=markers.get(mode, 'o'), 
            facecolors='none', 
            edgecolors=colors.get(mode, 'black'),
            s=60,
            zorder=2
        )
        
        # 2. Plot Model Prediction (Smooth Line)
        # Generate smooth stretch range for better visualization
        min_lam = np.min(exp_stretch)
        max_lam = np.max(exp_stretch)
        # Add a little padding to the range
        smooth_stretch = np.linspace(1.0, max_lam * 1.05, 100)
        
        model_stress_list = []
        for lam in smooth_stretch:
            # Construct F for this stretch
            F_smooth = get_deformation_gradient(lam, mode)
            
            # Calculate P tensor
            P_tensor = kinematics_solver.get_1st_PK_stress(F_smooth, fitted_params)
            
            # Extract scalar component
            val = get_stress_component(P_tensor, mode)
            model_stress_list.append(val)
            
        plt.plot(
            smooth_stretch, 
            model_stress_list, 
            color=colors.get(mode, 'black'), 
            linestyle='-', 
            linewidth=2,
            label=f"Fit: {mode}", 
            zorder=1
        )

    plt.xlabel("Stretch $\lambda$ [-]")
    plt.ylabel("Nominal Stress $P$ [MPa]")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# EOF
