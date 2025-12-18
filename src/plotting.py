import numpy as np
import matplotlib.pyplot as plt
from utils import get_deformation_gradient, get_stress_component

# Global Configuration
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 12

def plot_comparison(experimental_data, kinematics_solver, fitted_params, title="Model Fit Results", save_path=None):
    """
    Plots experimental data against model predictions.
    
    Args:
        save_path: If provided, saves the figure to this path instead of showing it.
    """
    plt.figure(figsize=(10, 7))
    
    markers = {'UT': 'o', 'ET': 's', 'PS': '^'}
    colors = {'UT': 'blue', 'ET': 'red', 'PS': 'green'}
    
    for dataset in experimental_data:
        mode = dataset['mode']
        tag = dataset['tag']
        exp_stretch = dataset['stretch']
        exp_stress = dataset['stress_exp']
        
        # Plot Experiment
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
        
        # Plot Model Prediction
        min_lam = np.min(exp_stretch)
        max_lam = np.max(exp_stretch)
        smooth_stretch = np.linspace(1.0, max_lam * 1.05, 100)
        
        model_stress_list = []
        for lam in smooth_stretch:
            F_smooth = get_deformation_gradient(lam, mode)
            P_tensor = kinematics_solver.get_1st_PK_stress(F_smooth, fitted_params)
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

    plt.xlabel(r"Stretch $\lambda$ [-]")
    plt.ylabel("Nominal Stress $P$ [MPa]")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")
        plt.close() # Close memory
    else:
        plt.show()

# EOF
