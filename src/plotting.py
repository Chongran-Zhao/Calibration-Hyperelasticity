import numpy as np
import matplotlib.pyplot as plt
from utils import get_deformation_gradient, get_stress_components

# Global Configuration
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 12

def calculate_r2(exp_stress, model_stress):
    """
    Helper to calculate R^2 (Coefficient of Determination).
    R^2 = 1 - (SS_res / SS_tot)
    """
    exp_flat = np.array(exp_stress).ravel()
    model_flat = np.array(model_stress).ravel()
    ss_res = np.sum((exp_flat - model_flat)**2)
    mean_exp = np.mean(exp_flat)
    ss_tot = np.sum((exp_flat - mean_exp)**2)
    
    # Avoid division by zero for constant data (though unlikely for stress-strain)
    if ss_tot < 1e-12:
        return 0.0 
        
    return 1 - (ss_res / ss_tot)

def plot_comparison(experimental_data, kinematics_solver, fitted_params, title="Model Fit Results", save_path=None):
    """
    Plots experimental data against model predictions.
    Calculates and displays R^2 for each mode in the legend.
    
    Args:
        experimental_data: List of dicts (from load_experimental_data).
        kinematics_solver: Instance of Kinematics class.
        fitted_params: Dictionary of optimized parameters.
        title: Plot title.
        save_path: If provided, saves the figure to this path instead of showing it.
    """
    plt.figure(figsize=(10, 7))
    
    markers = {'UT': 'o', 'ET': 's', 'PS': '^', 'BT': 'D'}
    colors = {'UT': 'blue', 'ET': 'red', 'PS': 'green', 'BT': 'purple'}
    
    for dataset in experimental_data:
        mode = dataset['mode']
        tag = dataset['tag']
        exp_stretch = dataset['stretch']
        exp_stress = dataset['stress_exp']
        f_list = dataset['F_list'] # Tensors at experimental points
        stress_type = dataset.get('stress_type', 'PK1')
        
        # --- 1. Calculate R^2 for this dataset ---
        # We need model predictions exactly at the experimental stretch points
        model_stress_at_exp = []
        for F in f_list:
            if stress_type == 'cauchy':
                stress_tensor = kinematics_solver.get_Cauchy_stress(F, fitted_params)
            else:
                stress_tensor = kinematics_solver.get_1st_PK_stress(F, fitted_params)
            comps = get_stress_components(stress_tensor, mode)
            if np.ndim(exp_stress) == 1:
                model_stress_at_exp.append(comps[0])
            else:
                model_stress_at_exp.append(comps[:exp_stress.shape[1]])
        model_stress_at_exp = np.array(model_stress_at_exp)
        
        r2 = calculate_r2(exp_stress, model_stress_at_exp)
        
        # --- 2. Plot Experimental Data (Scatter) ---
        # Add R^2 to the label
        stress_label = "Cauchy" if stress_type == "cauchy" else "Nominal"
        label_text = f"Exp: {tag} ({stress_label}, $R^2={r2:.3f}$)"
        
        if np.ndim(exp_stress) == 1:
            plt.scatter(
                exp_stretch,
                exp_stress,
                label=label_text,
                marker=markers.get(mode, 'o'),
                facecolors='none',
                edgecolors=colors.get(mode, 'black'),
                s=60,
                zorder=2
            )
        else:
            plt.scatter(
                exp_stretch,
                exp_stress[:, 0],
                label=f"{label_text} P11",
                marker=markers.get(mode, 'o'),
                facecolors='none',
                edgecolors=colors.get(mode, 'black'),
                s=60,
                zorder=2
            )
            plt.scatter(
                exp_stretch,
                exp_stress[:, 1],
                label=f"{label_text} P22",
                marker=markers.get(mode, '^'),
                facecolors='none',
                edgecolors=colors.get(mode, 'black'),
                s=60,
                zorder=2
            )
        
        # --- 3. Plot Model Prediction (Smooth Line) ---
        # Generate smooth stretch range for better visualization
        min_lam = np.min(exp_stretch)
        max_lam = np.max(exp_stretch)
        smooth_stretch = np.linspace(1.0, max_lam * 1.05, 100)
        
        model_stress_smooth = []
        model_stress_smooth_2 = []
        for lam in smooth_stretch:
            if mode == 'BT':
                lam2 = float(dataset['stretch_secondary'][0])
                F_smooth = get_deformation_gradient((lam, lam2), mode)
            else:
                F_smooth = get_deformation_gradient(lam, mode)
            if stress_type == 'cauchy':
                stress_tensor = kinematics_solver.get_Cauchy_stress(F_smooth, fitted_params)
            else:
                stress_tensor = kinematics_solver.get_1st_PK_stress(F_smooth, fitted_params)
            comps = get_stress_components(stress_tensor, mode)
            model_stress_smooth.append(comps[0])
            if len(comps) > 1:
                model_stress_smooth_2.append(comps[1])

        plt.plot(
            smooth_stretch,
            model_stress_smooth,
            color=colors.get(mode, 'black'),
            linestyle='-',
            linewidth=2,
            zorder=1
        )
        if model_stress_smooth_2:
            plt.plot(
                smooth_stretch,
                model_stress_smooth_2,
                color=colors.get(mode, 'black'),
                linestyle='--',
                linewidth=2,
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
        plt.close() # Close memory to avoid accumulation
    else:
        plt.show()
# EOF
