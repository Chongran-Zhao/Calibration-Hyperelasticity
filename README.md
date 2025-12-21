# Hyperelastic Material Calibration Framework

A Python-based framework for calibrating hyperelastic material constitutive models against experimental data. This tool utilizes symbolic differentiation (via SymPy) to automatically derive stress tensors, ensuring mathematical accuracy and flexibility for various material models. The primary interface is a modern Streamlit GUI.

## ✨ Key Features

- **Symbolic Kinematics**: Automatically derives Second Piola-Kirchhoff ($S$), First Piola-Kirchhoff ($P$), and Cauchy ($\sigma$) stress tensors from strain energy density functions.
- **Multi-Mode Fitting**: Supports simultaneous fitting of Uniaxial Tension (UT), Equibiaxial Tension (ET), and Pure Shear (PS) data.
- **Extensive Model Library**: Includes built-in support for Neo-Hookean, Mooney-Rivlin, Yeoh, Arruda-Boyce, Ogden, Gent, and Hill-type generalized strain models.
- **Modern GUI**: Interactive Streamlit interface for model configuration, optimization, and prediction.
- **Model Introspection**: Prints the exact symbolic mathematical expression of the energy density function being used.

## 📂 Project Structure

```
Calibration-Hyperelasticity/
│
├── data/                    # Repository for experimental data
│   └── Treloar_1944/        # Example dataset
│       ├── UT.txt           # Uniaxial Tension data
│       ├── ET.txt           # Equibiaxial Tension data
│       └── PS.txt           # Pure Shear data
│
├── src/                     # Source code modules
│   ├── __init__.py
│   ├── utils.py             # Data loading and tensor helpers
│   ├── material_models.py   # Strain energy density definitions
│   ├── generalized_strains.py # Library of generalized strains (Seth-Hill, etc.)
│   ├── kinematics.py        # Symbolic solver core
│   ├── optimization.py      # Scipy wrapper and loss calculation
│   └── plotting.py          # Visualization tools
│
├── output/                  # Generated results (Plots and JSON)
├── calibration_gui.py       # Streamlit GUI entry point
├── requirement.txt          # Python dependencies
└── README.md                # This documentation
```

## 💿 Installation

Follow these steps to set up the environment for the project.

### 1. Prerequisites

Ensure you have **Python 3.8** or higher installed on your system. You can check your version by running:

```
python --version
```

### 2. (Optional) Create a Virtual Environment

It is recommended to use a virtual environment to manage dependencies and avoid conflicts with other projects.

**For macOS/Linux:**

```
python -m venv venv
source venv/bin/activate
```

**For Windows:**

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Install the required Python packages listed in `requirement.txt`:

```
pip install -r requirement.txt
```

### 4. Verify Installation

To ensure everything is set up correctly, launch the Streamlit GUI.

```
streamlit run calibration_gui.py
```

## 📖 Usage Guide

### 1. Prepare Experimental Data

Place your experimental data in the `data/` directory. You should create a subfolder for the dataset author or material name.

**File Format (`.txt`):**

- Data must be space-separated.
- **Column 1**: Stretch ($\lambda = L/L_0$)
- **Column 2**: Nominal Stress ($P = Force/Area$)

**Example (`data/MyMaterial/UT.txt`):**

```
1.0000 0.0000
1.1000 0.1500
1.2000 0.3200
...
```

### 2. Configure in the GUI

Open the app and configure datasets, spring models, and initial guesses directly in the interface.

### 3. Run the Calibration

Press **Start Calibration** to run the optimization and review fitted parameters and plots.

### 4. Analyze Results

Upon completion, check the `output/` directory:

- **`fitting_result.png`**: A comparison plot showing experimental data vs. the fitted model curves.
- **`fitted_parameters.json`**: A JSON file containing the optimized parameters and final loss value.

## 🛠️ Extending the Framework

To add a custom material model, edit `src/material_models.py`.

**Example:**

```
@staticmethod
def MyNewModel(I1, params):
    """
    Psi = C1 * (I1 - 3)^2
    """
    return params['C1'] * (I1 - 3)**2
```

You can then reference your model in the GUI. The kinematic solver handles the derivative derivations automatically.

## 📝 Notes

- **Incompressibility**: The framework assumes the material is incompressible ($J=1$). The hydrostatic pressure $p$ is automatically solved using the boundary condition $\sigma_{33}=0$.
- **Fonts**: The plotting module is configured to use "Times New Roman". If this font is not available on your system, Matplotlib may fallback to a default font.
