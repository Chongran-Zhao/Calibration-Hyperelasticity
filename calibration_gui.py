import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))

from utils import load_experimental_data
from material_models import MaterialModels
from generalized_strains import STRAIN_CONFIGS
from parallel_springs import ParallelNetwork
from kinematics import Kinematics
from optimization import MaterialOptimizer

# ==========================================
# UI Configuration & Styling (Modern Academic Theme)
# ==========================================
st.set_page_config(
    page_title="Hyperelastic Calibration",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Modern, Academic, and Fancy (System Theme Compatible)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Source+Serif+4:wght@600;700&display=swap');

    :root {
        color-scheme: light dark;
        --bg-1: #f6f7fb;
        --bg-2: #eef2ff;
        --ink-1: #101828;
        --ink-2: #475467;
        --ink-3: #667085;
        --brand-1: #3b82f6;
        --brand-2: #06b6d4;
        --card: rgba(255, 255, 255, 0.72);
        --card-border: rgba(16, 24, 40, 0.08);
        --shadow: 0 12px 30px rgba(16, 24, 40, 0.08);
        --radius: 16px;
        --sidebar-bg: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        --sidebar-ink: #e2e8f0;
        --sidebar-muted: #94a3b8;
        --link: var(--brand-1);
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --bg-1: #0b1220;
            --bg-2: #111827;
            --ink-1: #e5e7eb;
            --ink-2: #cbd5f5;
            --ink-3: #94a3b8;
            --brand-1: #60a5fa;
            --brand-2: #22d3ee;
            --card: rgba(15, 23, 42, 0.78);
            --card-border: rgba(148, 163, 184, 0.12);
            --shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
            --sidebar-bg: linear-gradient(180deg, #0b0f1a 0%, #0f172a 100%);
            --sidebar-ink: #dbeafe;
            --sidebar-muted: #94a3b8;
            --link: #93c5fd;
        }
    }

    html, body, [class*="css"], .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, li, span, label, div, button, input, textarea, select {
        font-family: "Space Grotesk", -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif !important;
    }

    body {
        background: radial-gradient(1100px 600px at 10% 0%, var(--bg-2), transparent 60%),
                    radial-gradient(900px 500px at 90% 10%, color-mix(in srgb, var(--bg-2) 70%, transparent), transparent 60%),
                    linear-gradient(180deg, color-mix(in srgb, var(--bg-2) 30%, transparent), var(--bg-1));
        color: var(--ink-1);
    }

    [data-testid="stAppViewContainer"] {
        color: var(--ink-1);
    }
    [data-testid="stAppViewContainer"] .stMarkdown,
    [data-testid="stAppViewContainer"] .stText,
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] li,
    [data-testid="stAppViewContainer"] span,
    [data-testid="stAppViewContainer"] label {
        color: var(--ink-1);
    }
    [data-testid="stAppViewContainer"] .stCaption,
    [data-testid="stAppViewContainer"] [data-testid="stCaption"] {
        color: var(--ink-3);
    }
    [data-testid="stAppViewContainer"] a {
        color: var(--brand-1);
    }

    .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 4rem !important;
        max-width: 1200px !important;
    }

    .section-card {
        background: var(--card);
        border: 1px solid var(--card-border);
        border-radius: var(--radius);
        padding: 1.4rem 1.6rem;
        box-shadow: var(--shadow);
        backdrop-filter: blur(10px);
        margin-bottom: 1.6rem;
        animation: fadeUp 0.4s ease-out both;
    }

    .hero {
        padding: 1.2rem 1.6rem 0.6rem 1.6rem;
        border-radius: 20px;
        background: linear-gradient(120deg, color-mix(in srgb, var(--bg-2) 55%, #ffffff) 0%, var(--bg-2) 45%, color-mix(in srgb, var(--bg-2) 40%, #ffffff) 100%);
        border: 1px solid color-mix(in srgb, var(--brand-1) 30%, transparent);
        box-shadow: 0 18px 40px color-mix(in srgb, var(--brand-1) 20%, transparent);
        margin-bottom: 1.6rem;
    }

    .hero-title {
        font-family: "Source Serif 4", Georgia, serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.4px;
        margin-bottom: 0.4rem;
        font-size: 2.2rem;
    }

    .hero-subtitle {
        color: var(--ink-2);
        font-size: 1rem;
    }

    .author-bar {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.8rem;
        padding: 0.2rem 0.4rem 0.9rem 0.4rem;
        color: var(--ink-2);
        font-size: 0.95rem;
    }
    .author-name {
        font-weight: 600;
        color: var(--ink-1);
    }
    .author-item a {
        color: var(--link);
        text-decoration: none;
    }
    .author-item a:hover {
        text-decoration: underline;
    }

    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.8rem;
    }

    .pill {
        background: color-mix(in srgb, var(--card) 90%, transparent);
        border: 1px solid var(--card-border);
        border-radius: 999px;
        padding: 0.3rem 0.75rem;
        font-size: 0.85rem;
        color: var(--ink-2);
    }

    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        border-radius: 10px !important;
        border: 1px solid var(--card-border) !important;
        background: color-mix(in srgb, var(--card) 80%, transparent) !important;
        padding-top: 0.45rem !important;
        padding-bottom: 0.45rem !important;
        color: var(--ink-1) !important;
    }
    .stTextInput input::placeholder {
        color: var(--ink-3) !important;
    }
    .stSelectbox div[data-baseweb="select"] span {
        color: var(--ink-1) !important;
    }

    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        padding: 0.55rem 1.3rem !important;
        background: linear-gradient(135deg, var(--brand-1), var(--brand-2)) !important;
        border: none !important;
        color: white !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 16px rgba(2, 132, 199, 0.2);
    }

    h1 {
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        margin-bottom: 1.2rem !important;
    }
    h2 {
        font-weight: 700 !important;
        margin-top: 0.2rem !important;
        margin-bottom: 1rem !important;
        border-bottom: 1px solid var(--card-border);
        padding-bottom: 0.4rem;
    }

    [data-testid="stSidebar"] {
        background: var(--sidebar-bg);
        color: var(--sidebar-ink);
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] li {
        color: var(--sidebar-ink);
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] [data-testid="stCaption"] {
        color: var(--sidebar-muted);
    }
    [data-testid="stSidebar"] a {
        color: var(--link);
    }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Modern Matplotlib Styling (Clean & Academic)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa", # Very light grey plot bg
    "savefig.facecolor": "white",
    "text.color": "#2d3436",
    "axes.labelcolor": "#2d3436",
    "xtick.color": "#636e72",
    "ytick.color": "#636e72",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#dfe6e9",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "lines.linewidth": 2,
    "lines.markersize": 6
})

# ==========================================
# Helpers
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

def get_model_list():
    models = []
    for attr_name in dir(MaterialModels):
        attr = getattr(MaterialModels, attr_name)
        if hasattr(attr, 'model_type') and hasattr(attr, 'category'):
            if hasattr(attr, 'param_names') and attr.param_names:
                models.append(attr_name)
    models.append("Hill")
    return sorted(models)

def format_param_latex(param_key):
    latex_map = {
        'mu': r'\mu',
        'alpha': r'\alpha',
        'N': r'N',
        'C1': r'C_1',
        'C2': r'C_2',
        'C3': r'C_3',
        'm1': r'm',
        'n1': r'n'
    }
    symbol = latex_map.get(param_key, param_key)
    return f"${symbol}$"

MODE_DISPLAY_MAP = {
    'UT': 'Uniaxial Tension (UT)',
    'ET': 'Equibiaxial Tension (ET)',
    'PS': 'Pure Shear (PS)'
}

# ==========================================
# State Management
# ==========================================
if 'model_confirmed' not in st.session_state:
    st.session_state['model_confirmed'] = False
if 'run_triggered' not in st.session_state:
    st.session_state['run_triggered'] = False
if 'calibration_done' not in st.session_state:
    st.session_state['calibration_done'] = False
if 'optimized_params_vec' not in st.session_state:
    st.session_state['optimized_params_vec'] = []
if 'prediction_triggered' not in st.session_state:
    st.session_state['prediction_triggered'] = False
if 'overlay_prediction' not in st.session_state:
    st.session_state['overlay_prediction'] = False 

def reset_model_confirmation():
    st.session_state['model_confirmed'] = False
    reset_run()

def reset_run():
    st.session_state['run_triggered'] = False
    st.session_state['calibration_done'] = False
    st.session_state['prediction_triggered'] = False
    st.session_state['optimized_params_vec'] = []

# ==========================================
# Main Application Flow
# ==========================================

# Title & Author Info

col_icon, col_title = st.columns([0.08, 0.92])
with col_icon:
    st.image("https://img.icons8.com/ios-filled/100/000000/dna-helix.png", width=50)
with col_title:
    st.markdown("""
    <div class="author-bar">
        <div class="author-name">Chongran Zhao</div>
        <div class="author-item">📧 chongranzhao@outlook.com</div>
        <div class="author-item">🌐 <a href="https://chongran-zhao.github.io" target="_blank" rel="noopener">chongran-zhao.github.io</a></div>
    </div>
    <div class="hero">
        <div class="hero-title">Hyperelastic Calibration</div>
        <div class="hero-subtitle">Parallel spring networks, data-driven fitting, and fast predictive curves.</div>
        <div class="pill-row">
            <div class="pill">Nonlinear mechanics</div>
            <div class="pill">Model selection</div>
            <div class="pill">Optimization</div>
            <div class="pill">Prediction</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# STEP 1: STRATEGY (Implicitly Fixed)
# -----------------------------------------------------------------------------
# Strategy is always "Calibration + Prediction", so we just show the next steps.

# -----------------------------------------------------------------------------
# STEP 2: DATA
# -----------------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("1. Experimental Data")
datasets = get_available_datasets()
data_config_valid = False
data_configs = []

if datasets:
    with st.container():
        # Card-like container logic handled by Streamlit natively now
        col_d1, col_d2 = st.columns([1, 2]) 
        
        with col_d1:
            author_options = ["Select..."] + list(datasets.keys())
            selected_author_raw = st.selectbox(
                "Author / Dataset", 
                author_options,
                key="author_select",
                on_change=reset_model_confirmation
            )
        
        if selected_author_raw != "Select...":
            selected_author = selected_author_raw
            
            with col_d2:
                st.caption("Select Loading Modes")
                available_modes = datasets[selected_author]
                
                # Compact checkboxes
                cols_modes = st.columns(len(available_modes))
                selected_modes = []
                
                for i, mode in enumerate(available_modes):
                    display_name = MODE_DISPLAY_MAP.get(mode, mode)
                    if cols_modes[i].checkbox(
                        display_name, 
                        value=False,
                        key=f"mode_{selected_author}_{mode}",
                        on_change=reset_model_confirmation
                    ):
                        selected_modes.append(mode)
            
            if selected_modes:
                data_configs = [{'author': selected_author, 'mode': m} for m in selected_modes]
                data_config_valid = True
            else:
                st.warning("⚠️ Select at least one mode.")
        else:
            st.info("👈 Choose a dataset to start.")

st.markdown('</div>', unsafe_allow_html=True)
if not data_config_valid:
    st.stop()

# -----------------------------------------------------------------------------
# STEP 3: MODEL
# -----------------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("2. Model Architecture")

col_num, _ = st.columns([1, 4]) 
with col_num:
    num_springs = st.number_input(
        "Parallel Springs", 
        min_value=0, value=0, step=1,
        key="num_springs",
        on_change=reset_model_confirmation
    )

if num_springs == 0:
    st.info("Add springs to configure the network.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

model_options = ["Select..."] + get_model_list()
strain_options = list(STRAIN_CONFIGS.keys())

all_springs_valid = True
execution_user_guesses = []
execution_param_names = []
execution_network = ParallelNetwork()

st.markdown("<hr style='margin: 1rem 0; opacity: 0.2;'>", unsafe_allow_html=True)

for i in range(int(num_springs)):
    # Compact Row
    c1, c2, c3 = st.columns([1.5, 1.5, 4]) 
    
    with c1:
        st.markdown(f"**Spring {i+1}**")
        m_type_raw = st.selectbox(
            "Model Type", 
            model_options, 
            key=f"model_{i}", 
            label_visibility="collapsed",
            on_change=reset_model_confirmation
        )
    
    if m_type_raw != "Select...":
        m_type = m_type_raw
        s_type = None
        
        if m_type == "Hill":
            with c2:
                st.markdown("**Strain**")
                s_type = st.selectbox(
                    "Strain Measure", 
                    strain_options, 
                    key=f"strain_{i}", 
                    label_visibility="collapsed",
                    on_change=reset_model_confirmation
                )
        else:
            pass
        
        # --- Instantiate ---
        if m_type == "Hill":
            func = MaterialModels.create_hill_model(s_type)
        else:
            func = getattr(MaterialModels, m_type)
        
        temp_net_single = ParallelNetwork()
        branch_name = f"{m_type}_{i+1}"
        temp_net_single.add_model(func, branch_name)
        
        current_param_names = temp_net_single.param_names
        current_defaults = temp_net_single.initial_guess
        
        with c3:
            if current_param_names:
                # Dynamic grid for params
                n_params = len(current_param_names)
                p_cols = st.columns(n_params)
                
                for p_idx, (p_name, p_def) in enumerate(zip(current_param_names, current_defaults)):
                    with p_cols[p_idx]:
                        short_name = p_name.split('_')[-1]
                        latex_label = format_param_latex(short_name)
                        default_str = f"{float(p_def):.4g}"
                        
                        val_str = st.text_input(
                            latex_label,
                            value="",
                            placeholder=default_str,
                            key=f"p_input_{i}_{p_idx}",
                            on_change=reset_run
                        )
                        
                        if val_str.strip() == "":
                            final_val = float(p_def)
                        else:
                            try:
                                final_val = float(val_str)
                            except ValueError:
                                st.error("Invalid")
                                final_val = float(p_def)
                        
                        execution_user_guesses.append(final_val)
                        execution_param_names.append(p_name)
            else:
                st.caption("No parameters required.")
        
        execution_network.add_model(func, branch_name)
    else:
        all_springs_valid = False
        with c3: st.info("👈 Select model type")
    
    if i < int(num_springs) - 1:
        st.markdown("<hr style='margin: 0.5rem 0; border-top: 1px dashed #eee;'>", unsafe_allow_html=True)

if not all_springs_valid:
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# STEP 4: EXECUTION
# -----------------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("3. Optimization")

col_opt1, col_opt2 = st.columns([1, 1])

with col_opt1:
    opt_method = st.selectbox(
        "Algorithm", 
        ["L-BFGS-B", "Differential Evolution", "Nelder-Mead", "Powell", "TNC", "SLSQP"],
        key="opt_method_final",
        on_change=reset_run
    )

with col_opt2:
    if not st.session_state['run_triggered']:
        if st.button("🚀 Start Calibration", use_container_width=True, type="primary"):
            st.session_state['run_triggered'] = True
            st.rerun()
    else:
        if st.button("🔄 Reset Analysis", use_container_width=True):
            reset_run()
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# RESULTS
# -----------------------------------------------------------------------------
if st.session_state['run_triggered']:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("## 📊 Analysis Results")
    
    # Calibration Run
    if not st.session_state['calibration_done']:
        with st.spinner("Running optimization..."):
            try:
                exp_data = load_experimental_data(data_configs)
                solver = Kinematics(execution_network, execution_network.param_names)
                optimizer = MaterialOptimizer(solver, exp_data)
                bounds = execution_network.bounds

                result = optimizer.fit(execution_user_guesses, bounds, method=opt_method)
                
                if result.success:
                    st.session_state['calibration_done'] = True
                    st.session_state['optimized_params_vec'] = result.x.tolist() 
                    st.session_state['final_loss'] = result.fun
                    st.rerun()
                else:
                    st.error(f"Optimization Failed: {result.message}")
                    st.stop()
            except Exception as e:
                st.error(f"Execution Error: {e}")
                st.stop()

    # Display Results
    if st.session_state['calibration_done']:
        
        # Success Banner
        st.markdown(f"""
        <div style="background-color: #e8f5e9; padding: 1rem; border-radius: 8px; border: 1px solid #c8e6c9; color: #2e7d32; margin-bottom: 1rem;">
            <strong>✅ Calibration Converged!</strong> &nbsp;&nbsp; Final Loss: <code>{st.session_state['final_loss']:.6f}</code>
        </div>
        """, unsafe_allow_html=True)
        
        # Main Results Layout
        c_res1, c_res2 = st.columns([1, 1.8]) 
        
        # --- LEFT: Parameters & Prediction Controls ---
        with c_res1:
            st.markdown("### Optimized Parameters")
            st.caption("You can fine-tune these values and run a new prediction.")
            
            with st.container():
                current_optimized_values = st.session_state['optimized_params_vec']
                new_param_values = []
                current_spring_idx = None
                
                for i, p_name in enumerate(execution_network.param_names):
                    parts = p_name.split('_')
                    param_short = parts[-1]
                    branch_idx = parts[-2] if len(parts)>=2 and parts[-2].isdigit() else "?"
                    
                    if branch_idx != current_spring_idx:
                        if current_spring_idx is not None: st.markdown("---")
                        st.markdown(f"**Spring {branch_idx}**")
                        current_spring_idx = branch_idx
                    
                    latex_label = format_param_latex(param_short)
                    val = current_optimized_values[i]
                    
                    new_val_str = st.text_input(
                        latex_label,
                        value=f"{val:.6g}",
                        key=f"final_edit_{i}"
                    )
                    
                    try:
                        new_param_values.append(float(new_val_str))
                    except:
                        new_param_values.append(val)
                
            st.markdown("### Prediction Mode")
            with st.container():
                # Check for unused data
                all_available_modes = datasets[selected_author]
                used_modes = selected_modes
                unused_modes = [m for m in all_available_modes if m not in used_modes]
                prediction_modes = []
                
                if unused_modes:
                    st.write("**Add Unused Data:**")
                    p_cols = st.columns(len(unused_modes))
                    for i, mode in enumerate(unused_modes):
                        display_name = MODE_DISPLAY_MAP.get(mode, mode)
                        if p_cols[i].checkbox(display_name, key=f"pred_mode_{mode}"):
                            prediction_modes.append(mode)
                else:
                    st.caption("No additional data available for this author.")

                overlay_option = st.checkbox("Overlay on Calibration", value=True)

                if st.button("⚡ Update Prediction", use_container_width=True):
                    st.session_state['optimized_params_vec'] = new_param_values
                    st.session_state['prediction_triggered'] = True
                    st.session_state['prediction_modes'] = prediction_modes
                    st.session_state['overlay_prediction'] = overlay_option
                    st.rerun()

        # --- RIGHT: Plotting ---
        with c_res2:
            plot_params_vec = st.session_state['optimized_params_vec']
            final_params_dict = dict(zip(execution_network.param_names, plot_params_vec))
            solver = Kinematics(execution_network, execution_network.param_names)
            
            # Create Plot
            fig, ax = plt.subplots(figsize=(7, 5))
            
            # Define distinct color palettes
            colors_calib = {'UT': '#2980b9', 'ET': '#c0392b', 'PS': '#27ae60'}
            colors_pred = {'UT': '#3498db', 'ET': '#e74c3c', 'PS': '#2ecc71'} # Slightly lighter
            
            # 1. Plot Calibration Data
            if not st.session_state['prediction_triggered'] or st.session_state['overlay_prediction']:
                calib_data = load_experimental_data(data_configs)
                for d in calib_data:
                    mode = d['mode']
                    stretch = d['stretch']
                    stress = d['stress_exp']
                    
                    # Exp dots
                    ax.plot(stretch, stress, 'o', mfc='none', label=f"Exp {mode}", 
                            color=colors_calib.get(mode, 'black'), markersize=6, alpha=0.6, markeredgewidth=1.2)
                    
                    # Model line
                    max_exp_lam = max(stretch)
                    smooth_stretch = np.linspace(1.0, max_exp_lam * 1.05, 100)
                    model_stress = []
                    from utils import get_stress_component
                    for lam in smooth_stretch:
                        if mode == 'UT': F = np.diag([lam, 1/np.sqrt(lam), 1/np.sqrt(lam)])
                        elif mode == 'ET': F = np.diag([lam, lam, 1/lam**2])
                        elif mode == 'PS': F = np.diag([lam, 1, 1/lam])
                        P_tensor = solver.get_1st_PK_stress(F, final_params_dict)
                        model_stress.append(get_stress_component(P_tensor, mode))
                    
                    ax.plot(smooth_stretch, model_stress, '-', linewidth=2, 
                            color=colors_calib.get(mode, 'black'), label=f"Fit {mode}")

            # 2. Plot Prediction Data
            if st.session_state['prediction_triggered'] and st.session_state.get('prediction_modes'):
                pred_configs = [{'author': selected_author, 'mode': m} for m in st.session_state['prediction_modes']]
                try:
                    pred_data = load_experimental_data(pred_configs)
                    for d in pred_data:
                        mode = d['mode']
                        stretch = d['stretch']
                        stress = d['stress_exp']
                        
                        # Exp dots (Square marker for prediction data)
                        ax.plot(stretch, stress, 's', mfc='none', label=f"Pred Exp {mode}", 
                                color=colors_pred.get(mode, 'black'), markersize=6, alpha=0.6, markeredgewidth=1.2)
                        
                        # Model line (Dashed)
                        max_exp_lam = max(stretch)
                        smooth_stretch = np.linspace(1.0, max_exp_lam * 1.05, 100)
                        model_stress = []
                        from utils import get_stress_component
                        for lam in smooth_stretch:
                            if mode == 'UT': F = np.diag([lam, 1/np.sqrt(lam), 1/np.sqrt(lam)])
                            elif mode == 'ET': F = np.diag([lam, lam, 1/lam**2])
                            elif mode == 'PS': F = np.diag([lam, 1, 1/lam])
                            P_tensor = solver.get_1st_PK_stress(F, final_params_dict)
                            model_stress.append(get_stress_component(P_tensor, mode))
                        
                        ax.plot(smooth_stretch, model_stress, '--', linewidth=2, 
                                color=colors_pred.get(mode, 'black'), label=f"Pred {mode}")
                                
                except Exception as e:
                    st.error(f"Error loading prediction data: {e}")

            # Styling the plot
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best', frameon=True, fancybox=True, fontsize=9)
            
            ax.set_xlabel(r"Stretch $\lambda$ [-]", fontweight='bold')
            ax.set_ylabel(r"Nominal Stress $P$ [MPa]", fontweight='bold')
            ax.set_title("Experimental vs Model Response", pad=15)
            
            st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)
