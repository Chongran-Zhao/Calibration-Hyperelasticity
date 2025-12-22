import streamlit as st
import os
import sys
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))

from utils import load_experimental_data, get_deformation_gradient, get_stress_components
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

    .status-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.8rem;
        background: var(--card);
        border: 1px solid var(--card-border);
        border-radius: 999px;
        padding: 0.4rem 0.9rem;
        margin: 0.4rem 0 1rem 0;
        box-shadow: var(--shadow);
        font-size: 0.9rem;
        color: var(--ink-2);
    }
    .status-item strong {
        color: var(--ink-1);
    }
    .badge-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin: 0.6rem 0 0.2rem 0;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-size: 0.8rem;
        border: 1px solid var(--card-border);
    }
    .badge-ok {
        background: color-mix(in srgb, #22c55e 20%, transparent);
        color: var(--ink-1);
    }
    .badge-off {
        background: color-mix(in srgb, #94a3b8 20%, transparent);
        color: var(--ink-3);
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
    if not os.path.exists(data_dir):
        return {}

    data_h5 = os.path.join(data_dir, "data.h5")
    if os.path.exists(data_h5):
        try:
            import h5py
            with h5py.File(data_h5, "r") as h5f:
                for author in h5f.keys():
                    modes = list(h5f[author].keys())
                    if modes:
                        datasets[author] = sorted(modes)
            if datasets:
                return datasets
        except Exception:
            pass

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

def get_model_display_name(model_key):
    return model_key

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
    if param_key in latex_map:
        return latex_map[param_key]

    match = re.match(r"^([a-zA-Z]+)(\d+)$", param_key)
    if match:
        base, idx = match.groups()
        base_symbol = latex_map.get(base, base)
        return rf"{base_symbol}_{{{idx}}}"

    return rf"\mathrm{{{param_key}}}"

MODE_DISPLAY_MAP = {
    'UT': 'Uniaxial Tension (UT)',
    'ET': 'Equibiaxial Tension (ET)',
    'PS': 'Pure Shear (PS)',
    'BT': 'Biaxial Tension (BT)'
}

DATASET_REFERENCES = {
    "Treloar_1944": ("Treloar 1944, Rubber Chemistry and Technology", "https://doi.org/10.5254/1.3546701"),
    "Kawabata_1981": ("Kawabata et al. 1981, Macromolecules", "https://doi.org/10.1021/ma50002a032"),
    "Meunier_2008": ("Meunier et al. 2008, Polymer Testing", "https://doi.org/10.1016/j.polymertesting.2008.05.011"),
    "James_1975": ("James et al. 1975, J. Appl. Polym. Sci.", "https://doi.org/10.1002/app.1975.070190723"),
    "Jones_1975": ("Jones & Treloar 1975, J. Phys. D", "https://doi.org/10.1088/0022-3727/8/11/007"),
    "Kawamura_2001": ("Kawamura et al. 2001, Macromolecules", "https://doi.org/10.1021/ma002165y"),
    "Katashima_2012": ("Katashima et al. 2012, Soft Matter", "https://doi.org/10.1039/c2sm25340b"),
}

def format_mode_label(mode_key):
    if mode_key.startswith('BT_lambda_'):
        lam2 = mode_key.replace('BT_lambda_', '').replace('d', '.')
        return f"Biaxial Tension, λ2={lam2}"
    return MODE_DISPLAY_MAP.get(mode_key, mode_key)

def format_mode_label_short(mode_key):
    label = format_mode_label(mode_key)
    return label.replace(" (UT)", "").replace(" (ET)", "").replace(" (PS)", "")

def get_stress_type_label(stress_type):
    if stress_type == "cauchy":
        return "Cauchy stress"
    return "Nominal stress"

def get_dataset_reference(author):
    if not author:
        return None
    return DATASET_REFERENCES.get(author)

def render_mode_checkboxes(modes, key_prefix, on_change=None, max_cols=3):
    selected = []
    for row_start in range(0, len(modes), max_cols):
        row_modes = modes[row_start:row_start + max_cols]
        cols = st.columns(len(row_modes))
        for i, mode in enumerate(row_modes):
            display_name = format_mode_label_short(mode)
            if cols[i].checkbox(
                display_name,
                value=False,
                key=f"{key_prefix}_{mode}",
                on_change=on_change
            ):
                selected.append(mode)
    return selected

def render_status_strip(selected_author, selected_modes, num_springs):
    author_text = selected_author if selected_author else "None"
    modes_text = ", ".join(selected_modes) if selected_modes else "None"
    springs_text = str(num_springs) if num_springs else "0"
    st.markdown(
        f"""
        <div class="status-strip">
            <div class="status-item"><strong>Dataset:</strong> {author_text}</div>
            <div class="status-item"><strong>Modes:</strong> {modes_text}</div>
            <div class="status-item"><strong>Springs:</strong> {springs_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_step_badges(data_ok, model_ok, run_ok):
    def badge(text, ok):
        cls = "badge-ok" if ok else "badge-off"
        return f'<span class="badge {cls}">{text}</span>'
    st.markdown(
        f"""
        <div class="badge-row">
            {badge("Step 1: Data", data_ok)}
            {badge("Step 2: Model", model_ok)}
            {badge("Step 3: Optimize", run_ok)}
        </div>
        """,
        unsafe_allow_html=True,
    )

def get_selected_modes_for_author(author):
    modes = []
    if not author:
        return modes
    prefix = f"mode_{author}_"
    for key, val in st.session_state.items():
        if key.startswith(prefix) and val:
            modes.append(key.replace(prefix, ""))
    return modes

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

render_status_strip(
    st.session_state.get("author_select") if st.session_state.get("author_select") != "Select..." else None,
    get_selected_modes_for_author(st.session_state.get("author_select") if st.session_state.get("author_select") != "Select..." else None),
    st.session_state.get("num_springs"),
)

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
selected_author = None
selected_modes = []

if datasets:
    with st.container():
        # Card-like container logic handled by Streamlit natively now
        col_d1, col_d2 = st.columns([1, 2]) 
        preview_slot = st.empty()
        
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
                use_interactive_preview = st.toggle("Interactive preview", value=True, key="preview_toggle")
                available_modes = datasets[selected_author]
                
                # Compact checkboxes with row wrapping
                selected_modes = render_mode_checkboxes(
                    available_modes,
                    key_prefix=f"mode_{selected_author}",
                    on_change=reset_model_confirmation,
                    max_cols=3
                )
            
            if selected_modes:
                data_configs = [{'author': selected_author, 'mode': m} for m in selected_modes]
                data_config_valid = True
                preview_data = load_experimental_data(data_configs)
                with preview_slot.container():
                    st.markdown("#### Data Preview")
                    preview_cols = st.columns(len(preview_data))
                    for i, d in enumerate(preview_data):
                        mode_key = d.get('mode_raw', d['mode'])
                        stress_type = d.get('stress_type', 'PK1')
                        stretch = d['stretch']
                        stress = d['stress_exp']
                        title_mode = format_mode_label(mode_key)

                        if use_interactive_preview:
                            try:
                                import plotly.graph_objects as go
                                fig = go.Figure()
                                if d['mode'] == 'BT':
                                    if np.ndim(stress) == 1:
                                        fig.add_trace(go.Scatter(x=stretch, y=stress, mode="markers", name="11"))
                                    else:
                                        fig.add_trace(go.Scatter(x=stretch, y=stress[:, 0], mode="markers", name="11"))
                                        fig.add_trace(go.Scatter(x=stretch, y=stress[:, 1], mode="markers", name="22"))
                                    fig.update_xaxes(title_text="lambda_1")
                                    if stress_type == "cauchy":
                                        fig.update_yaxes(title_text="sigma_11 / sigma_22")
                                    else:
                                        fig.update_yaxes(title_text="P_11 / P_22")
                                else:
                                    fig.add_trace(go.Scatter(x=stretch, y=stress, mode="markers", name="data"))
                                    fig.update_xaxes(title_text="lambda")
                                    fig.update_yaxes(title_text=get_stress_type_label(stress_type))
                                fig.update_layout(
                                    title=title_mode,
                                    height=220,
                                    margin=dict(l=10, r=10, t=30, b=10),
                                    showlegend=False,
                                )
                                preview_cols[i].plotly_chart(fig, width="stretch")
                            except Exception:
                                fig, ax = plt.subplots(figsize=(3.0, 2.2))
                                if d['mode'] == 'BT':
                                    if np.ndim(stress) == 1:
                                        ax.plot(stretch, stress, 'o', mfc='none', color='#8e44ad')
                                    else:
                                        ax.plot(stretch, stress[:, 0], 'o', mfc='none', color='#8e44ad', label='11')
                                        ax.plot(stretch, stress[:, 1], '^', mfc='none', color='#a569bd', label='22')
                                        ax.legend(frameon=False, fontsize=8)
                                    ax.set_xlabel(r"Stretch $\lambda_1$")
                                    if stress_type == "cauchy":
                                        ax.set_ylabel(r"$\sigma_{11}$ / $\sigma_{22}$")
                                    else:
                                        ax.set_ylabel(r"$P_{11}$ / $P_{22}$")
                                else:
                                    ax.plot(stretch, stress, 'o', mfc='none', color='#2980b9')
                                    ax.set_xlabel(r"Stretch $\lambda$")
                                    ax.set_ylabel(get_stress_type_label(stress_type))
                                ax.set_title(title_mode, fontsize=10)
                                ax.grid(True, linestyle="--", alpha=0.3)
                                preview_cols[i].pyplot(fig, width="stretch")
                        else:
                            fig, ax = plt.subplots(figsize=(3.0, 2.2))
                            if d['mode'] == 'BT':
                                if np.ndim(stress) == 1:
                                    ax.plot(stretch, stress, 'o', mfc='none', color='#8e44ad')
                                else:
                                    ax.plot(stretch, stress[:, 0], 'o', mfc='none', color='#8e44ad', label='11')
                                    ax.plot(stretch, stress[:, 1], '^', mfc='none', color='#a569bd', label='22')
                                    ax.legend(frameon=False, fontsize=8)
                                ax.set_xlabel(r"Stretch $\lambda_1$")
                                if stress_type == "cauchy":
                                    ax.set_ylabel(r"$\sigma_{11}$ / $\sigma_{22}$")
                                else:
                                    ax.set_ylabel(r"$P_{11}$ / $P_{22}$")
                            else:
                                ax.plot(stretch, stress, 'o', mfc='none', color='#2980b9')
                                ax.set_xlabel(r"Stretch $\lambda$")
                                ax.set_ylabel(get_stress_type_label(stress_type))
                            ax.set_title(title_mode, fontsize=10)
                            ax.grid(True, linestyle="--", alpha=0.3)
                            preview_cols[i].pyplot(fig, width="stretch")
            else:
                preview_slot.empty()
                st.warning("⚠️ Select at least one mode.")
        else:
            preview_slot.empty()
            st.info("👈 Choose a dataset to start.")
        reference = get_dataset_reference(selected_author)
        if reference:
            label, url = reference
            st.markdown(f"#### Reference\n[{label}]({url})")

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

model_keys = get_model_list()
model_display = [get_model_display_name(m) for m in model_keys]
model_options = ["Select..."] + model_display
model_display_to_key = {get_model_display_name(m): m for m in model_keys}
strain_options = list(STRAIN_CONFIGS.keys())

all_springs_valid = True
execution_user_guesses = []
execution_param_names = []
execution_network = ParallelNetwork()

st.markdown("<hr style='margin: 1rem 0; opacity: 0.2;'>", unsafe_allow_html=True)

for i in range(int(num_springs)):
    # Compact Row
    c1, c2, c3, c4 = st.columns([1.5, 1.5, 4, 3])
    
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
        m_type = model_display_to_key.get(m_type_raw, m_type_raw)
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
        elif m_type == "Ogden":
            with c2:
                st.markdown("**Terms**")
                ogden_terms = st.number_input(
                    "Ogden Terms",
                    min_value=1,
                    max_value=6,
                    value=1,
                    step=1,
                    key=f"ogden_terms_{i}",
                    label_visibility="collapsed",
                    on_change=reset_model_confirmation
                )
        else:
            pass
        
        # --- Instantiate ---
        if m_type == "Hill":
            func = MaterialModels.create_hill_model(s_type)
        elif m_type == "Ogden":
            func = MaterialModels.create_ogden_model(ogden_terms)
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

                        st.latex(latex_label)
                        val_str = st.text_input(
                            "param",
                            value="",
                            placeholder=default_str,
                            key=f"p_input_{i}_{p_idx}",
                            on_change=reset_run,
                            label_visibility="collapsed"
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

        with c4:
            st.caption("Formula")
            formula = getattr(func, "formula", "No formula")
            st.latex(formula)
            if m_type == "Hill":
                strain_formula = getattr(func, "strain_formula", "")
                if strain_formula:
                    st.caption("Strain measure")
                    st.latex(strain_formula)
        
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
model_config_valid = True

# -----------------------------------------------------------------------------
# STEP 4: EXECUTION
# -----------------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("3. Optimization")
render_step_badges(data_config_valid, model_config_valid, st.session_state['run_triggered'])

col_opt1, col_opt2 = st.columns([1, 1])

with col_opt1:
    opt_method = st.selectbox(
        "Algorithm", 
        ["L-BFGS-B", "trust-constr", "CG", "Newton-CG"],
        key="opt_method_final",
        on_change=reset_run
    )

with col_opt2:
    if not st.session_state['run_triggered']:
        if st.button("🚀 Start Calibration", width="stretch", type="primary"):
            st.session_state['run_triggered'] = True
            st.rerun()
    else:
        if st.button("🔄 Reset Analysis", width="stretch"):
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

                    st.latex(latex_label)
                    new_val_str = st.text_input(
                        "param",
                        value=f"{val:.6g}",
                        key=f"final_edit_{i}",
                        label_visibility="collapsed"
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
                    prediction_modes = render_mode_checkboxes(
                        unused_modes,
                        key_prefix="pred_mode",
                        on_change=None,
                        max_cols=3
                    )
                else:
                    st.caption("No additional data available for this author.")

                overlay_option = st.checkbox("Overlay on Calibration", value=True)

                if st.button("⚡ Update Prediction", width="stretch"):
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
            colors_calib = {'UT': '#2980b9', 'ET': '#c0392b', 'PS': '#27ae60', 'BT': '#8e44ad'}
            colors_pred = {'UT': '#3498db', 'ET': '#e74c3c', 'PS': '#2ecc71', 'BT': '#a569bd'}
            
            # 1. Plot Calibration Data
            if not st.session_state['prediction_triggered'] or st.session_state['overlay_prediction']:
                calib_data = load_experimental_data(data_configs)
                for d in calib_data:
                    mode = d['mode']
                    stretch = d['stretch']
                    stress = d['stress_exp']
                    mode_label = d.get('mode_raw', mode)
                    stress_type = d.get('stress_type', 'PK1')
                    stress_label = get_stress_type_label(stress_type)

                    if mode == 'BT':
                        stretch_secondary = d.get('stretch_secondary')
                        if np.ndim(stress) == 1:
                            ax.plot(stretch, stress, 'o', mfc='none', label=f"Exp {mode_label} P11 ({stress_label})",
                                    color=colors_calib.get(mode, 'black'), markersize=6, alpha=0.6, markeredgewidth=1.2)
                        else:
                            ax.plot(stretch, stress[:, 0], 'o', mfc='none', label=f"Exp {mode_label} P11 ({stress_label})",
                                    color=colors_calib.get(mode, 'black'), markersize=6, alpha=0.6, markeredgewidth=1.2)
                            ax.plot(stretch, stress[:, 1], '^', mfc='none', label=f"Exp {mode_label} P22 ({stress_label})",
                                    color=colors_pred.get(mode, 'black'), markersize=6, alpha=0.6, markeredgewidth=1.2)

                        sort_idx = np.argsort(stretch)
                        lam1_sorted = np.array(stretch)[sort_idx]
                        lam2_sorted = np.array(stretch_secondary)[sort_idx]
                        model_stress_11 = []
                        model_stress_22 = []
                        for lam1, lam2 in zip(lam1_sorted, lam2_sorted):
                            F = get_deformation_gradient((lam1, lam2), mode)
                            if stress_type == 'cauchy':
                                stress_tensor = solver.get_Cauchy_stress(F, final_params_dict)
                            else:
                                stress_tensor = solver.get_1st_PK_stress(F, final_params_dict)
                            comps = get_stress_components(stress_tensor, mode)
                            model_stress_11.append(comps[0])
                            model_stress_22.append(comps[1])

                        ax.plot(lam1_sorted, model_stress_11, '-', linewidth=2,
                                color=colors_calib.get(mode, 'black'), label=f"Fit {mode_label} P11 ({stress_label})")
                        if np.ndim(stress) > 1:
                            ax.plot(lam1_sorted, model_stress_22, '--', linewidth=2,
                                    color=colors_pred.get(mode, 'black'), label=f"Fit {mode_label} P22 ({stress_label})")
                    else:
                        # Exp dots
                        ax.plot(stretch, stress, 'o', mfc='none', label=f"Exp {mode} ({stress_label})",
                                color=colors_calib.get(mode, 'black'), markersize=6, alpha=0.6, markeredgewidth=1.2)

                        # Model line
                        max_exp_lam = max(stretch)
                        smooth_stretch = np.linspace(1.0, max_exp_lam * 1.05, 100)
                        model_stress = []
                        for lam in smooth_stretch:
                            F = get_deformation_gradient(lam, mode)
                            if stress_type == 'cauchy':
                                stress_tensor = solver.get_Cauchy_stress(F, final_params_dict)
                            else:
                                stress_tensor = solver.get_1st_PK_stress(F, final_params_dict)
                            model_stress.append(get_stress_components(stress_tensor, mode)[0])

                        ax.plot(smooth_stretch, model_stress, '-', linewidth=2,
                                color=colors_calib.get(mode, 'black'), label=f"Fit {mode} ({stress_label})")

            # 2. Plot Prediction Data
            if st.session_state['prediction_triggered'] and st.session_state.get('prediction_modes'):
                pred_configs = [{'author': selected_author, 'mode': m} for m in st.session_state['prediction_modes']]
                try:
                    pred_data = load_experimental_data(pred_configs)
                    for d in pred_data:
                        mode = d['mode']
                        stretch = d['stretch']
                        stress = d['stress_exp']
                        mode_label = d.get('mode_raw', mode)
                        stress_type = d.get('stress_type', 'PK1')
                        stress_label = get_stress_type_label(stress_type)

                        if mode == 'BT':
                            stretch_secondary = d.get('stretch_secondary')
                            if np.ndim(stress) == 1:
                                ax.plot(stretch, stress, 's', mfc='none', label=f"Pred Exp {mode_label} P11 ({stress_label})",
                                        color=colors_pred.get(mode, 'black'), markersize=6, alpha=0.6, markeredgewidth=1.2)
                            else:
                                ax.plot(stretch, stress[:, 0], 's', mfc='none', label=f"Pred Exp {mode_label} P11 ({stress_label})",
                                        color=colors_pred.get(mode, 'black'), markersize=6, alpha=0.6, markeredgewidth=1.2)
                                ax.plot(stretch, stress[:, 1], 'v', mfc='none', label=f"Pred Exp {mode_label} P22 ({stress_label})",
                                        color=colors_calib.get(mode, 'black'), markersize=6, alpha=0.6, markeredgewidth=1.2)

                            sort_idx = np.argsort(stretch)
                            lam1_sorted = np.array(stretch)[sort_idx]
                            lam2_sorted = np.array(stretch_secondary)[sort_idx]
                            model_stress_11 = []
                            model_stress_22 = []
                            for lam1, lam2 in zip(lam1_sorted, lam2_sorted):
                                F = get_deformation_gradient((lam1, lam2), mode)
                                if stress_type == 'cauchy':
                                    stress_tensor = solver.get_Cauchy_stress(F, final_params_dict)
                                else:
                                    stress_tensor = solver.get_1st_PK_stress(F, final_params_dict)
                                comps = get_stress_components(stress_tensor, mode)
                                model_stress_11.append(comps[0])
                                model_stress_22.append(comps[1])

                            ax.plot(lam1_sorted, model_stress_11, '--', linewidth=2,
                                    color=colors_pred.get(mode, 'black'), label=f"Pred {mode_label} P11 ({stress_label})")
                            if np.ndim(stress) > 1:
                                ax.plot(lam1_sorted, model_stress_22, ':', linewidth=2,
                                        color=colors_calib.get(mode, 'black'), label=f"Pred {mode_label} P22 ({stress_label})")
                        else:
                            # Exp dots (Square marker for prediction data)
                            ax.plot(stretch, stress, 's', mfc='none', label=f"Pred Exp {mode} ({stress_label})",
                                    color=colors_pred.get(mode, 'black'), markersize=6, alpha=0.6, markeredgewidth=1.2)

                            # Model line (Dashed)
                            max_exp_lam = max(stretch)
                            smooth_stretch = np.linspace(1.0, max_exp_lam * 1.05, 100)
                            model_stress = []
                            for lam in smooth_stretch:
                                F = get_deformation_gradient(lam, mode)
                                if stress_type == 'cauchy':
                                    stress_tensor = solver.get_Cauchy_stress(F, final_params_dict)
                                else:
                                    stress_tensor = solver.get_1st_PK_stress(F, final_params_dict)
                                model_stress.append(get_stress_components(stress_tensor, mode)[0])

                            ax.plot(smooth_stretch, model_stress, '--', linewidth=2,
                                    color=colors_pred.get(mode, 'black'), label=f"Pred {mode} ({stress_label})")

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
