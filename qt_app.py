import os
import sys
import tempfile
import re
import gc
import warnings
import multiprocessing
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import sympy as sp
try:
    from platformdirs import user_cache_dir
except Exception:
    user_cache_dir = None

from PySide6.QtCore import Qt, QThread, Signal, QUrl, QSize, QEvent
from PySide6.QtGui import QFont, QPalette, QIcon, QColor, QFontDatabase, QDesktopServices, QPainter, QPen, QFontMetrics
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QApplication,
    QAbstractSpinBox,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QListWidget,
    QListView,
    QFileDialog,
    QInputDialog,
    QListWidgetItem,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

def _default_cache_root():
    if user_cache_dir:
        return Path(user_cache_dir("HyperelasticCalibration"))
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "HyperelasticCalibration"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "HyperelasticCalibration"
    return Path.home() / ".cache" / "HyperelasticCalibration"


cache_root = _default_cache_root()
try:
    cache_root.mkdir(parents=True, exist_ok=True)
except OSError:
    cache_root = Path(tempfile.gettempdir()) / "HyperelasticCalibration"
    cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
# Force single-threaded math to avoid multiprocessing resource_tracker warnings.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: process died unexpectedly.*",
    category=UserWarning,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Add src to path (support bundled app)
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = getattr(sys, "_MEIPASS", current_dir)
sys.path.append(os.path.join(base_dir, "src"))

from utils import load_experimental_data, get_deformation_gradient, get_stress_components
from material_models import MaterialModels
from generalized_strains import STRAIN_CONFIGS
from parallel_springs import ParallelNetwork
from kinematics import Kinematics
from optimization import MaterialOptimizer

MODE_DISPLAY_MAP = {
    "UT": "Uniaxial Tension",
    "UC": "Uniaxial Compression",
    "ET": "Equibiaxial Tension",
    "PS": "Pure Shear",
    "SS": "Simple Shear",
    "CSS": "Compound Simple Shear",
    "BT": "Biaxial Tension",
}

COMPONENT_AUTHOR_CONFIG = {
    "Budday_2017": {
        "component_label": "Region",
        "mode_prefixes": ("SS_", "UT_T_", "UT_C_", "CSS_"),
        "mode_labels": {
            "SS_": "Simple Shear",
            "UT_T_": "Uniaxial Tension",
            "UT_C_": "Uniaxial Compression",
            "CSS_": "Compound Simple Shear",
        },
    }
}

DATASET_REFERENCES = {
    "Treloar_1944": ("Treloar 1944, Rubber Chemistry and Technology", "https://doi.org/10.5254/1.3546701"),
    "Kawabata_1981": ("Kawabata et al. 1981, Macromolecules", "https://doi.org/10.1021/ma50002a032"),
    "Meunier_2008": ("Meunier et al. 2008, Polymer Testing", "https://doi.org/10.1016/j.polymertesting.2008.05.011"),
    "James_1975": ("James et al. 1975, J. Appl. Polym. Sci.", "https://doi.org/10.1002/app.1975.070190723"),
    "Jones_1975": ("Jones & Treloar 1975, J. Phys. D", "https://doi.org/10.1088/0022-3727/8/11/007"),
    "Kawamura_2001": ("Kawamura et al. 2001, Macromolecules", "https://doi.org/10.1021/ma002165y"),
    "Katashima_2012": ("Katashima et al. 2012, Soft Matter", "https://doi.org/10.1039/c2sm25340b"),
    "Budday_2017": ("Budday et al. 2017, Acta Biomaterialia", "https://doi.org/10.1016/j.actbio.2016.10.036"),
}

DATASET_N_GUESS = {
    "Treloar_1944": 10.0,
    "Jones_1975": 6.0,
    "James_1975": 10.0,
    "Kawamura_2001": 8.0,
    "Katashima_2012": 8.0,
    "Kawabata_1981": 6.0,
    "Meunier_2008": 6.0,
}

MODEL_REFERENCES = {
    "NeoHookean": ("Treloar 1943, Rubber Chemistry and Technology", "https://doi.org/10.5254/1.3540158"),
    "MooneyRivlin": ("Mooney 1940, Journal of Applied Physics", "https://doi.org/10.1063/1.1712836"),
    "Yeoh": ("Yeoh 1993, Rubber Chemistry and Technology", "https://doi.org/10.5254/1.3538343"),
    "ArrudaBoyce": ("Arruda & Boyce 1993, J. Mech. Phys. Solids", "https://doi.org/10.1016/0022-5096(93)90013-6"),
    "Ogden": ("Ogden 1972, Proc. R. Soc. A", "https://doi.org/10.1098/rspa.1972.0026"),
    "ModifiedOgden": ("Budday et al. 2017, Acta Biomaterialia", "https://doi.org/10.1016/j.actbio.2016.10.036"),
    "Hill": (
        "Liu, Guan, Zhao, Luo 2024, Computer Methods in Applied Mechanics and Engineering 430:117248",
        "https://doi.org/10.1016/j.cma.2024.117248",
    ),
    "ZhanGaussian": ("Zhan et al. 2022, J. Mech. Phys. Solids", "https://www.sciencedirect.com/science/article/abs/pii/S0022509622003325"),
    "ZhanNonGaussian": ("Zhan et al. 2022, J. Mech. Phys. Solids", "https://www.sciencedirect.com/science/article/abs/pii/S0022509622003325"),
}

MODEL_DISPLAY_NAMES = {
    "ZhanGaussian": "Zhan (Gaussian)",
    "ZhanNonGaussian": "Zhan (non-Gaussian)",
    "ModifiedOgden": "Modified Ogden",
}
MODEL_DISPLAY_LOOKUP = {v: k for k, v in MODEL_DISPLAY_NAMES.items()}


def get_available_datasets():
    data_dir = os.path.join(base_dir, "data")
    datasets = {}
    data_h5 = os.path.join(data_dir, "data.h5")
    if os.path.exists(data_h5):
        try:
            import h5py
            with h5py.File(data_h5, "r") as h5f:
                for author in h5f.keys():
                    modes = list(h5f[author].keys())
                    if modes:
                        datasets[author] = sorted(modes)
            return datasets
        except Exception:
            pass
    return datasets


def format_mode_label(mode_key):
    if mode_key.startswith("BT_lambda_"):
        lam2 = mode_key.replace("BT_lambda_", "").replace("d", ".")
        return f"Biaxial Tension, λ₂={lam2}"
    if mode_key.startswith("UT_T_"):
        return "Uniaxial Tension"
    if mode_key.startswith("UT_C_"):
        return "Uniaxial Compression"
    if mode_key.startswith("CSS_") and "_lambda_" in mode_key:
        lam = mode_key.split("_lambda_")[-1].replace("d", ".")
        return f"Simple shear under λ₁ = {lam}"
    if mode_key.startswith("UT_"):
        return "Uniaxial (Tension/Compression)"
    if mode_key.startswith("SS_"):
        return "Simple Shear"
    return MODE_DISPLAY_MAP.get(mode_key, mode_key)

def format_component_label(component_key):
    return component_key.replace("_", " ").title()

def extract_component_from_mode(mode_key, prefixes):
    for prefix in prefixes:
        if prefix == "CSS_" and mode_key.startswith(prefix) and "_lambda_" in mode_key:
            return mode_key[len(prefix):].split("_lambda_")[0]
        if mode_key.startswith(prefix):
            return mode_key[len(prefix):]
    return None


def get_stress_type_label(stress_type):
    return "Cauchy stress" if stress_type == "cauchy" else "Nominal stress"

def get_bt_component_labels(stress_type):
    if stress_type == "cauchy":
        return r"$\sigma_{11}$", r"$\sigma_{22}$"
    return r"$P_{11}$", r"$P_{22}$"

def get_component_label(stress_type, component):
    comp_11, comp_22 = get_bt_component_labels(stress_type)
    if component == "22":
        return comp_22
    if component == "11":
        return comp_11
    return get_uniaxial_component_label(stress_type)

def get_bt_diff_label(stress_type):
    if stress_type == "cauchy":
        return r"$\sigma_{11}-\sigma_{22}$"
    return r"$P_{11}-P_{22}$"

def get_uniaxial_component_label(stress_type):
    return r"$\sigma_{11}$" if stress_type == "cauchy" else r"$P_{11}$"

def get_shear_component_label(stress_type):
    return r"$\sigma_{12}$" if stress_type == "cauchy" else r"$P_{12}$"

def get_mode_xlabel(mode):
    return r"$\gamma$" if mode == "SS" else r"$\lambda_1$"

def choose_xlabel_for_modes(modes):
    if modes and all(m in ("SS", "CSS") for m in modes):
        return r"$\gamma$"
    return r"$\lambda_1$"

def parse_lambda2(mode_raw):
    if mode_raw.startswith("BT_lambda_"):
        return mode_raw.replace("BT_lambda_", "").replace("d", ".")
    return None

def apply_theme_to_axis(ax):
    palette = QApplication.palette()
    text = palette.color(QPalette.WindowText)
    grid = palette.color(QPalette.Mid)
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)
    ax.tick_params(colors=(text.redF(), text.greenF(), text.blueF(), 1.0))
    ax.xaxis.label.set_color((text.redF(), text.greenF(), text.blueF(), 1.0))
    ax.yaxis.label.set_color((text.redF(), text.greenF(), text.blueF(), 1.0))
    ax.title.set_color((text.redF(), text.greenF(), text.blueF(), 1.0))
    for spine in ax.spines.values():
        spine.set_color((grid.redF(), grid.greenF(), grid.blueF(), 1.0))
    ax.grid(True, linestyle="--", alpha=0.25, color=(grid.redF(), grid.greenF(), grid.blueF(), 1.0))


def build_app_palette():
    palette = QPalette()
    window = QColor("#F5F5F7")
    base = QColor("#FFFFFF")
    alt_base = QColor("#F2F2F7")
    text = QColor("#1C1C1E")
    mid = QColor("#E5E5EA")
    dark = QColor("#D1D1D6")
    button = QColor("#FFFFFF")
    highlight = QColor("#007AFF")
    link = QColor("#007AFF")
    disabled = QColor("#8E8E93")
    tooltip_base = QColor("#FFFFFF")
    tooltip_text = text
    bright = QColor("#FF3B30")

    palette.setColor(QPalette.Window, window)
    palette.setColor(QPalette.WindowText, text)
    palette.setColor(QPalette.Base, base)
    palette.setColor(QPalette.AlternateBase, alt_base)
    palette.setColor(QPalette.ToolTipBase, tooltip_base)
    palette.setColor(QPalette.ToolTipText, tooltip_text)
    palette.setColor(QPalette.Text, text)
    palette.setColor(QPalette.Button, button)
    palette.setColor(QPalette.ButtonText, text)
    palette.setColor(QPalette.BrightText, bright)
    palette.setColor(QPalette.Mid, mid)
    palette.setColor(QPalette.Dark, dark)
    palette.setColor(QPalette.Highlight, highlight)
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.Link, link)
    palette.setColor(QPalette.LinkVisited, link)
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, disabled)
    if hasattr(QPalette, "PlaceholderText"):
        palette.setColor(QPalette.PlaceholderText, disabled)
    return palette


def build_app_stylesheet():
    return (
        "QWidget { color: palette(windowtext); font-family: -apple-system, \"SF Pro Text\", \"Helvetica Neue\", sans-serif; }"
        "QMainWindow, QDialog { background: #F5F5F7; }"
        "QFrame { background: transparent; }"
        "QScrollArea, QScrollArea > QWidget, QScrollArea > QWidget > QWidget { background: transparent; border: none; }"
        "QGroupBox { background: #FFFFFF; border: 1px solid #E5E5EA; border-radius: 12px; margin-top: 18px; padding-top: 8px; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 16px; padding: 0 2px;"
        " color: #1C1C1E; background: transparent; font-size: 15px; font-weight: 700; }"
        "QListWidget, QTableView, QTreeView { background: #FFFFFF; border: 1px solid #E5E5EA; border-radius: 10px; }"
        "QListWidget::item:selected { background: #EAF3FF; color: #1C1C1E; border-radius: 6px; }"
        "QHeaderView::section { background: #F2F2F7; color: #3A3A3C; padding: 6px 8px; border: none; font-weight: 600; }"
        "QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QAbstractSpinBox {"
        " background: #FFFFFF; color: #1C1C1E; border: 1px solid #D1D1D6; border-radius: 8px; padding: 6px 10px; }"
        "QLineEdit:hover, QPlainTextEdit:hover, QTextEdit:hover, QSpinBox:hover, QAbstractSpinBox:hover { border: 1px solid #B9B9C1; }"
        "QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus, QAbstractSpinBox:focus { border: 1px solid #007AFF; }"
        "QComboBox { background: #FFFFFF; color: #1C1C1E; border: 1px solid #D1D1D6;"
        " border-radius: 8px; padding: 6px 30px 6px 10px; min-height: 24px; }"
        "QComboBox:hover { border: 1px solid #B9B9C1; }"
        "QComboBox:focus { border: 1px solid #007AFF; }"
        "QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 24px; border: none; }"
        "QComboBox::down-arrow { width: 10px; height: 10px; }"
        "QComboBox QAbstractItemView { border: 1px solid #D1D1D6; border-radius: 8px; padding: 4px; background: #FFFFFF; }"
        "QComboBox QAbstractItemView::item { min-height: 28px; border-radius: 6px; margin: 2px 4px; padding: 4px 8px; color: #1C1C1E; }"
        "QComboBox QAbstractItemView::item:hover, QComboBox QAbstractItemView::item:selected { background-color: #007AFF; color: #FFFFFF; }"
        "QPushButton { background: #FFFFFF; color: #1C1C1E; border: 1px solid #D1D1D6;"
        " border-radius: 8px; padding: 8px 14px; font-weight: 600; }"
        "QPushButton:hover { background: #F2F2F7; border: 1px solid #B9B9C1; }"
        "QPushButton:pressed { background: #ECECF1; }"
        "QPushButton:disabled { color: #8E8E93; background: #F7F7FA; border: 1px solid #E5E5EA; }"
        "QPushButton#primaryButton { background: #007AFF; color: #FFFFFF; border: 1px solid #007AFF;"
        " border-radius: 8px; padding: 8px 16px; font-weight: 700; }"
        "QPushButton#primaryButton:hover { background: #0A84FF; border: 1px solid #0A84FF; }"
        "QPushButton#primaryButton:pressed { background: #0062CC; border: 1px solid #0062CC; }"
        "QPushButton#primaryButton:disabled { background: #AFCFFF; border: 1px solid #AFCFFF; color: #F7FBFF; }"
        "QPushButton#secondaryButton { background: #FFFFFF; color: #3A3A3C; border: 1px solid #D1D1D6;"
        " border-radius: 8px; padding: 6px 12px; }"
        "QPushButton#secondaryButton:hover { background: #F2F2F7; }"
        "QToolButton#iconButton { background: #FFFFFF; border: 1px solid #D1D1D6; border-radius: 8px; padding: 4px 8px; }"
        "QToolButton#iconButton:hover { background: #F2F2F7; border: 1px solid #B9B9C1; }"
        "QCheckBox, QRadioButton { spacing: 8px; color: #1C1C1E; }"
        "QCheckBox#modeOption { padding: 4px 0; }"
        "QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #C7C7CC; border-radius: 4px; background: #FFFFFF; }"
        "QCheckBox::indicator:checked { border: 1px solid #007AFF; background: #007AFF; }"
        "QRadioButton::indicator { width: 16px; height: 16px; border: 1px solid #C7C7CC; border-radius: 8px; background: #FFFFFF; }"
        "QRadioButton::indicator:checked { border: 1px solid #007AFF; background: #007AFF; }"
        "QProgressBar { background: #F2F2F7; border: 1px solid #E5E5EA; border-radius: 6px; text-align: center; color: #3A3A3C; }"
        "QProgressBar::chunk { background: #007AFF; border-radius: 6px; }"
        "QLabel { color: #1C1C1E; }"
        "QLabel a { color: #007AFF; }"
        "QTabWidget::pane { border: 1px solid #E5E5EA; border-radius: 10px; padding: 6px; background: #FFFFFF; }"
        "QTabBar::tab { background: #FFFFFF; border: 1px solid #E5E5EA;"
        " border-radius: 8px; padding: 6px 12px; margin-right: 6px; min-height: 22px; font-size: 12px; font-weight: 600; color: #3A3A3C; }"
        "QTabBar::tab:selected { background: #007AFF; color: #FFFFFF; border: 1px solid #007AFF; }"
        "QTabBar::tab:hover { background: #F2F2F7; }"
        "QFrame#stepCard { background: #FFFFFF; border: 1px solid #E5E5EA; border-radius: 12px; }"
        "QFrame#stepCard[state=\"active\"] { background: #F9FAFF; border: 1px solid #007AFF; }"
        "QFrame#stepCard[state=\"locked\"] { background: #FAFAFC; border: 1px dashed #D1D1D6; }"
        "QFrame#stepCard QLabel#stepStatus { color: #8E8E93; }"
        "QFrame#stepCard QLabel#stepArrow { color: #007AFF; }"
        "QFrame#stepCard[state=\"complete\"] QLabel#stepStatus { color: #007AFF; }"
        "QFrame#stepConnector { background: #D1D1D6; border-radius: 1px; }"
        "QGroupBox#springSourceBox { margin-top: 16px; border-radius: 10px; }"
        "QGroupBox#springSourceBox::title { subcontrol-origin: margin; left: 12px; padding: 0 2px;"
        " color: #3A3A3C; background: transparent; font-size: 13px; font-weight: 700; }"
    )


def make_font(point_size, weight=QFont.Normal):
    font = QFont()
    font.setPointSize(point_size)
    font.setWeight(weight)
    return font


def _strip_html(text):
    plain = re.sub(r"<[^>]*>", "", text or "")
    plain = plain.replace("&mu;", "mu").replace("&alpha;", "alpha").replace("&lambda;", "lambda")
    return plain


def _label_width_for_text(widget, text, padding=24, minimum=70, maximum=280):
    metrics = QFontMetrics(widget.font())
    width = metrics.horizontalAdvance(_strip_html(text)) + padding
    return max(minimum, min(maximum, width))


def select_app_font():
    if sys.platform == "darwin":
        candidates = ["SF Pro Text", "SF Pro Display", "Helvetica Neue", "Helvetica"]
    else:
        candidates = ["Segoe UI", "Helvetica Neue", "Noto Sans", "Ubuntu", "Arial"]
    for name in candidates:
        if QFontDatabase.hasFamily(name):
            return QFont(name, 16)
    font = QFont()
    font.setPointSize(16)
    return font


def get_model_list():
    models = []
    for attr_name in dir(MaterialModels):
        attr = getattr(MaterialModels, attr_name)
        if hasattr(attr, "model_type") and hasattr(attr, "category"):
            if hasattr(attr, "param_names") and attr.param_names:
                models.append(MODEL_DISPLAY_NAMES.get(attr_name, attr_name))
    models.append("Hill")
    return sorted(models)

def resolve_model_name(display_name):
    return MODEL_DISPLAY_LOOKUP.get(display_name, display_name)


class LatexLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._latex_image_data = None
        self._latex_text = None
        self.setMinimumHeight(48)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet("background: transparent;")
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def set_latex(self, latex):
        self._latex_text = latex
        if not latex:
            self.clear()
            self._latex_image_data = None
            return
        render_scale = max(1.5, float(self.devicePixelRatioF()))
        with matplotlib.rc_context(
            {
                "mathtext.fontset": "stixsans",
                "font.family": "sans-serif",
                "font.sans-serif": ["SF Pro Text", "Helvetica Neue", "Arial", "DejaVu Sans"],
            }
        ):
            fig = Figure(figsize=(2.6, 0.55), dpi=160 * render_scale)
            fig.patch.set_alpha(0.0)
            fig.patch.set_facecolor("none")
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            ax.set_facecolor("none")
            ax.patch.set_alpha(0.0)
            text_color = QApplication.palette().color(QPalette.WindowText)
            color = (text_color.redF(), text_color.greenF(), text_color.blueF(), 1.0)
            ax.text(0.0, 0.5, f"${latex}$", fontsize=11, va="center", ha="left", color=color)
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            image = canvas.buffer_rgba()
        from PySide6.QtGui import QImage, QPixmap
        self._latex_image_data = bytes(image)
        width = int(round(width))
        height = int(round(height))
        qimage = QImage(
            self._latex_image_data,
            width,
            height,
            width * 4,
            QImage.Format_RGBA8888,
        )
        qimage.setDevicePixelRatio(render_scale)
        self.setPixmap(QPixmap.fromImage(qimage))
        self.resize(int(round(width / render_scale)), int(round(height / render_scale)))

    def refresh_theme(self):
        if self._latex_text:
            self.set_latex(self._latex_text)


class SmallLatexLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._latex_image_data = None
        self._latex_text = None
        self.setMinimumHeight(28)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setStyleSheet("background: transparent;")
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def set_latex(self, latex):
        self._latex_text = latex
        if not latex:
            self.clear()
            self._latex_image_data = None
            return
        render_scale = max(1.5, float(self.devicePixelRatioF()))
        with matplotlib.rc_context(
            {
                "mathtext.fontset": "stixsans",
                "font.family": "sans-serif",
                "font.sans-serif": ["SF Pro Text", "Helvetica Neue", "Arial", "DejaVu Sans"],
            }
        ):
            fig = Figure(figsize=(0.75, 0.2), dpi=180 * render_scale)
            fig.patch.set_alpha(0.0)
            fig.patch.set_facecolor("none")
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            ax.set_facecolor("none")
            ax.patch.set_alpha(0.0)
            text_color = QApplication.palette().color(QPalette.WindowText)
            color = (text_color.redF(), text_color.greenF(), text_color.blueF(), 1.0)
            ax.text(0.02, 0.5, f"${latex}$", fontsize=8, va="center", ha="left", color=color)
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            image = canvas.buffer_rgba()
        from PySide6.QtGui import QImage, QPixmap
        self._latex_image_data = bytes(image)
        width = int(round(width))
        height = int(round(height))
        qimage = QImage(
            self._latex_image_data,
            width,
            height,
            width * 4,
            QImage.Format_RGBA8888,
        )
        qimage.setDevicePixelRatio(render_scale)
        self.setPixmap(QPixmap.fromImage(qimage))
        self.resize(int(round(width / render_scale)), int(round(height / render_scale)))

    def refresh_theme(self):
        if self._latex_text:
            self.set_latex(self._latex_text)


class LatexParameterCard(QFrame):
    removed = Signal(str)

    def __init__(self, raw_name, latex_text, parent=None):
        super().__init__(parent)
        self.raw_name = raw_name
        self.setObjectName("latexParameterCard")
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setStyleSheet(
            "QFrame#latexParameterCard {"
            " background: #EAF3FF;"
            " border: 1px solid #CFE3FF;"
            " border-radius: 6px;"
            "}"
            "QToolButton#latexParameterCardDelete {"
            " background: transparent;"
            " border: none;"
            " color: #007AFF;"
            " font-weight: 700;"
            " padding: 0px 2px;"
            "}"
            "QToolButton#latexParameterCardDelete:hover { color: #005FCC; }"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 6, 2)
        layout.setSpacing(4)

        self.label = SmallLatexLabel()
        self.label.set_latex(latex_text)
        self.delete_btn = QToolButton()
        self.delete_btn.setObjectName("latexParameterCardDelete")
        self.delete_btn.setText("×")
        self.delete_btn.setCursor(Qt.PointingHandCursor)
        self.delete_btn.clicked.connect(self._emit_removed)

        layout.addWidget(self.label)
        layout.addWidget(self.delete_btn)
        self.adjustSize()

    def _emit_removed(self):
        self.removed.emit(self.raw_name)

    def refresh_theme(self):
        if self.label:
            self.label.refresh_theme()


class ClickableLatexBadge(QFrame):
    insert_requested = Signal(str)

    def __init__(self, latex_text, raw_text, parent=None):
        super().__init__(parent)
        self.raw_text = raw_text
        self.setObjectName("clickableLatexBadge")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setStyleSheet(
            "QFrame#clickableLatexBadge {"
            " background: #F2F2F7;"
            " border: 1px solid #D1D1D6;"
            " border-radius: 6px;"
            "}"
            "QFrame#clickableLatexBadge:hover {"
            " background: #E5E5EA;"
            " border: 1px solid #C7C7CC;"
            "}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(0)

        self.label = SmallLatexLabel()
        self.label.set_latex(latex_text)
        layout.addWidget(self.label)
        self.adjustSize()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.insert_requested.emit(self.raw_text)
        super().mousePressEvent(event)

    def refresh_theme(self):
        if self.label:
            self.label.refresh_theme()


class SmallHtmlLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.setTextFormat(Qt.RichText)
        self.setStyleSheet("background: transparent;")
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._html_text = ""

    def set_html(self, html):
        self._html_text = html
        self.setText(html)

    def refresh_theme(self):
        if self._html_text:
            self.setText(self._html_text)


class StepCard(QFrame):
    clicked = Signal(int)

    def __init__(self, index, title, parent=None):
        super().__init__(parent)
        self.index = index
        self._locked = False
        self.setObjectName("stepCard")
        self.setProperty("state", "locked")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setCursor(Qt.PointingHandCursor)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 8, 0)
        text_layout.setSpacing(2)
        self.title_label = QLabel(f"{index + 1}. {title}")
        self.title_label.setObjectName("stepTitle")
        self.title_label.setFont(make_font(15, QFont.DemiBold))
        self.title_label.setWordWrap(True)
        self.status_label = QLabel("")
        self.status_label.setObjectName("stepStatus")
        self.status_label.setFont(make_font(12))
        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.status_label)
        layout.addLayout(text_layout, 1)
        self.arrow_label = QLabel(">")
        self.arrow_label.setObjectName("stepArrow")
        self.arrow_label.setFont(make_font(16, QFont.DemiBold))
        self.arrow_label.setAlignment(Qt.AlignCenter)
        self.arrow_label.setStyleSheet("background: transparent;")
        self.arrow_label.setVisible(False)
        layout.addWidget(self.arrow_label, 0, Qt.AlignRight | Qt.AlignVCenter)

    def set_state(self, state):
        self.setProperty("state", state)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def set_locked(self, locked):
        self._locked = locked
        if locked:
            self.setCursor(Qt.ForbiddenCursor)
        else:
            self.setCursor(Qt.PointingHandCursor)

    def set_current(self, is_current):
        if self.arrow_label:
            self.arrow_label.setVisible(is_current)

    def mousePressEvent(self, event):
        if not self._locked and event.button() == Qt.LeftButton:
            self.clicked.emit(self.index)
        super().mousePressEvent(event)


@dataclass
class SpringConfig:
    model_name: str
    strain_name: Optional[str] = None
    ogden_terms: int = 1
    param_values: Optional[List[float]] = None


class OptimizerWorker(QThread):
    finished = Signal(object, object)
    failed = Signal(str)
    progress = Signal(int, float, object)

    def __init__(self, optimizer, initial_guess, bounds, method):
        super().__init__()
        self.optimizer = optimizer
        self.initial_guess = initial_guess
        self.bounds = bounds
        self.method = method
        self._stop_requested = False

    def run(self):
        try:
            result = self.optimizer.fit(
                self.initial_guess,
                self.bounds,
                method=self.method,
                progress_cb=self._emit_progress,
            )
            self.finished.emit(result, self.optimizer)
        except Exception as exc:
            if self._stop_requested:
                self.failed.emit("Optimization aborted by user.")
                return
            self.failed.emit(str(exc))

    def stop(self):
        self._stop_requested = True
        self.requestInterruption()
        if hasattr(self.optimizer, "request_stop"):
            self.optimizer.request_stop()

    def _emit_progress(self, iteration, params, loss):
        if self._stop_requested or self.isInterruptionRequested():
            if hasattr(self.optimizer, "request_stop"):
                self.optimizer.request_stop()
            return
        try:
            params_list = list(params)
        except Exception:
            params_list = params
        self.progress.emit(int(iteration), float(loss), params_list)


class MatplotlibCanvas(FigureCanvas):
    def __init__(self, width=5, height=3):
        fig = Figure(figsize=(width, height), dpi=120)
        self.ax = fig.add_subplot(111)
        self.axes = [self.ax]
        super().__init__(fig)
        self.setStyleSheet("background: transparent;")
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.apply_theme()

    def _qcolor_to_mpl(self, color):
        return (color.redF(), color.greenF(), color.blueF(), 1.0)

    def set_axes(self, axes):
        self.axes = axes
        if axes:
            self.ax = axes[0]

    def reset_axes(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.set_axes([ax])
        self.apply_theme()
        return ax

    def teardown(self):
        try:
            self.figure.clear()
            self.axes = []
            self.ax = None
            self.draw_idle()
        except Exception:
            pass
        self.deleteLater()
        gc.collect()

    def apply_theme(self):
        self.figure.patch.set_facecolor("none")
        for ax in self.axes:
            apply_theme_to_axis(ax)
            ax.tick_params(labelsize=10)


class CustomDataEntry(QWidget):
    def __init__(self, index, on_change=None, on_remove=None):
        super().__init__()
        self.index = index
        self.on_change = on_change
        self.on_remove = on_remove
        self._normal_editor_style = "QPlainTextEdit { border: 1px solid #D1D1D6; background-color: #FFFFFF; }"
        self._invalid_editor_style = "QPlainTextEdit { border: 2px solid #FF3B30; background-color: #FFF0F0; }"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        header = QHBoxLayout()
        self.title_label = QLabel(f"Dataset {index}")
        self.title_label.setFont(make_font(15, QFont.DemiBold))
        self.import_btn = QPushButton("Import CSV/TXT...")
        self.import_btn.setObjectName("secondaryButton")
        self.import_btn.clicked.connect(self._import_from_file)
        self.remove_btn = QPushButton("Delete")
        self.remove_btn.setObjectName("secondaryButton")
        self.remove_btn.clicked.connect(self._remove)
        header.addWidget(self.title_label)
        header.addStretch()
        header.addWidget(self.import_btn)
        header.addWidget(self.remove_btn)
        layout.addLayout(header)
        self.set_deletable(index != 1)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.mode_combo = QComboBox()
        self.mode_combo.setView(QListView())
        self.mode_combo.addItem("Uniaxial Tension", "UT")
        self.mode_combo.addItem("Uniaxial Compression", "UC")
        self.mode_combo.addItem("Equibiaxial Tension", "ET")
        self.mode_combo.addItem("Pure Shear", "PS")
        self.mode_combo.addItem("Simple Shear", "SS")
        self.mode_combo.addItem("Biaxial Tension", "BT")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.stress_combo = QComboBox()
        self.stress_combo.setView(QListView())
        self.stress_combo.addItem("Nominal (1st PK)", "PK1")
        self.stress_combo.addItem("Cauchy", "cauchy")
        self.stress_combo.currentIndexChanged.connect(self._on_mode_changed)
        for combo in (self.mode_combo, self.stress_combo):
            combo.setMinimumHeight(30)
            combo.setMinimumWidth(180)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        form.addRow("Loading mode", self.mode_combo)
        form.addRow("Stress type", self.stress_combo)
        layout.addLayout(form)

        self.data_grid = QGridLayout()
        self.data_grid.setHorizontalSpacing(4)
        self.data_grid.setVerticalSpacing(10)
        self.data_grid.setAlignment(Qt.AlignLeft)
        layout.addLayout(self.data_grid)

        self._build_data_fields()
        self._validate_data()

    def set_index(self, index):
        self.index = index
        if self.title_label:
            self.title_label.setText(f"Dataset {index}")
        self.set_deletable(index != 1)

    def set_deletable(self, can_delete):
        if self.remove_btn:
            self.remove_btn.setVisible(can_delete)

    def _remove(self):
        if self.on_remove:
            self.on_remove(self)

    def _emit_change(self, *_args):
        if self.on_change:
            self.on_change()

    def _build_data_fields(self):
        while self.data_grid.count():
            item = self.data_grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.data_inputs = []
        mode = self.mode_combo.currentData()
        stress_type = self.stress_combo.currentData()
        if mode == "BT":
            if stress_type == "cauchy":
                labels = [r"$\lambda_1$", r"$\lambda_2$", r"$\sigma_{11}$", r"$\sigma_{22}$"]
            else:
                labels = [r"$\lambda_1$", r"$\lambda_2$", r"$P_{11}$", r"$P_{22}$"]
            placeholders = [
                "1.0\n1.1\n1.2\n1.3",
                "1.0\n1.0\n1.0\n1.0",
                "0.0\n0.2\n0.4\n0.6",
                "0.0\n0.1\n0.2\n0.3",
            ]
        elif mode == "SS":
            stress_label = r"$P_{12}$" if stress_type != "cauchy" else r"$\sigma_{12}$"
            labels = [r"$\gamma$", stress_label]
            placeholders = ["0.0\n0.1\n0.2\n0.3", "0.0\n0.2\n0.4\n0.6"]
        else:
            stress_label = r"$P_{11}$" if stress_type != "cauchy" else r"$\sigma_{11}$"
            labels = [r"$\lambda_1$", stress_label]
            placeholders = ["1.0\n1.1\n1.2\n1.3", "0.0\n0.2\n0.4\n0.6"]

        for col, text in enumerate(labels):
            label = SmallLatexLabel()
            label.set_latex(text.strip("$"))
            edit = QPlainTextEdit()
            edit.setPlaceholderText(placeholders[col] if col < len(placeholders) else "")
            edit.setMinimumHeight(80)
            edit.setMinimumWidth(150)
            edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            edit.setStyleSheet(self._normal_editor_style)
            edit.textChanged.connect(self._on_data_text_changed)
            self.data_grid.addWidget(label, 0, col)
            self.data_grid.addWidget(edit, 1, col)
            self.data_inputs.append(edit)
            self.data_grid.setColumnStretch(col, 1)
            self.data_grid.setRowMinimumHeight(1, 90)
        self.data_grid.setRowMinimumHeight(0, 36)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.data_grid.addWidget(spacer, 0, len(labels), 2, 1)

    def _on_mode_changed(self):
        self._build_data_fields()
        self._validate_data()
        self._emit_change()

    def _on_data_text_changed(self):
        self._validate_data()
        self._emit_change()

    @staticmethod
    def _non_empty_lines(text):
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _validate_data(self):
        if not hasattr(self, "data_inputs"):
            return True

        columns = [self._non_empty_lines(edit.toPlainText()) for edit in self.data_inputs]
        invalid_columns = set()

        lengths = [len(col) for col in columns]
        non_zero_lengths = [length for length in lengths if length > 0]
        if non_zero_lengths:
            expected_length = non_zero_lengths[0]
            for index, length in enumerate(lengths):
                if length not in (0, expected_length):
                    invalid_columns.add(index)

        for index, column in enumerate(columns):
            for value in column:
                try:
                    float(value)
                except ValueError:
                    invalid_columns.add(index)
                    break

        for index, edit in enumerate(self.data_inputs):
            edit.setStyleSheet(self._invalid_editor_style if index in invalid_columns else self._normal_editor_style)
        return not invalid_columns

    def _import_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Data File",
            "",
            "Data Files (*.csv *.txt *.dat);;All Files (*)",
        )
        if not file_path:
            return

        content = None
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                with open(file_path, "r", encoding=encoding) as handle:
                    content = handle.read()
                break
            except (OSError, UnicodeDecodeError):
                continue

        if content is None:
            QMessageBox.warning(self, "Import Failed", "Could not read the selected file.")
            return

        expected_columns = len(self.data_inputs)
        parsed_rows = []
        for line_number, raw_line in enumerate(content.splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            tokens = [token for token in re.split(r"[,\t ]+", stripped) if token]
            if not tokens:
                continue
            if len(tokens) < expected_columns:
                QMessageBox.warning(
                    self,
                    "Import Failed",
                    f"Line {line_number} has {len(tokens)} columns, expected at least {expected_columns}.",
                )
                return
            parsed_rows.append(tokens[:expected_columns])

        if not parsed_rows:
            QMessageBox.warning(self, "Import Failed", "No valid numeric rows were found in the selected file.")
            return

        for line_number, row in enumerate(parsed_rows, start=1):
            for value in row:
                try:
                    float(value)
                except ValueError:
                    QMessageBox.warning(
                        self,
                        "Import Failed",
                        f"Line {line_number} contains a non-numeric value: {value}",
                    )
                    return

        columns = [[] for _ in range(expected_columns)]
        for row in parsed_rows:
            for index, value in enumerate(row):
                columns[index].append(value)

        for index, edit in enumerate(self.data_inputs):
            edit.blockSignals(True)
            edit.setPlainText("\n".join(columns[index]))
            edit.blockSignals(False)

        self._validate_data()
        self._emit_change()

    def _parse_column(self, text):
        values = []
        for part in text.replace(",", " ").split():
            try:
                values.append(float(part))
            except ValueError:
                continue
        return values

    def get_data(self, validate_lengths=True):
        mode = self.mode_combo.currentData()
        stress_type = self.stress_combo.currentData()
        columns = [self._parse_column(edit.toPlainText()) for edit in self.data_inputs]
        if not columns or not columns[0]:
            return None
        lengths = [len(col) for col in columns]
        if len(set(lengths)) != 1:
            if validate_lengths:
                raise ValueError("Each column must have the same number of values.")
            return None
        mode_label = format_mode_label(mode)
        data = {
            "author": "Custom",
            "mode": mode,
            "mode_raw": mode,
            "label": f"Dataset {self.index} - {mode_label}",
            "stress_type": stress_type,
        }
        if mode == "BT":
            data["stretch"] = np.array(columns[0], dtype=float)
            data["stretch_secondary"] = np.array(columns[1], dtype=float)
            stress_primary = np.array(columns[2], dtype=float)
            stress_secondary = np.array(columns[3], dtype=float) if len(columns) > 3 else None
            if stress_secondary is not None:
                data["stress_exp"] = np.column_stack([stress_primary, stress_secondary])
            else:
                data["stress_exp"] = stress_primary
        else:
            data["stretch"] = np.array(columns[0], dtype=float)
            data["stress_exp"] = np.array(columns[1], dtype=float)
        return data


class SavedDatasetRow(QWidget):
    def __init__(self, text, on_remove=None, parent=None):
        super().__init__(parent)
        self.on_remove = on_remove
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(6)
        self.label = QLabel(text)
        self.label.setFont(make_font(12))
        self.label.setWordWrap(True)
        remove_btn = QToolButton()
        remove_btn.setText("x")
        remove_btn.setObjectName("iconButton")
        remove_btn.setToolTip("Remove")
        remove_btn.clicked.connect(self._remove)
        layout.addWidget(self.label, 1)
        layout.addWidget(remove_btn, 0, Qt.AlignRight)

    def _remove(self):
        if self.on_remove:
            self.on_remove(self)


class SpringIcon(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        palette = QApplication.palette()
        line_color = palette.color(QPalette.WindowText)
        pen = QPen(line_color, 1.6)
        painter.setPen(pen)

        rect = self.rect().adjusted(8, 8, -8, -8)
        left = rect.left()
        right = rect.right()
        mid = rect.center().y()
        if right <= left:
            return

        segments = 8
        span = right - left
        step = span / segments
        amplitude = max(4, min(8, rect.height() / 2 - 2))

        points = []
        points.append((left, mid))
        for i in range(1, segments):
            offset = amplitude if i % 2 else -amplitude
            points.append((left + step * i, mid + offset))
        points.append((right, mid))
        for i in range(len(points) - 1):
            painter.drawLine(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])

class SpringWidget(QGroupBox):
    def __init__(self, index, parent=None, on_change=None, author_provider=None, on_remove=None):
        super().__init__("")
        self.index = index
        self.on_change = on_change
        self.author_provider = author_provider
        self.on_remove = on_remove
        self.model_combo = QComboBox()
        self.model_combo.setView(QListView())
        self.model_combo.addItems(["Select..."] + get_model_list())
        self.model_label = QLabel("Model")
        self.model_combo.setMinimumWidth(180)
        self.model_combo.setMinimumHeight(30)
        self.model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.strain_label = QLabel("Strain")
        self.strain_combo = QComboBox()
        self.strain_combo.setView(QListView())
        self.strain_combo.addItems(list(STRAIN_CONFIGS.keys()))
        self.strain_combo.setEnabled(False)
        self.strain_combo.setVisible(False)
        self.strain_label.setVisible(False)
        self.strain_combo.setMinimumWidth(160)
        self.strain_combo.setMinimumHeight(30)
        self.strain_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.ogden_label = QLabel("Ogden terms")
        self.ogden_terms = QSpinBox()
        self.ogden_terms.setRange(1, 6)
        self.ogden_terms.setValue(1)
        self.ogden_terms.setObjectName("ogdenTerms")
        self.ogden_terms.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.ogden_terms.setAlignment(Qt.AlignRight)
        self.ogden_terms.lineEdit().setAlignment(Qt.AlignRight)
        self.ogden_terms.setMinimumWidth(72)
        self.ogden_terms.setMinimumHeight(30)
        self.ogden_terms.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.ogden_dec_btn = QToolButton()
        self.ogden_dec_btn.setObjectName("iconButton")
        self.ogden_dec_btn.setText("-")
        self.ogden_dec_btn.clicked.connect(self.ogden_terms.stepDown)
        self.ogden_inc_btn = QToolButton()
        self.ogden_inc_btn.setObjectName("iconButton")
        self.ogden_inc_btn.setText("+")
        self.ogden_inc_btn.clicked.connect(self.ogden_terms.stepUp)
        self.ogden_terms_widget = QWidget()
        ogden_layout = QHBoxLayout(self.ogden_terms_widget)
        ogden_layout.setContentsMargins(0, 0, 0, 0)
        ogden_layout.setSpacing(4)
        ogden_layout.addWidget(self.ogden_terms)
        ogden_layout.addWidget(self.ogden_dec_btn)
        ogden_layout.addWidget(self.ogden_inc_btn)
        self.ogden_terms_widget.setEnabled(False)
        self.ogden_terms_widget.setVisible(False)
        self.ogden_row = QWidget()
        ogden_row_layout = QHBoxLayout(self.ogden_row)
        ogden_row_layout.setContentsMargins(0, 0, 0, 0)
        ogden_row_layout.setSpacing(6)
        ogden_row_layout.addWidget(self.ogden_label)
        ogden_row_layout.addWidget(self.ogden_terms_widget)
        ogden_row_layout.addStretch()
        self.ogden_row.setVisible(False)
        self.ogden_label.setVisible(False)

        self.formula_label = LatexLabel()
        self.strain_formula_title = QLabel("Generalized Strain")
        self.strain_formula_title.setFont(make_font(10, QFont.DemiBold))
        self.strain_formula_title.setVisible(False)
        self.strain_formula_label = LatexLabel()
        self.reference_title = QLabel("Reference")
        self.reference_title.setFont(make_font(10, QFont.DemiBold))
        self.reference_title.setVisible(False)
        self.reference_label = QLabel()
        self.reference_label.setTextFormat(Qt.RichText)
        self.reference_label.setOpenExternalLinks(True)
        self.reference_label.setWordWrap(True)
        self.params_layout = QGridLayout()
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        self.params_layout.setHorizontalSpacing(8)
        self.params_layout.setVerticalSpacing(2)
        self.params_layout.setColumnStretch(1, 1)
        self.params_layout.setColumnStretch(3, 1)
        self.params_layout.setColumnStretch(5, 1)

        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(8, 8, 8, 8)

        header_row = QHBoxLayout()
        header = QLabel(f"Spring {index}")
        header.setFont(make_font(13, QFont.DemiBold))
        header_row.addWidget(header)
        header_row.addStretch()
        self.remove_btn = QToolButton()
        self.remove_btn.setObjectName("iconButton")
        self.remove_btn.setText("×")
        self.remove_btn.setToolTip("Delete spring")
        self.remove_btn.clicked.connect(self._on_remove_clicked)
        header_row.addWidget(self.remove_btn)
        layout.addLayout(header_row)

        controls = QGridLayout()
        controls.setHorizontalSpacing(4)
        controls.setVerticalSpacing(4)
        controls.addWidget(self.model_label, 0, 0)
        controls.addWidget(self.model_combo, 0, 1)
        controls.addWidget(self.ogden_row, 0, 2, 1, 2)
        controls.addWidget(self.strain_label, 1, 0)
        controls.addWidget(self.strain_combo, 1, 1)
        controls.setColumnStretch(1, 1)
        controls.setColumnStretch(2, 1)
        controls.setColumnStretch(3, 1)
        layout.addLayout(controls)
        self.controls_layout = controls

        content = QGridLayout()
        content.setHorizontalSpacing(10)
        content.setVerticalSpacing(2)
        content.setContentsMargins(0, 0, 0, 0)

        self.params_block_widget = QWidget()
        params_block = QVBoxLayout(self.params_block_widget)
        params_block.setSpacing(0)
        params_block.setContentsMargins(0, 0, 0, 0)
        params_label = QLabel("Parameters (Built-in)")
        params_label.setFont(make_font(12, QFont.DemiBold))
        params_label.setContentsMargins(0, 0, 0, 0)
        params_label.setStyleSheet("margin-bottom: 0px;")
        params_block.addWidget(params_label)
        self.params_widget = QWidget()
        self.params_widget.setLayout(self.params_layout)
        self.params_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        params_block.addWidget(self.params_widget)

        self.formula_block_widget = QWidget()
        formula_block = QVBoxLayout(self.formula_block_widget)
        formula_block.setSpacing(2)
        formula_block.setContentsMargins(0, 0, 0, 0)
        formula_label = QLabel("Formula")
        formula_label.setFont(make_font(12, QFont.DemiBold))
        formula_label.setContentsMargins(0, 0, 0, 0)
        formula_block.addWidget(formula_label)
        self.strain_formula_container = QWidget()
        strain_formula_layout = QVBoxLayout(self.strain_formula_container)
        strain_formula_layout.setContentsMargins(0, 0, 0, 0)
        strain_formula_layout.setSpacing(2)
        strain_formula_layout.addWidget(self.strain_formula_title)
        strain_formula_layout.addWidget(self.strain_formula_label)

        self.formula_row = QHBoxLayout()
        self.formula_row.setContentsMargins(0, 0, 0, 0)
        self.formula_row.setSpacing(8)
        self.formula_row.addWidget(self.formula_label, 2)
        self.formula_row.addWidget(self.strain_formula_container, 2)
        formula_block.addLayout(self.formula_row)
        formula_block.addWidget(self.reference_title)
        formula_block.addWidget(self.reference_label)

        content.addWidget(self.params_block_widget, 0, 0)
        content.addWidget(self.formula_block_widget, 0, 1)
        content.setColumnStretch(0, 2)
        content.setColumnStretch(1, 3)
        layout.addLayout(content)
        self.params_block_widget.setVisible(False)
        self.formula_block_widget.setVisible(False)
        self.strain_formula_container.setVisible(False)

        self.setLayout(layout)
        self.setMinimumWidth(520)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setFlat(True)
        self.apply_theme()
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.strain_combo.currentTextChanged.connect(self._on_model_changed)
        self.ogden_terms.valueChanged.connect(self._on_model_changed)
        self.param_edits = []
        self._param_prefix = f"{self.model_combo.currentText()}_{self.index}_"
        self._model_name = "Select..."
        self._custom_error = ""
        self._on_model_changed()

    def apply_theme(self):
        self.setStyleSheet(
            "QGroupBox { border: 1px solid palette(mid); border-radius: 12px; background: palette(base); }"
        )

    def _clear_params(self):
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.param_edits = []

    def _format_param_label(self, name):
        if self._param_prefix and name.startswith(self._param_prefix):
            name = name[len(self._param_prefix):]

        def italic(text):
            return f"<i>{text}</i>"

        def format_base(base):
            if base.lower() == "mu":
                return "&mu;"
            if base.lower() == "alpha":
                return "&alpha;"
            return base

        match = re.match(r"^([A-Za-z]+)(\d+)$", name)
        if match:
            base, idx = match.groups()
            if self._model_name == "Hill" and base.lower() in ("m", "n"):
                return f"{italic(base)}<sub>{idx}</sub>"
            base_html = format_base(base)
            return f"{italic(base_html)}<sub>{idx}</sub>"
        match = re.match(r"^([A-Za-z]+)_([0-9]+)$", name)
        if match:
            base, idx = match.groups()
            base_html = format_base(base)
            return f"{italic(base_html)}<sub>{idx}</sub>"
        if name.lower() == "mu":
            return italic("&mu;")
        if name.lower() == "alpha":
            return italic("&alpha;")
        return italic(name)

    def _on_param_changed(self):
        if self.on_change:
            self.on_change(changed=True)

    def _on_remove_clicked(self):
        if self.on_remove:
            self.on_remove(self.index)

    def set_removable(self, enabled):
        self.remove_btn.setEnabled(enabled)
        self.remove_btn.setVisible(enabled)


    def _on_model_changed(self):
        display_name = self.model_combo.currentText()
        model_name = resolve_model_name(display_name)
        self._model_name = display_name
        self.model_label.setVisible(True)
        self.model_combo.setVisible(True)
        self.model_combo.setEnabled(True)
        is_hill = display_name == "Hill"
        is_ogden = display_name == "Ogden"
        show_hill = is_hill
        show_ogden = is_ogden
        self.strain_combo.setEnabled(show_hill)
        self.strain_combo.setVisible(show_hill)
        self.strain_label.setVisible(show_hill)
        self.ogden_terms_widget.setEnabled(show_ogden)
        self.ogden_terms_widget.setVisible(show_ogden)
        self.ogden_label.setVisible(show_ogden)
        self.ogden_row.setVisible(show_ogden)
        details_visible = display_name != "Select..."
        self.params_block_widget.setVisible(details_visible)
        self.formula_block_widget.setVisible(details_visible)
        self._clear_params()

        if not details_visible:
            self.formula_label.clear()
            self.strain_formula_label.clear()
            self.strain_formula_title.setVisible(False)
            if hasattr(self, "strain_formula_container"):
                self.strain_formula_container.setVisible(False)
            self.reference_title.setVisible(False)
            self.reference_label.clear()
            if self.on_change:
                self.on_change(changed=False)
            return

        if display_name == "Hill":
            strain_name = self.strain_combo.currentText()
            func = MaterialModels.create_hill_model(strain_name)
        elif display_name == "Ogden":
            func = MaterialModels.create_ogden_model(self.ogden_terms.value())
        else:
            func = getattr(MaterialModels, model_name)

        self._param_prefix = f"{model_name}_{self.index}_"
        formula = getattr(func, "formula", "")
        self.formula_label.set_latex(formula)
        strain_formula = getattr(func, "strain_formula", "")
        show_strain_formula = bool(strain_formula) and show_hill
        if show_strain_formula:
            self.strain_formula_title.setVisible(True)
            self.strain_formula_label.set_latex(strain_formula)
            if hasattr(self, "strain_formula_container"):
                self.strain_formula_container.setVisible(True)
        else:
            self.strain_formula_title.setVisible(False)
            self.strain_formula_label.clear()
            if hasattr(self, "strain_formula_container"):
                self.strain_formula_container.setVisible(False)

        reference = MODEL_REFERENCES.get(model_name)
        if reference:
            label, url = reference
            self.reference_title.setVisible(True)
            self.reference_label.setText(f"<a href='{url}'>{label}</a>")
        else:
            self.reference_title.setVisible(False)
            self.reference_label.clear()

        self._param_prefix = f"{model_name}_{self.index}_"
        temp_net = ParallelNetwork()
        temp_net.add_model(func, f"{model_name}_{self.index}")
        author_name = self.author_provider() if self.author_provider else None
        n_guess = DATASET_N_GUESS.get(author_name, None)
        for idx, (name, default) in enumerate(zip(temp_net.param_names, temp_net.initial_guess)):
            row = idx
            col = 0
            if (
                n_guess is not None
                and model_name != "ZhanNonGaussian"
                and (name.lower().endswith("_n") or name.lower().endswith("n"))
            ):
                default = n_guess
            label = QLabel()
            label.setTextFormat(Qt.RichText)
            label_html = self._format_param_label(name)
            label.setText(label_html)
            label.setMinimumWidth(_label_width_for_text(label, label_html, minimum=64, maximum=190))
            label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            edit = QLineEdit()
            text_value = f"{float(default):.4g}"
            edit.setText(text_value)
            edit.setPlaceholderText(text_value)
            edit.setMinimumWidth(120)
            edit.setMinimumHeight(28)
            edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            edit.textChanged.connect(self._on_param_changed)
            self.params_layout.addWidget(label, row, col)
            self.params_layout.addWidget(edit, row, col + 1)
            self.param_edits.append((name, edit, default))
        self._custom_error = ""
        if self.on_change:
            self.on_change(changed=True)

    def is_valid(self):
        return self.model_combo.currentText() != "Select..."

    def build_config(self):
        display_name = self.model_combo.currentText()
        model_name = resolve_model_name(display_name)
        if display_name == "Hill":
            strain_name = self.strain_combo.currentText()
            func = MaterialModels.create_hill_model(strain_name)
        elif display_name == "Ogden":
            func = MaterialModels.create_ogden_model(self.ogden_terms.value())
        elif display_name != "Select...":
            func = getattr(MaterialModels, model_name)
        else:
            raise ValueError("Select a model.")

        user_params = []
        for _, edit, default in self.param_edits:
            text = edit.text().strip()
            if not text:
                user_params.append(float(default))
            else:
                try:
                    user_params.append(float(text))
                except ValueError:
                    user_params.append(float(default))

        return func, user_params

    def get_model_config(self):
        return self.build_config()

    def get_state(self):
        return {
            "model": self.model_combo.currentText(),
            "strain": self.strain_combo.currentText(),
            "ogden_terms": self.ogden_terms.value(),
            "params": [edit.text().strip() for _, edit, _ in self.param_edits],
            "model_source": "builtin",
            "use_custom": False,
        }

    def apply_state(self, state):
        if not state:
            return
        model = state.get("model", "Select...")
        if model in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
            self.model_combo.setCurrentText(model)
        if model == "Hill":
            strain = state.get("strain")
            if strain in [self.strain_combo.itemText(i) for i in range(self.strain_combo.count())]:
                self.strain_combo.setCurrentText(strain)
        if model == "Ogden":
            self.ogden_terms.setValue(state.get("ogden_terms", 1))
        self._on_model_changed()
        # Apply params after widgets are built
        values = state.get("params", [])
        for i, (_, edit, _) in enumerate(self.param_edits):
            if i < len(values) and values[i]:
                edit.setText(values[i])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calibration for Hyperelasticity")
        self.setMinimumSize(1000, 700)
        self.resize(1300, 850)
        icon_path = os.path.join(base_dir, "assets", "icons", "app.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        os.environ["CALIBRATION_DATA_DIR"] = os.path.join(base_dir, "data")
        self.step_names = ["Experimental Data", "Model Architecture", "Optimization", "Prediction"]
        self.current_step = 0
        self.section_widgets = {}
        self.latest_optimizer = None
        self.latest_result = None
        self.latest_network = None
        self.worker = None
        self._optimization_running = False
        self._bt_mix_warned = False
        self._custom_source_warned = False
        self.prediction_selection = {}

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(16)

        sidebar = self._build_sidebar()
        root_layout.addWidget(sidebar)

        self.content = self._build_content()
        root_layout.addWidget(self.content, 1)

        self.setCentralWidget(root)

        self.datasets = get_available_datasets()
        self._populate_authors()
        self._update_data_source()
        self._apply_theme()
        self._update_workflow_cards()

    def _build_sidebar(self):
        sidebar = QGroupBox("Navigation")
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        about_box = QGroupBox("About")
        about_layout = QVBoxLayout()
        about_layout.setContentsMargins(12, 12, 12, 12)
        about_layout.setSpacing(10)
        about_title = QLabel("Hyperelastic Calibration")
        about_title.setFont(make_font(14, QFont.DemiBold))
        about_text = QLabel("Fit models to experimental data.")
        about_text.setWordWrap(True)
        about_text.setFont(make_font(13))
        about_text.setStyleSheet("color: palette(windowtext);")
        about_layout.addWidget(about_title)
        about_layout.addWidget(about_text)
        about_box.setLayout(about_layout)
        layout.addWidget(about_box)

        workflow_box = QGroupBox("Workflow")
        workflow_layout = QVBoxLayout()
        workflow_layout.setContentsMargins(12, 12, 12, 12)
        workflow_layout.setSpacing(12)
        self.step_cards = []
        self.step_connectors = []
        for idx, name in enumerate(self.step_names):
            card = StepCard(idx, name)
            card.clicked.connect(self._on_step_card_clicked)
            workflow_layout.addWidget(card)
            self.step_cards.append(card)
            if idx < len(self.step_names) - 1:
                connector = QFrame()
                connector.setObjectName("stepConnector")
                connector.setMinimumSize(2, 12)
                workflow_layout.addWidget(connector, alignment=Qt.AlignHCenter)
                self.step_connectors.append(connector)
        workflow_box.setLayout(workflow_layout)
        layout.addWidget(workflow_box)

        author_box = QGroupBox("Author")
        author_layout = QVBoxLayout()
        author_layout.setContentsMargins(12, 12, 12, 12)
        author_layout.setSpacing(10)
        name = QLabel("Chongran Zhao")
        name.setFont(make_font(15, QFont.DemiBold))
        author_layout.addWidget(name)
        link_row = QHBoxLayout()
        link_row.setSpacing(10)
        email_icon = QApplication.style().standardIcon(QStyle.SP_MessageBoxInformation)
        site_icon = QApplication.style().standardIcon(QStyle.SP_ComputerIcon)
        github_icon = QApplication.style().standardIcon(QStyle.SP_ComputerIcon)
        orcid_icon = QApplication.style().standardIcon(QStyle.SP_DialogYesButton)
        rg_icon = QApplication.style().standardIcon(QStyle.SP_DirLinkIcon)
        icon_dir = os.path.join(base_dir, "assets", "icons")

        def _load_icon(filename, theme_name, fallback):
            icon_path = os.path.join(icon_dir, filename)
            if os.path.exists(icon_path):
                icon = QIcon(icon_path)
            else:
                icon = QIcon.fromTheme(theme_name, fallback)
            if icon.isNull():
                icon = fallback
            return icon
        self.email_btn = QToolButton()
        self.email_btn.setObjectName("iconButton")
        self.email_btn.setToolTip("Email")
        self.email_btn.setIcon(QIcon.fromTheme("mail-message-new", email_icon))
        self.email_btn.setIconSize(QSize(18, 18))
        self.email_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("mailto:chongranzhao@outlook.com")))
        self.site_btn = QToolButton()
        self.site_btn.setObjectName("iconButton")
        self.site_btn.setToolTip("Website")
        self.site_btn.setIcon(QIcon.fromTheme("internet-services", site_icon))
        self.site_btn.setIconSize(QSize(18, 18))
        self.site_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://chongran-zhao.github.io")))
        self.orcid_btn = QToolButton()
        self.orcid_btn.setObjectName("iconButton")
        self.orcid_btn.setToolTip("ORCID")
        self.orcid_btn.setIcon(_load_icon("orcid.svg", "orcid", orcid_icon))
        self.orcid_btn.setIconSize(QSize(18, 18))
        self.orcid_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://orcid.org/0009-0005-4164-8051"))
        )
        self.github_btn = QToolButton()
        self.github_btn.setObjectName("iconButton")
        self.github_btn.setToolTip("GitHub")
        self.github_btn.setIcon(_load_icon("github.svg", "github", github_icon))
        self.github_btn.setIconSize(QSize(18, 18))
        self.github_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://github.com/Chongran-Zhao"))
        )
        self.rg_btn = QToolButton()
        self.rg_btn.setObjectName("iconButton")
        self.rg_btn.setToolTip("ResearchGate")
        self.rg_btn.setIcon(_load_icon("researchgate.svg", "researchgate", rg_icon))
        self.rg_btn.setIconSize(QSize(18, 18))
        self.rg_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://www.researchgate.net/profile/Chongran-Zhao"))
        )
        buttons = [self.email_btn, self.site_btn, self.orcid_btn, self.github_btn, self.rg_btn]
        for idx, btn in enumerate(buttons):
            link_row.addWidget(btn)
            if idx < len(buttons) - 1:
                link_row.addStretch()
        author_layout.addLayout(link_row)
        author_box.setLayout(author_layout)
        layout.addWidget(author_box)

        layout.addStretch()
        sidebar.setLayout(layout)
        sidebar.setMinimumWidth(260)
        return sidebar

    def _build_content(self):
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        header = QLabel("Calibration for Hyperelasticity")
        header.setFont(make_font(26, QFont.DemiBold))
        header.setStyleSheet("color: palette(windowtext);")
        layout.addWidget(header)

        data_section = self._build_data_section()
        model_section = self._build_model_section()
        opt_section = self._build_optimization_section()
        prediction_section = self._build_prediction_section()
        layout.addWidget(data_section)
        layout.addWidget(model_section)
        layout.addWidget(opt_section)
        layout.addWidget(prediction_section)
        layout.addStretch()

        self.section_widgets = {
            self.step_names[0]: data_section,
            self.step_names[1]: model_section,
            self.step_names[2]: opt_section,
            self.step_names[3]: prediction_section,
        }
        self._update_section_visibility()
        self.scroll.setWidget(container)
        return self.scroll

    def _apply_theme(self):
        app = QApplication.instance()
        if app:
            app.setPalette(build_app_palette())
            app.setStyleSheet(build_app_stylesheet())
        self._refresh_latex_labels()
        self._refresh_spring_widgets()
        self._refresh_canvases()
        self._update_workflow_cards()

    def _refresh_canvases(self):
        for canvas in (
            getattr(self, "preview_canvas", None),
            getattr(self, "calib_canvas", None),
            getattr(self, "prediction_canvas", None),
        ):
            if canvas:
                canvas.apply_theme()
                canvas.draw()

    def _refresh_latex_labels(self):
        for label in self.findChildren(LatexLabel):
            label.refresh_theme()
        for label in self.findChildren(SmallLatexLabel):
            label.refresh_theme()
        for label in self.findChildren(SmallHtmlLabel):
            label.refresh_theme()
        for label in self.findChildren(QLabel):
            if label.textFormat() == Qt.RichText:
                label.setText(label.text())

    def _refresh_spring_widgets(self):
        for widget in getattr(self, "spring_widgets", []):
            if widget:
                widget.apply_theme()
        for icon in getattr(self, "spring_icons", []):
            if icon:
                icon.update()

    def _build_data_section(self):
        self.data_box = QGroupBox("1. Experimental Data")
        layout = QHBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        source_box = QGroupBox("Data source")
        source_layout = QVBoxLayout()
        source_layout.setContentsMargins(12, 12, 12, 12)
        source_layout.setSpacing(12)
        self.data_source_group = QButtonGroup(self)
        self.use_builtin_radio = QRadioButton("Use built-in datasets")
        self.use_custom_radio = QRadioButton("Use your own data")
        self.use_builtin_radio.setChecked(True)
        self.data_source_group.addButton(self.use_builtin_radio)
        self.data_source_group.addButton(self.use_custom_radio)
        self.use_builtin_radio.toggled.connect(self._update_data_source)
        self.use_custom_radio.toggled.connect(self._update_data_source)
        source_toggle = QHBoxLayout()
        source_toggle.addWidget(self.use_builtin_radio)
        source_toggle.addWidget(self.use_custom_radio)
        source_toggle.addStretch()
        source_layout.addLayout(source_toggle)

        self.author_combo = QComboBox()
        self.author_combo.setView(QListView())
        self.author_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.author_combo.currentTextChanged.connect(self._on_author_changed)
        self.author_row = QWidget()
        author_layout = QFormLayout(self.author_row)
        author_layout.addRow("Author / Dataset", self.author_combo)
        source_layout.addWidget(self.author_row)

        self.component_row = QWidget()
        component_layout = QFormLayout(self.component_row)
        self.component_label = QLabel("Component")
        self.component_combo = QComboBox()
        self.component_combo.setView(QListView())
        self.component_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.component_combo.currentIndexChanged.connect(self._on_component_changed)
        component_layout.addRow(self.component_label, self.component_combo)
        self.component_row.setVisible(False)
        source_layout.addWidget(self.component_row)
        source_box.setLayout(source_layout)

        self.builtin_widget = QWidget()
        builtin_layout = QVBoxLayout(self.builtin_widget)
        builtin_layout.setContentsMargins(0, 0, 0, 0)
        builtin_layout.setSpacing(10)
        self.modes_grid = QGridLayout()
        self.modes_grid.setHorizontalSpacing(8)
        self.modes_grid.setVerticalSpacing(12)
        self.modes_grid.setContentsMargins(0, 6, 0, 6)
        builtin_layout.addLayout(self.modes_grid)

        self.custom_widget = QWidget()
        custom_layout = QVBoxLayout(self.custom_widget)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        custom_layout.setSpacing(10)
        custom_header = QHBoxLayout()
        custom_title = QLabel("Custom datasets")
        custom_title.setFont(make_font(14, QFont.DemiBold))
        add_btn = QToolButton()
        add_btn.setText("+")
        add_btn.setObjectName("iconButton")
        add_btn.clicked.connect(self._add_custom_entry)
        custom_header.addWidget(custom_title)
        custom_header.addStretch()
        custom_header.addWidget(add_btn)
        custom_layout.addLayout(custom_header)
        self.custom_tabs = QTabWidget()
        self.custom_tabs.setTabsClosable(False)
        self.custom_tabs.setDocumentMode(True)
        self.custom_tabs.tabBar().setDrawBase(False)
        self.custom_tabs.tabBar().setExpanding(False)
        custom_layout.addWidget(self.custom_tabs)

        saved_header = QHBoxLayout()
        saved_label = QLabel("Saved datasets")
        saved_label.setFont(make_font(13, QFont.DemiBold))
        self.custom_save_btn = QPushButton("Save")
        self.custom_save_btn.setObjectName("secondaryButton")
        self.custom_save_btn.clicked.connect(self._save_custom_datasets)
        saved_header.addWidget(saved_label)
        saved_header.addStretch()
        saved_header.addWidget(self.custom_save_btn)
        custom_layout.addLayout(saved_header)
        self.custom_saved_list = QListWidget()
        self.custom_saved_list.setMinimumHeight(120)
        self.custom_saved_list.setSpacing(6)
        custom_layout.addWidget(self.custom_saved_list)
        self.custom_widget.setVisible(False)

        self.custom_entries = []
        self.saved_custom_sets = []
        self.saved_custom_counter = 0

        self.reference_label = QLabel()
        self.reference_label.setTextFormat(Qt.RichText)
        self.reference_label.setOpenExternalLinks(True)
        self.reference_label.setWordWrap(True)

        self.preview_canvas = MatplotlibCanvas(width=8.8, height=4.6)
        self.preview_canvas.setMinimumHeight(360)

        preview_actions = QHBoxLayout()
        preview_actions.addStretch()
        self.preview_save_btn = QPushButton("Save Preview Plot")
        self.preview_save_btn.setEnabled(False)
        self.preview_save_btn.clicked.connect(lambda: self._save_figure(self.preview_canvas, "preview_plot"))
        preview_actions.addWidget(self.preview_save_btn)

        self.data_next_btn = QPushButton("Next: Model Architecture")
        self.data_next_btn.setEnabled(False)
        self.data_next_btn.clicked.connect(lambda: self._set_step(1))
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        source_layout.addWidget(self.builtin_widget)
        source_layout.addWidget(self.custom_widget)
        left_layout.addWidget(source_box)
        self.data_reference_title = QLabel("Reference")
        self.data_reference_title.setFont(make_font(12, QFont.DemiBold))
        self.data_reference_title.setVisible(False)
        left_layout.addWidget(self.data_reference_title)
        left_layout.addWidget(self.reference_label)
        left_layout.addStretch()
        left_layout.addWidget(self.data_next_btn, alignment=Qt.AlignRight)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        right_layout.addWidget(self.preview_canvas)
        right_layout.addLayout(preview_actions)

        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel, 2)

        self.data_box.setLayout(layout)
        if not self.custom_entries:
            self._add_custom_entry(update_preview=False)
        return self.data_box

    def _build_model_section(self):
        self.model_box = QGroupBox("2. Model Architecture")
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.spring_count = 1
        self.max_springs = 6
        self.spring_widgets = []
        self.spring_icons = []

        self.add_spring_btn = QPushButton("Add Parallel Spring")
        self.add_spring_btn.clicked.connect(self._add_spring)
        self.add_spring_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        button_row = QHBoxLayout()
        button_row.addWidget(self.add_spring_btn)
        button_row.addStretch()
        layout.addLayout(button_row)

        self.spring_rows_widget = QWidget()
        self.spring_rows_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        spring_rows_layout = QHBoxLayout(self.spring_rows_widget)
        spring_rows_layout.setContentsMargins(0, 0, 0, 0)
        spring_rows_layout.setSpacing(8)
        self.spring_grid = QGridLayout()
        self.spring_grid.setContentsMargins(0, 0, 0, 0)
        self.spring_grid.setHorizontalSpacing(14)
        self.spring_grid.setVerticalSpacing(12)
        self.spring_grid.setColumnStretch(0, 0)
        self.spring_grid.setColumnStretch(1, 0)
        spring_rows_layout.addLayout(self.spring_grid)
        spring_rows_layout.addStretch()
        layout.addWidget(self.spring_rows_widget)
        layout.addStretch()

        self.model_next_btn = QPushButton("Next: Optimization")
        self.model_next_btn.setEnabled(False)
        self.model_next_btn.clicked.connect(lambda: self._set_step(2))
        layout.addWidget(self.model_next_btn, alignment=Qt.AlignRight)

        self.model_box.setLayout(layout)

        self._rebuild_springs(self.spring_count)
        return self.model_box

    def _build_optimization_section(self):
        self.opt_box = QGroupBox("3. Optimization")
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(16, 16, 16, 16)
        outer_layout.setSpacing(12)
        body_layout = QHBoxLayout()
        body_layout.setSpacing(16)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        method_label = QLabel("Optimization method")
        method_label.setFont(make_font(12, QFont.DemiBold))
        self.method_combo = QComboBox()
        self.method_combo.setView(QListView())
        self.method_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.method_combo.addItem("L-BFGS-B", "L-BFGS-B")
        self.method_combo.addItem("Trust-Region Reflective (lsqnonlin)", "trf")
        self.method_combo.addItem("Levenberg-Marquardt (lsqnonlin)", "lm")
        self.method_combo.addItem("Dogbox (lsqnonlin)", "dogbox")
        left_layout.addWidget(method_label)
        left_layout.addWidget(self.method_combo)

        self.calib_params_box = QGroupBox("Parameters")
        params_layout = QVBoxLayout()
        self.calib_params_area = QScrollArea()
        self.calib_params_area.setWidgetResizable(True)
        self.calib_params_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.calib_params_area.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.calib_params_widget = QWidget()
        self.calib_params_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.calib_params_layout = QVBoxLayout(self.calib_params_widget)
        self.calib_params_layout.setSpacing(8)
        self.calib_params_layout.setAlignment(Qt.AlignTop)
        self.calib_params_layout.addStretch()
        self.calib_params_area.setWidget(self.calib_params_widget)
        params_layout.addWidget(self.calib_params_area)
        self.calib_params_box.setLayout(params_layout)
        left_layout.addWidget(self.calib_params_box, 1)

        self.run_button = QPushButton("Start Calibration")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(self._run_optimization)
        left_layout.addWidget(self.run_button)

        self.opt_status = QLabel("")
        left_layout.addWidget(self.opt_status)

        self.loss_label = QLabel("Loss: -")
        left_layout.addWidget(self.loss_label)

        self.opt_log_box = QGroupBox("Iteration log")
        log_layout = QVBoxLayout()
        self.opt_log = QPlainTextEdit()
        self.opt_log.setReadOnly(True)
        self.opt_log.setMaximumBlockCount(2000)
        self.opt_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        log_layout.addWidget(self.opt_log)
        self.opt_log_box.setLayout(log_layout)
        left_layout.addWidget(self.opt_log_box, 2)

        left_layout.addStretch()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.calib_results_box = QGroupBox("Calibration Plot")
        calib_layout = QVBoxLayout()
        self.calib_dataset_label = QLabel("Dataset: -")
        self.calib_dataset_label.setFont(make_font(12, QFont.DemiBold))
        self.calib_dataset_label.setWordWrap(True)
        self.calib_dataset_label.setStyleSheet("color: palette(windowText);")
        calib_layout.addWidget(self.calib_dataset_label)
        self.calib_canvas = MatplotlibCanvas(width=8.0, height=4.4)
        self.calib_canvas.setMinimumHeight(340)
        calib_layout.addWidget(self.calib_canvas, 1)
        calib_actions = QHBoxLayout()
        calib_actions.addStretch()
        self.calib_save_btn = QPushButton("Save Calibration Plot")
        self.calib_save_btn.setEnabled(False)
        self.calib_save_btn.clicked.connect(lambda: self._save_figure(self.calib_canvas, "calibration_plot"))
        calib_actions.addWidget(self.calib_save_btn)
        calib_layout.addLayout(calib_actions)
        self.calib_results_box.setLayout(calib_layout)
        right_layout.addWidget(self.calib_results_box, 1)

        body_layout.addWidget(left_panel, 1)
        body_layout.addWidget(right_panel, 3)

        outer_layout.addLayout(body_layout)

        self.opt_next_btn = QPushButton("Next: Prediction")
        self.opt_next_btn.setEnabled(False)
        self.opt_next_btn.clicked.connect(lambda: self._set_step(3))
        outer_layout.addWidget(self.opt_next_btn, alignment=Qt.AlignRight)

        self.opt_box.setLayout(outer_layout)
        self._refresh_opt_params_from_springs()
        return self.opt_box

    def _build_prediction_section(self):
        self.prediction_box = QGroupBox("4. Prediction")
        layout = QHBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        self.prediction_params_box = QGroupBox("Parameters")
        pred_params_layout = QVBoxLayout()
        self.prediction_params_area = QScrollArea()
        self.prediction_params_area.setWidgetResizable(True)
        self.prediction_params_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.prediction_params_area.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.prediction_params_widget = QWidget()
        self.prediction_params_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.prediction_params_layout = QVBoxLayout(self.prediction_params_widget)
        self.prediction_params_layout.setSpacing(8)
        self.prediction_params_layout.setAlignment(Qt.AlignTop)
        self.prediction_params_layout.addStretch()
        self.prediction_params_area.setWidget(self.prediction_params_widget)
        pred_params_layout.addWidget(self.prediction_params_area)
        self.prediction_params_box.setLayout(pred_params_layout)
        left_layout.addWidget(self.prediction_params_box)

        self.prediction_modes_box = QGroupBox("Prediction Data")
        modes_layout = QVBoxLayout()
        modes_layout.setContentsMargins(12, 12, 12, 12)
        modes_layout.setSpacing(10)
        self.prediction_author_label = QLabel("Author: -")
        self.prediction_author_label.setFont(make_font(13, QFont.DemiBold))
        modes_layout.addWidget(self.prediction_author_label)
        self.prediction_component_row = QWidget()
        pred_component_layout = QFormLayout(self.prediction_component_row)
        self.prediction_component_label = QLabel("Component")
        self.prediction_component_combo = QComboBox()
        self.prediction_component_combo.setView(QListView())
        self.prediction_component_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.prediction_component_combo.currentIndexChanged.connect(self._on_prediction_component_changed)
        pred_component_layout.addRow(self.prediction_component_label, self.prediction_component_combo)
        self.prediction_component_row.setVisible(False)
        modes_layout.addWidget(self.prediction_component_row)
        self.prediction_modes_widget = QWidget()
        self.prediction_modes_layout = QVBoxLayout(self.prediction_modes_widget)
        self.prediction_modes_layout.setContentsMargins(0, 6, 0, 6)
        self.prediction_modes_layout.setSpacing(10)
        modes_layout.addWidget(self.prediction_modes_widget)
        self.prediction_button = QPushButton("Update Prediction")
        self.prediction_button.clicked.connect(self._update_prediction_plot)
        modes_layout.addWidget(self.prediction_button)
        self.prediction_modes_box.setLayout(modes_layout)
        left_layout.addWidget(self.prediction_modes_box)
        left_layout.addStretch()

        layout.addWidget(left_panel, 1)

        self.prediction_canvas = MatplotlibCanvas(width=8.0, height=4.4)
        self.prediction_canvas.setMinimumHeight(340)
        pred_plot_container = QWidget()
        pred_plot_layout = QVBoxLayout(pred_plot_container)
        pred_plot_layout.setContentsMargins(0, 0, 0, 0)
        pred_plot_layout.addWidget(self.prediction_canvas, 1)
        pred_actions = QHBoxLayout()
        pred_actions.addStretch()
        self.pred_save_btn = QPushButton("Save Prediction Plot")
        self.pred_save_btn.setEnabled(False)
        self.pred_save_btn.clicked.connect(lambda: self._save_figure(self.prediction_canvas, "prediction_plot"))
        pred_actions.addWidget(self.pred_save_btn)
        pred_plot_layout.addLayout(pred_actions)
        layout.addWidget(pred_plot_container, 3)

        self.prediction_box.setLayout(layout)
        return self.prediction_box

    def _update_data_source(self):
        use_builtin = self.use_builtin_radio.isChecked()
        self.builtin_widget.setVisible(use_builtin)
        self.custom_widget.setVisible(not use_builtin)
        if hasattr(self, "author_row"):
            self.author_row.setVisible(use_builtin)
        if use_builtin:
            self.reference_label.setVisible(True)
            if hasattr(self, "data_reference_title"):
                self.data_reference_title.setVisible(bool(self.reference_label.text()))
            self._reset_custom_entries()
        else:
            self.reference_label.setText("")
            self.reference_label.setVisible(False)
            if hasattr(self, "data_reference_title"):
                self.data_reference_title.setVisible(False)
            self._clear_builtin_selection()
            if not self.custom_entries:
                self._add_custom_entry(update_preview=False)
        self._update_preview()

    def _clear_builtin_selection(self):
        if not hasattr(self, "author_combo"):
            return
        self.author_combo.blockSignals(True)
        self.author_combo.setCurrentIndex(0)
        self.author_combo.blockSignals(False)
        self._on_author_changed("Select...")

    def _reset_custom_entries(self):
        for entry in list(self.custom_entries):
            tab_index = self.custom_tabs.indexOf(entry)
            if tab_index != -1:
                self.custom_tabs.removeTab(tab_index)
            entry.setParent(None)
            entry.deleteLater()
        self.custom_entries = []
        self._clear_saved_custom_sets()
        self._add_custom_entry(update_preview=False)

    def _sync_custom_tabs(self):
        for idx, entry in enumerate(self.custom_entries, start=1):
            entry.set_index(idx)
            tab_index = self.custom_tabs.indexOf(entry)
            if tab_index != -1:
                self.custom_tabs.setTabText(tab_index, str(idx))

    def _add_custom_entry(self, update_preview=True):
        entry = CustomDataEntry(
            len(self.custom_entries) + 1,
            on_change=self._update_preview,
            on_remove=self._remove_custom_entry,
        )
        self.custom_entries.append(entry)
        self.custom_tabs.addTab(entry, str(entry.index))
        self.custom_tabs.setCurrentWidget(entry)
        self._sync_custom_tabs()
        if update_preview:
            self._update_preview()

    def _remove_custom_entry(self, entry):
        if entry in self.custom_entries:
            self.custom_entries.remove(entry)
            tab_index = self.custom_tabs.indexOf(entry)
            if tab_index != -1:
                self.custom_tabs.removeTab(tab_index)
            entry.setParent(None)
            entry.deleteLater()
            self._clear_saved_custom_sets()
        if not self.custom_entries:
            self._add_custom_entry(update_preview=False)
        self._sync_custom_tabs()
        self._update_preview()

    def _load_custom_entries(self, validate_lengths=True, show_message=True):
        data = []
        for entry in self.custom_entries:
            try:
                item = entry.get_data(validate_lengths=validate_lengths)
            except ValueError as exc:
                if show_message:
                    QMessageBox.warning(self, "Custom data error", str(exc))
                return None
            if item:
                data.append(item)
        return data

    def _format_saved_summary(self, data):
        parts = []
        for entry in data:
            base_label = entry.get("label") or format_mode_label(entry.get("mode_raw", entry.get("mode", "")))
            stress_type = entry.get("stress_type", "PK1")
            stress_label = "Cauchy" if stress_type == "cauchy" else "Nominal"
            parts.append(f"{base_label} ({stress_label})")
        return "; ".join(parts)

    def _build_saved_signature(self, data):
        def _to_tuple(value):
            if value is None:
                return None
            if isinstance(value, np.ndarray):
                if value.ndim > 1:
                    return tuple(tuple(row) for row in value.tolist())
                return tuple(value.tolist())
            if isinstance(value, (list, tuple)):
                return tuple(_to_tuple(item) for item in value)
            return value

        entries = []
        for entry in data:
            signature = (
                entry.get("mode"),
                entry.get("stress_type"),
                _to_tuple(entry.get("stretch")),
                _to_tuple(entry.get("stretch_secondary")),
                _to_tuple(entry.get("stress_exp")),
            )
            entries.append(signature)
        return tuple(entries)

    def _add_saved_dataset_entry(self, label, data, saved_id):
        row = SavedDatasetRow(label, on_remove=self._remove_saved_dataset)
        item = QListWidgetItem(self.custom_saved_list)
        item.setSizeHint(row.sizeHint())
        self.custom_saved_list.addItem(item)
        self.custom_saved_list.setItemWidget(item, row)
        signature = self._build_saved_signature(data)
        self.saved_custom_sets.append(
            {
                "label": label,
                "data": data,
                "item": item,
                "row": row,
                "signature": signature,
                "saved_id": saved_id,
            }
        )

    def _remove_saved_dataset(self, row):
        for idx, saved in enumerate(list(self.saved_custom_sets)):
            if saved.get("row") is row:
                item = saved.get("item")
                if item:
                    self.custom_saved_list.takeItem(self.custom_saved_list.row(item))
                row.deleteLater()
                self.saved_custom_sets.pop(idx)
                break

    def _clear_saved_custom_sets(self):
        if hasattr(self, "custom_saved_list"):
            self.custom_saved_list.clear()
        self.saved_custom_sets = []
        self.saved_custom_counter = 0

    def _save_custom_datasets(self):
        data = self._load_custom_entries(validate_lengths=True, show_message=True)
        if data is None:
            return
        if not data:
            QMessageBox.information(self, "Save custom datasets", "Add data before saving.")
            return
        summary = self._format_saved_summary(data)
        signature = self._build_saved_signature(data)
        for saved in self.saved_custom_sets:
            if saved.get("signature") == signature:
                label = f"Saved {saved.get('saved_id', 1)}: {summary}"
                saved["label"] = label
                saved["data"] = data
                saved["signature"] = signature
                if saved.get("row"):
                    saved["row"].label.setText(label)
                return
        self.saved_custom_counter += 1
        label = f"Saved {self.saved_custom_counter}: {summary}"
        self._add_saved_dataset_entry(label, data, self.saved_custom_counter)

    def _collect_experimental_data(self):
        data = []
        if self.use_builtin_radio.isChecked():
            author = self.author_combo.currentText()
            modes = self._selected_modes()
            if author != "Select..." and modes:
                configs = [{"author": author, "mode": m} for m in modes]
                data.extend(load_experimental_data(configs))
        else:
            custom_data = self._load_custom_entries(validate_lengths=False, show_message=False)
            if custom_data:
                data.extend(custom_data)
        return data

    def _save_figure(self, canvas, default_name):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save plot",
            f"{default_name}.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not file_path:
            return
        try:
            canvas.figure.savefig(file_path, dpi=300, bbox_inches="tight", transparent=True)
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", f"Could not save plot: {exc}")

    def _export_report(self):
        if not self.latest_optimizer or not self.latest_result:
            QMessageBox.information(self, "Report", "Run calibration first.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export report",
            "calibration_report.pdf",
            "PDF (*.pdf)",
        )
        if not file_path:
            return
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            with matplotlib.rc_context({"font.family": "Times New Roman"}):
                with PdfPages(file_path) as pdf:
                    # Cover page with summary
                    fig = Figure(figsize=(8.5, 11), dpi=150)
                    ax = fig.add_subplot(111)
                    ax.axis("off")
                    optimizer = self.latest_optimizer
                    lines = []
                    lines.append("Calibration for Hyperelasticity")
                    lines.append("")
                    lines.append(f"Final loss: {self.latest_result.fun:.6g}")
                    lines.append("")
                    if self.use_builtin_radio.isChecked():
                        lines.append(f"Dataset: {self.author_combo.currentText()}")
                        lines.append(f"Modes: {', '.join(self._selected_modes())}")
                    else:
                        lines.append("Dataset: Custom")
                        lines.append(f"Custom entries: {len(self.custom_entries)}")
                    lines.append("")
                    lines.append("Model architecture:")
                    for idx, spring in enumerate(self.spring_widgets):
                        state = spring.get_state()
                        is_custom = state.get("model_source") == "custom" or state.get("use_custom")
                        if is_custom:
                            lines.append(f"  Spring {idx+1}: Custom ({state.get('custom_type')})")
                            lines.append(f"    Formula: {state.get('custom_formula')}")
                            lines.append(f"    Params: {state.get('custom_params')}")
                        else:
                            lines.append(f"  Spring {idx+1}: {state.get('model')}")
                    lines.append("")
                    lines.append("Optimized parameters:")
                    for name, value in zip(optimizer.param_names, self.latest_result.x):
                        lines.append(f"  {name}: {value:.6g}")
                    ax.text(0.05, 0.98, "\n".join(lines), va="top", fontsize=10)
                    pdf.savefig(fig, bbox_inches="tight")
                    fig.clear()

                    # Formula pages
                    for idx, spring in enumerate(self.spring_widgets):
                        state = spring.get_state()
                        fig = Figure(figsize=(8.5, 5.5), dpi=150)
                        ax = fig.add_subplot(111)
                        ax.axis("off")
                        is_custom = state.get("model_source") == "custom" or state.get("use_custom")
                        if is_custom:
                            formula = state.get("custom_formula", "")
                            title = f"Spring {idx+1} Formula (Custom)"
                        else:
                            model_name = resolve_model_name(state.get("model", ""))
                            func = None
                            if model_name == "Ogden":
                                func = MaterialModels.create_ogden_model(state.get("ogden_terms", 1))
                            elif model_name == "Hill":
                                func = MaterialModels.create_hill_model(state.get("strain"))
                            elif model_name:
                                func = getattr(MaterialModels, model_name, None)
                            formula = getattr(func, "formula", "") if func else ""
                            title = f"Spring {idx+1} Formula"
                        ax.text(0.05, 0.9, title, fontsize=12, fontweight="bold")
                        if formula:
                            ax.text(0.05, 0.6, f"${formula}$", fontsize=12)
                        pdf.savefig(fig, bbox_inches="tight")
                        fig.clear()

                    # Experimental data table
                    data = self._collect_experimental_data()
                    fig = Figure(figsize=(8.5, 11), dpi=150)
                    ax = fig.add_subplot(111)
                    ax.axis("off")
                    table_lines = ["Experimental data summary:"]
                    for d in data:
                        mode_label = d.get("label") or format_mode_label(d.get("mode_raw", d.get("mode")))
                        stress_type = get_stress_type_label(d.get("stress_type", "PK1"))
                        table_lines.append(
                            f"- {mode_label} ({stress_type}), points: {len(d.get('stretch', []))}"
                        )
                    ax.text(0.05, 0.98, "\n".join(table_lines), va="top", fontsize=10)
                    pdf.savefig(fig, bbox_inches="tight")
                    fig.clear()

                    # Plots
                    pdf.savefig(self.calib_canvas.figure, bbox_inches="tight")
                    if self.prediction_canvas and self.prediction_canvas.figure:
                        pdf.savefig(self.prediction_canvas.figure, bbox_inches="tight")
        except Exception as exc:
            QMessageBox.warning(self, "Report failed", f"Could not export report:\n{exc}")

    def _populate_authors(self):
        self.author_combo.blockSignals(True)
        self.author_combo.clear()
        self.author_combo.addItem("Select...")
        for author in sorted(self.datasets.keys()):
            self.author_combo.addItem(author)
        self.author_combo.blockSignals(False)

    def _on_component_changed(self, _index):
        if not hasattr(self, "component_combo") or not hasattr(self, "modes_grid"):
            return
        author = self.author_combo.currentText()
        component_key = self.component_combo.currentData()
        if not author or author == "Select..." or not component_key:
            return
        self._build_component_modes(author, component_key)
        self._update_preview()

    def _build_component_modes(self, author, component_key):
        for i in reversed(range(self.modes_grid.count())):
            widget = self.modes_grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        config = COMPONENT_AUTHOR_CONFIG.get(author)
        if not config:
            return
        prefixes = config.get("mode_prefixes", ())
        mode_labels = config.get("mode_labels", {})
        available = set(self.datasets.get(author, []))
        row = 0
        for prefix in prefixes:
            if prefix == "CSS_":
                compound_modes = [
                    m for m in available
                    if m.startswith(prefix) and extract_component_from_mode(m, (prefix,)) == component_key
                ]
                for mode_key in sorted(compound_modes):
                    label = format_mode_label(mode_key)
                    checkbox = QCheckBox(label)
                    checkbox.setObjectName("modeOption")
                    checkbox.setMinimumHeight(26)
                    checkbox.setProperty("mode_key", mode_key)
                    checkbox.stateChanged.connect(self._update_preview)
                    self.modes_grid.addWidget(checkbox, row, 0)
                    row += 1
                continue
            mode_key = f"{prefix}{component_key}"
            if mode_key not in available:
                continue
            label = mode_labels.get(prefix, format_mode_label(mode_key))
            checkbox = QCheckBox(label)
            checkbox.setObjectName("modeOption")
            checkbox.setMinimumHeight(26)
            checkbox.setProperty("mode_key", mode_key)
            checkbox.stateChanged.connect(self._update_preview)
            self.modes_grid.addWidget(checkbox, row, 0)
            row += 1

    def _on_author_changed(self, text):
        self._reset_results()
        self._bt_mix_warned = False
        if hasattr(self, "component_row"):
            self.component_row.setVisible(False)
        if hasattr(self, "component_combo"):
            self.component_combo.blockSignals(True)
            self.component_combo.clear()
            self.component_combo.blockSignals(False)
        for i in reversed(range(self.modes_grid.count())):
            widget = self.modes_grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        if text == "Select..." or text not in self.datasets:
            self.reference_label.setText("")
            if hasattr(self, "data_reference_title"):
                self.data_reference_title.setVisible(False)
            self.preview_canvas.ax.clear()
            self.preview_canvas.apply_theme()
            self.preview_canvas.draw()
            self.data_next_btn.setEnabled(False)
            return

        modes = self.datasets[text]
        component_config = COMPONENT_AUTHOR_CONFIG.get(text)
        if component_config:
            prefixes = component_config.get("mode_prefixes", ())
            component_modes = [m for m in modes if extract_component_from_mode(m, prefixes)]
            if component_modes:
                self.component_label.setText(component_config.get("component_label", "Component"))
                self.component_combo.blockSignals(True)
                self.component_combo.clear()
                component_keys = []
                for mode in component_modes:
                    component = extract_component_from_mode(mode, prefixes)
                    if component:
                        component_keys.append(component)
                for component in sorted(set(component_keys)):
                    self.component_combo.addItem(format_component_label(component), component)
                self.component_combo.setCurrentIndex(0)
                self.component_combo.blockSignals(False)
                self.component_row.setVisible(True)
                component_key = self.component_combo.currentData()
                if component_key:
                    self._build_component_modes(text, component_key)
            else:
                component_config = None
        if not component_config:
            for idx, mode in enumerate(modes):
                checkbox = QCheckBox(format_mode_label(mode))
                checkbox.setObjectName("modeOption")
                checkbox.setMinimumHeight(26)
                checkbox.setProperty("mode_key", mode)
                checkbox.stateChanged.connect(self._update_preview)
                row = idx
                col = 0
                self.modes_grid.addWidget(checkbox, row, col)

        reference = DATASET_REFERENCES.get(text)
        if reference:
            label, url = reference
            if url:
                self.reference_label.setText(f"<a href='{url}'>{label}</a>")
            else:
                self.reference_label.setText(label)
            if hasattr(self, "data_reference_title"):
                self.data_reference_title.setVisible(True)
        else:
            self.reference_label.setText("")
            if hasattr(self, "data_reference_title"):
                self.data_reference_title.setVisible(False)

        self._update_preview()

    def _selected_modes(self):
        modes = []
        for i in range(self.modes_grid.count()):
            widget = self.modes_grid.itemAt(i).widget()
            if isinstance(widget, QCheckBox) and widget.isChecked():
                mode_key = widget.property("mode_key") or widget.text()
                modes.append(str(mode_key))
        return modes

    def _format_modes_summary(self, modes):
        if not modes:
            return ""
        labels = [format_mode_label(m) for m in modes]
        if len(labels) <= 3:
            return ", ".join(labels)
        return ", ".join(labels[:3]) + f", +{len(labels) - 3} more"

    def _format_calibration_dataset_label(self):
        if not hasattr(self, "use_builtin_radio") or not hasattr(self, "author_combo"):
            return "Dataset: -"
        if self.use_builtin_radio.isChecked():
            author = self.author_combo.currentText()
            if not author or author == "Select...":
                return "Dataset: -"
            parts = [f"Dataset: {author}"]
            component_key = None
            if hasattr(self, "component_combo") and self.component_combo.isVisible():
                component_key = self.component_combo.currentData()
            if not component_key:
                component_config = COMPONENT_AUTHOR_CONFIG.get(author)
                prefixes = component_config.get("mode_prefixes", ()) if component_config else ()
                for mode in self._selected_modes():
                    component_key = extract_component_from_mode(mode, prefixes)
                    if component_key:
                        break
            if component_key:
                parts.append(f"Component: {format_component_label(component_key)}")
            modes = self._selected_modes()
            summary = self._format_modes_summary(modes)
            if summary:
                parts.append(f"Modes: {summary}")
            return " • ".join(parts)
        count = len(getattr(self, "custom_entries", []))
        return f"Dataset: Custom • Entries: {count}"

    def _update_preview(self):
        self._reset_results()
        author = self.author_combo.currentText()
        data = self._collect_experimental_data()
        if not data:
            self.preview_canvas.reset_axes()
            self.preview_canvas.draw()
            self.data_next_btn.setEnabled(False)
            self.preview_save_btn.setEnabled(False)
            self._update_workflow_cards()
            return
        bt_selected = any(d["mode"] == "BT" for d in data)
        non_bt_selected = any(d["mode"] != "BT" for d in data)
        if bt_selected and non_bt_selected and not self._bt_mix_warned:
            QMessageBox.information(
                self,
                "Mixed loading modes",
                "Selecting BT together with other modes is not recommended. "
                "BT datasets are typically used for prediction.",
            )
            self._bt_mix_warned = True

        if data and all(d["mode"] == "BT" for d in data):
            self._plot_bt_preview(self.preview_canvas, data)
            self.preview_canvas.figure.tight_layout()
            self.preview_canvas.draw()
            self.data_next_btn.setEnabled(True)
            self.preview_save_btn.setEnabled(True)
            self._update_workflow_cards()
            return

        ax = self.preview_canvas.reset_axes()
        for idx, d in enumerate(data):
            stretch = d["stretch"]
            stress = d["stress_exp"]
            stress_type = d.get("stress_type", "PK1")
            component = d.get("component")
            label = d.get("label") or format_mode_label(d.get("mode_raw", d["mode"]))
            if d["mode"] == "BT":
                if component:
                    comp_label = get_component_label(stress_type, component)
                    ax.plot(stretch, stress, "o", label=f"{label} {comp_label}")
                else:
                    comp_11, comp_22 = get_bt_component_labels(stress_type)
                    if np.ndim(stress) == 1:
                        ax.plot(stretch, stress, "o", label=f"{label} {comp_11}")
                    else:
                        ax.plot(stretch, stress[:, 0], "o", label=f"{label} {comp_11}")
                        ax.plot(stretch, stress[:, 1], "^", label=f"{label} {comp_22}")
            else:
                if component:
                    comp_label = get_component_label(stress_type, component)
                    ax.plot(stretch, stress, "o", label=f"{label} {comp_label}")
                elif np.ndim(stress) == 1:
                    ax.plot(stretch, stress, "o", label=label)
                else:
                    comp_11, comp_22 = get_bt_component_labels(stress_type)
                    ax.plot(stretch, stress[:, 0], "o", label=f"{label} {comp_11}")
                    ax.plot(stretch, stress[:, 1], "^", label=f"{label} {comp_22}")
        non_bt_modes = [d["mode"] for d in data if d["mode"] != "BT"]
        ax.set_xlabel(choose_xlabel_for_modes(non_bt_modes))
        stress_types = {d.get("stress_type", "PK1") for d in data if d["mode"] != "BT"}
        has_multi = any(
            d["mode"] != "BT" and (np.ndim(d.get("stress_exp")) > 1 or d.get("component"))
            for d in data
        )
        if non_bt_modes and all(m in ("SS", "CSS") for m in non_bt_modes) and not has_multi:
            if len(stress_types) == 1:
                ax.set_ylabel(get_shear_component_label(next(iter(stress_types))))
            else:
                ax.set_ylabel(r"$P_{12}$")
        elif has_multi:
            if len(stress_types) == 1:
                stress_type = next(iter(stress_types))
                ax.set_ylabel(r"$\sigma$" if stress_type == "cauchy" else r"$P$")
            else:
                ax.set_ylabel(r"$P$")
        elif len(stress_types) == 1:
            ax.set_ylabel(get_uniaxial_component_label(next(iter(stress_types))))
        else:
            ax.set_ylabel(r"$P_{11}$")
        ax.legend(fontsize=8)
        self.preview_canvas.figure.tight_layout()
        self.preview_canvas.draw()
        self.preview_save_btn.setEnabled(True)
        self.data_next_btn.setEnabled(True)
        self._update_workflow_cards()

    def _rebuild_springs(self, count, states_override=None):
        count = max(1, min(int(count), getattr(self, "max_springs", 6)))
        self.spring_count = count
        if hasattr(self, "add_spring_btn"):
            self.add_spring_btn.setEnabled(count < getattr(self, "max_springs", 6))
        previous_states = []
        if states_override is not None:
            previous_states = list(states_override)
        else:
            for widget in getattr(self, "spring_widgets", []):
                if widget:
                    previous_states.append(widget.get_state())
        if hasattr(self, "spring_grid"):
            while self.spring_grid.count():
                item = self.spring_grid.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
        self.spring_widgets = []
        self.spring_icons = []

        for i in range(1, count + 1):
            spring_icon = SpringIcon()
            self.spring_grid.addWidget(spring_icon, i - 1, 0, Qt.AlignTop)
            spring_widget = SpringWidget(
                i,
                on_change=self._on_spring_config_changed,
                author_provider=lambda: self.author_combo.currentText(),
                on_remove=self._remove_spring,
            )
            if i <= len(previous_states):
                spring_widget.apply_state(previous_states[i - 1])
            spring_widget.set_removable(count > 1)
            self.spring_grid.addWidget(spring_widget, i - 1, 1)
            self.spring_icons.append(spring_icon)
            self.spring_widgets.append(spring_widget)
        self._on_spring_config_changed()

    def _add_spring(self):
        if getattr(self, "spring_count", 1) >= getattr(self, "max_springs", 6):
            return
        self._rebuild_springs(self.spring_count + 1)

    def _remove_spring(self, index):
        if getattr(self, "spring_count", 1) <= 1:
            return
        states = [widget.get_state() for widget in self.spring_widgets]
        if 1 <= index <= len(states):
            states.pop(index - 1)
        if not states:
            self._rebuild_springs(1)
            return
        self._rebuild_springs(len(states), states_override=states)

    def _set_optimization_controls(self, running):
        self._optimization_running = bool(running)
        if hasattr(self, "run_button"):
            self.run_button.setEnabled(True)
            self.run_button.setText("Abort Calibration" if running else "Start Calibration")

    def _abort_optimization(self, wait=False):
        worker = getattr(self, "worker", None)
        if not worker or not worker.isRunning():
            return
        worker.stop()
        if wait:
            worker.wait(2500)

    def _run_optimization(self):
        if self._optimization_running:
            self._abort_optimization(wait=False)
            return
        self._reset_prediction_results()
        self._clear_prediction_selection()
        if hasattr(self, "calib_save_btn"):
            self.calib_save_btn.setEnabled(False)
        if hasattr(self, "calib_dataset_label"):
            dataset_text = self._format_calibration_dataset_label()
            self.calib_dataset_label.setText(dataset_text)
            self.calib_dataset_label.setToolTip(dataset_text)
        exp_data = self._collect_experimental_data()
        if not exp_data:
            QMessageBox.warning(self, "Missing data", "Select at least one dataset or load custom data.")
            return

        execution_network = ParallelNetwork()
        initial_guess = []
        for idx, spring in enumerate(self.spring_widgets):
            if not spring.is_valid():
                if getattr(spring, "_custom_error", ""):
                    QMessageBox.warning(self, "Model error", f"Custom model error:\n{spring._custom_error}")
                else:
                    QMessageBox.warning(self, "Model error", "Select a model for each spring.")
                return
            func, params = spring.build_config()
            execution_network.add_model(func, f"{func.__name__}_{idx+1}")
            initial_guess.extend(params)

        solver = Kinematics(execution_network, execution_network.param_names)
        optimizer = MaterialOptimizer(solver, exp_data)
        bounds = execution_network.bounds

        method = self.method_combo.currentData() or self.method_combo.currentText()
        self.opt_status.setText("Running optimization...")
        self.opt_status.setStyleSheet("color: palette(windowtext);")
        self.loss_label.setText("Loss: -")
        if hasattr(self, "opt_log"):
            self.opt_log.clear()
        self._populate_calibration_params(execution_network.param_names, initial_guess)
        self._set_optimization_controls(True)
        self._set_step(2)

        self.worker = OptimizerWorker(optimizer, initial_guess, bounds, method)
        self.worker.progress.connect(self._on_optimization_progress)
        self.worker.finished.connect(self._on_optimization_finished)
        self.worker.failed.connect(self._on_optimization_failed)
        self.worker.start()

    def _on_optimization_finished(self, result, optimizer):
        self._set_optimization_controls(False)
        self.worker = None
        if getattr(result, "aborted", False):
            self.opt_status.setText("Optimization aborted")
            self.opt_status.setStyleSheet("color: #b45309;")
            self.opt_next_btn.setEnabled(False)
            self._update_workflow_cards()
            return
        if not result.success:
            self.opt_status.setText("Optimization failed")
            self.opt_status.setStyleSheet("color: #dc2626;")
            QMessageBox.warning(
                self,
                "Optimization failed",
                "Calibration did not converge. Return to Step 2 and adjust the initial parameters.",
            )
            self._update_workflow_cards()
            return
        self.opt_status.setText("Optimization completed")
        self.opt_status.setStyleSheet("color: palette(highlight);")
        self.loss_label.setText(f"Final Loss: {result.fun:.6f}")
        self.opt_next_btn.setEnabled(True)
        self.latest_optimizer = optimizer
        self.latest_result = result
        self.latest_network = optimizer.solver.energy_function
        self._populate_calibration_params(optimizer.param_names, result.x)
        self._populate_prediction_params(optimizer.param_names, result.x)
        self._refresh_prediction_modes()
        self._plot_calibration_results()
        self._update_workflow_cards()

    def _on_optimization_failed(self, message):
        self._set_optimization_controls(False)
        self.worker = None
        if "aborted" in (message or "").lower():
            self.opt_status.setText("Optimization aborted")
            self.opt_status.setStyleSheet("color: #b45309;")
            self.opt_next_btn.setEnabled(False)
            self._update_workflow_cards()
            return
        self.opt_status.setText("Optimization failed")
        self.opt_status.setStyleSheet("color: #dc2626;")
        QMessageBox.warning(
            self,
            "Optimization failed",
            "Calibration did not converge. Return to Step 2 and adjust the initial parameters.",
        )
        self.opt_next_btn.setEnabled(False)
        self._update_workflow_cards()

    def _on_optimization_progress(self, iteration, loss, params):
        self.opt_status.setText(f"Running optimization... Iter {iteration}")
        self.opt_status.setStyleSheet("color: palette(windowtext);")
        self.loss_label.setText(f"Loss: {loss:.6f}")
        self._update_calibration_params_values(params)
        if hasattr(self, "opt_log"):
            self.opt_log.appendPlainText(f"Iter {iteration}: Loss {loss:.6f}")

    def _on_step_selected(self, index):
        if index < 0:
            return
        self._set_step(index)

    def _on_step_card_clicked(self, index):
        self._set_step(index)

    def _set_step(self, index):
        if index == self.current_step:
            return
        if not self._is_step_unlocked(index):
            return
        self.current_step = index
        step_name = self.step_names[index]
        widget = self.section_widgets.get(step_name)
        if widget:
            self._update_section_visibility()
            if step_name == "Prediction":
                if self.latest_result is not None and self.latest_optimizer is not None:
                    self._populate_prediction_params(self.latest_optimizer.param_names, self.latest_result.x)
                self._refresh_prediction_modes()
            self.scroll.ensureWidgetVisible(widget, 0, 20)
        self._update_workflow_cards()

    def _update_section_visibility(self):
        for idx, step_name in enumerate(self.step_names):
            widget = self.section_widgets.get(step_name)
            if widget:
                widget.setVisible(idx == self.current_step)

    def _is_step_complete(self, index):
        def _btn_enabled(attr_name):
            btn = getattr(self, attr_name, None)
            return bool(btn and btn.isEnabled())

        if index == 0:
            return _btn_enabled("data_next_btn")
        if index == 1:
            return _btn_enabled("model_next_btn")
        if index == 2:
            return self.latest_result is not None
        if index == 3:
            return self.latest_result is not None
        return False

    def _is_step_unlocked(self, index):
        if index in (0, 1):
            return True
        if index == 2:
            return self._is_step_complete(0) and self._is_step_complete(1)
        if index == 3:
            return self._is_step_complete(2)
        return False

    def _update_workflow_cards(self):
        if not hasattr(self, "step_cards"):
            return
        max_unlocked = 0
        for idx in range(len(self.step_cards)):
            if self._is_step_unlocked(idx):
                max_unlocked = idx
            else:
                break
        if self.current_step > max_unlocked:
            self.current_step = max_unlocked
            self._update_section_visibility()
        for idx, card in enumerate(self.step_cards):
            complete = self._is_step_complete(idx)
            unlocked = self._is_step_unlocked(idx)
            if idx == 3:
                if not unlocked:
                    state = "locked"
                    status = "Locked"
                else:
                    state = "ready"
                    status = ""
            elif not unlocked:
                state = "locked"
                status = "Locked"
            elif complete:
                state = "complete"
                status = "Complete"
            else:
                state = "ready"
                status = ""
            card.title_label.setText(f"{idx + 1}. {self.step_names[idx]}")
            card.title_label.setStyleSheet("color: palette(windowtext);")
            card.status_label.setText(status)
            card.set_state(state)
            card.set_locked(not unlocked)
            card.set_current(idx == self.current_step)
            if state == "complete":
                card.status_label.setStyleSheet("color: palette(highlight);")
            elif state == "locked":
                card.status_label.setStyleSheet("color: #475569;")
            else:
                card.status_label.setStyleSheet("color: palette(mid);")
        for idx, connector in enumerate(self.step_connectors):
            if self._is_step_complete(idx):
                connector.setStyleSheet("background: palette(highlight);")
            else:
                connector.setStyleSheet("background: palette(mid);")

    def _on_spring_config_changed(self, changed=False):
        if changed:
            self._reset_results()
        springs = list(getattr(self, "spring_widgets", []))
        if springs and all(spring.is_valid() for spring in springs):
            self.model_next_btn.setEnabled(True)
        else:
            self.model_next_btn.setEnabled(False)
        self._update_workflow_cards()
        self._refresh_opt_params_from_springs()

    def _collect_initial_params(self):
        param_names = []
        values = []
        for idx, spring in enumerate(self.spring_widgets):
            if not spring.is_valid():
                return [], []
            try:
                func, params = spring.build_config()
            except Exception:
                return [], []
            prefix = f"{func.__name__}_{idx + 1}"
            model_params = getattr(func, "param_names", [])
            for name, value in zip(model_params, params):
                param_names.append(f"{prefix}_{name}")
                values.append(value)
        return param_names, values

    def _refresh_opt_params_from_springs(self):
        if not hasattr(self, "calib_params_layout"):
            return
        param_names, values = self._collect_initial_params()
        if not param_names:
            self._clear_calibration_params()
            return
        self._populate_calibration_params(param_names, values)

    def _format_result_param_label(self, name):
        parts = name.split("_")
        for idx, part in enumerate(parts[:-1]):
            if part.isdigit():
                name = "_".join(parts[idx + 1:])
                break

        def italic(text):
            return f"<i>{text}</i>"

        def format_base(base):
            if base.lower() == "mu":
                return "&mu;"
            if base.lower() == "alpha":
                return "&alpha;"
            return base

        match = re.match(r"^([A-Za-z]+)(\d+)$", name)
        if match:
            base, idx = match.groups()
            base_html = format_base(base)
            return f"{italic(base_html)}<sub>{idx}</sub>"
        match = re.match(r"^([A-Za-z]+)_([0-9]+)$", name)
        if match:
            base, idx = match.groups()
            base_html = format_base(base)
            return f"{italic(base_html)}<sub>{idx}</sub>"
        if name.lower() == "mu":
            return italic("&mu;")
        if name.lower() == "alpha":
            return italic("&alpha;")
        return italic(name)

    def _clear_calibration_params(self):
        self.calib_param_edits = []
        self.calib_param_names = []
        while self.calib_params_layout.count():
            item = self.calib_params_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def _populate_calibration_params(self, param_names, values):
        self._clear_calibration_params()
        self.calib_param_names = list(param_names)
        for name, value in zip(param_names, values):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            label = QLabel()
            label.setTextFormat(Qt.RichText)
            label_html = self._format_result_param_label(name)
            label.setText(label_html)
            label.setMinimumWidth(_label_width_for_text(label, label_html, minimum=70, maximum=210))
            label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            edit = QLineEdit(f"{value:.6g}")
            edit.setReadOnly(True)
            edit.setMinimumWidth(120)
            edit.setMinimumHeight(28)
            edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row_layout.addWidget(label)
            row_layout.addWidget(edit, 1)
            self.calib_params_layout.addWidget(row)
            self.calib_param_edits.append(edit)
        self.calib_params_layout.addStretch()

    def _update_calibration_params_values(self, values):
        edits = getattr(self, "calib_param_edits", [])
        if not edits or len(values) != len(edits):
            return
        for edit, value in zip(edits, values):
            try:
                edit.setText(f"{float(value):.6g}")
            except (TypeError, ValueError):
                edit.setText(str(value))

    def _clear_prediction_params(self):
        while self.prediction_params_layout.count():
            item = self.prediction_params_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def _populate_prediction_params(self, param_names, values):
        self._clear_prediction_params()
        for name, value in zip(param_names, values):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            label = QLabel()
            label.setTextFormat(Qt.RichText)
            label_html = self._format_result_param_label(name)
            label.setText(label_html)
            label.setMinimumWidth(_label_width_for_text(label, label_html, minimum=70, maximum=210))
            label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            edit = QLineEdit(f"{value:.6g}")
            edit.setMinimumWidth(140)
            edit.setMinimumHeight(28)
            edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            edit.textEdited.connect(self._on_prediction_param_edited)
            row_layout.addWidget(label)
            row_layout.addWidget(edit, 1)
            self.prediction_params_layout.addWidget(row)
        self.prediction_params_layout.addStretch()

    def _store_prediction_selection(self):
        if not hasattr(self, "prediction_selection"):
            self.prediction_selection = {}
        if not hasattr(self, "prediction_modes_layout") or not hasattr(self, "author_combo"):
            return
        author = self.author_combo.currentText()
        if author == "Select...":
            return
        selected = set()
        for i in range(self.prediction_modes_layout.count()):
            widget = self.prediction_modes_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox) and widget.isChecked():
                mode_key = widget.property("mode_key") or widget.text()
                selected.add(str(mode_key))
        self.prediction_selection[author] = selected

    def _clear_prediction_selection(self):
        if not hasattr(self, "prediction_selection"):
            self.prediction_selection = {}
        author = self.author_combo.currentText() if hasattr(self, "author_combo") else None
        if author and author != "Select...":
            self.prediction_selection[author] = set()
        if hasattr(self, "prediction_modes_layout"):
            for i in range(self.prediction_modes_layout.count()):
                widget = self.prediction_modes_layout.itemAt(i).widget()
                if isinstance(widget, QCheckBox):
                    widget.blockSignals(True)
                    widget.setChecked(False)
                    widget.blockSignals(False)

    def _refresh_prediction_modes(self):
        self._store_prediction_selection()
        for i in reversed(range(self.prediction_modes_layout.count())):
            widget = self.prediction_modes_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        if hasattr(self, "prediction_component_row"):
            self.prediction_component_row.setVisible(False)
        if hasattr(self, "prediction_component_combo"):
            self.prediction_component_combo.blockSignals(True)
            self.prediction_component_combo.clear()
            self.prediction_component_combo.blockSignals(False)
        author = self.author_combo.currentText()
        if hasattr(self, "prediction_author_label"):
            if author == "Select...":
                self.prediction_author_label.setText("Author: -")
            else:
                self.prediction_author_label.setText(f"Author: {author}")
        if author == "Select...":
            return
        available = self.datasets.get(author, [])
        saved = self.prediction_selection.get(author, set())
        component_config = COMPONENT_AUTHOR_CONFIG.get(author)
        if component_config:
            prefixes = component_config.get("mode_prefixes", ())
            component_modes = [m for m in available if extract_component_from_mode(m, prefixes)]
            if component_modes:
                self.prediction_component_label.setText(component_config.get("component_label", "Component"))
                self.prediction_component_combo.blockSignals(True)
                self.prediction_component_combo.clear()
                component_keys = []
                for mode in component_modes:
                    component = extract_component_from_mode(mode, prefixes)
                    if component:
                        component_keys.append(component)
                component_keys = sorted(set(component_keys))
                for component in component_keys:
                    self.prediction_component_combo.addItem(format_component_label(component), component)
                selected_mode = next((m for m in component_modes if m in saved), component_modes[0])
                selected_component = extract_component_from_mode(selected_mode, prefixes)
                if selected_component in component_keys:
                    self.prediction_component_combo.setCurrentIndex(component_keys.index(selected_component))
                self.prediction_component_combo.blockSignals(False)
                self.prediction_component_row.setVisible(True)
                if selected_component:
                    self._build_prediction_component_modes(author, selected_component, saved)
                    return
        for idx, mode in enumerate(available):
            checkbox = QCheckBox(format_mode_label(mode))
            checkbox.setObjectName("modeOption")
            checkbox.setMinimumHeight(26)
            checkbox.setProperty("mode_key", mode)
            checkbox.stateChanged.connect(self._on_prediction_mode_toggled)
            if mode in saved:
                checkbox.setChecked(True)
            self.prediction_modes_layout.addWidget(checkbox)
        self._apply_prediction_mode_rules()

    def _on_prediction_mode_toggled(self, _state):
        if hasattr(self, "pred_save_btn"):
            self.pred_save_btn.setEnabled(False)
        self._apply_prediction_mode_rules()

    def _apply_prediction_mode_rules(self):
        author = ""
        if hasattr(self, "author_combo"):
            author = self.author_combo.currentText()
        allow_mixed_bt = author == "Katashima_2012"
        bt_selected = False
        non_bt_selected = False
        checkboxes = []
        for i in range(self.prediction_modes_layout.count()):
            widget = self.prediction_modes_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                mode_key = widget.property("mode_key") or ""
                is_bt = str(mode_key).startswith("BT")
                if widget.isChecked():
                    if is_bt:
                        bt_selected = True
                    else:
                        non_bt_selected = True
                checkboxes.append((widget, is_bt))

        if allow_mixed_bt:
            for checkbox, _is_bt in checkboxes:
                checkbox.setEnabled(True)
            return

        for checkbox, is_bt in checkboxes:
            if bt_selected and not is_bt and not checkbox.isChecked():
                checkbox.setEnabled(False)
            elif non_bt_selected and is_bt and not checkbox.isChecked():
                checkbox.setEnabled(False)
            else:
                checkbox.setEnabled(True)

    def _on_prediction_component_changed(self, _index):
        if not hasattr(self, "prediction_component_combo") or not hasattr(self, "prediction_modes_layout"):
            return
        author = self.author_combo.currentText()
        component_key = self.prediction_component_combo.currentData()
        if not author or author == "Select..." or not component_key:
            return
        saved = self.prediction_selection.get(author, set()) if hasattr(self, "prediction_selection") else set()
        self._build_prediction_component_modes(author, component_key, saved)
        if hasattr(self, "pred_save_btn"):
            self.pred_save_btn.setEnabled(False)
        self._apply_prediction_mode_rules()

    def _build_prediction_component_modes(self, author, component_key, saved):
        for i in reversed(range(self.prediction_modes_layout.count())):
            widget = self.prediction_modes_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        config = COMPONENT_AUTHOR_CONFIG.get(author)
        if not config:
            return
        prefixes = config.get("mode_prefixes", ())
        mode_labels = config.get("mode_labels", {})
        available = set(self.datasets.get(author, []))
        row = 0
        for prefix in prefixes:
            if prefix == "CSS_":
                compound_modes = [
                    m for m in available
                    if m.startswith(prefix) and extract_component_from_mode(m, (prefix,)) == component_key
                ]
                for mode_key in sorted(compound_modes):
                    label = format_mode_label(mode_key)
                    checkbox = QCheckBox(label)
                    checkbox.setObjectName("modeOption")
                    checkbox.setMinimumHeight(26)
                    checkbox.setProperty("mode_key", mode_key)
                    checkbox.stateChanged.connect(self._on_prediction_mode_toggled)
                    if mode_key in saved:
                        checkbox.setChecked(True)
                    self.prediction_modes_layout.addWidget(checkbox)
                    row += 1
                continue
            mode_key = f"{prefix}{component_key}"
            if mode_key not in available:
                continue
            label = mode_labels.get(prefix, format_mode_label(mode_key))
            checkbox = QCheckBox(label)
            checkbox.setObjectName("modeOption")
            checkbox.setMinimumHeight(26)
            checkbox.setProperty("mode_key", mode_key)
            checkbox.stateChanged.connect(self._on_prediction_mode_toggled)
            if mode_key in saved:
                checkbox.setChecked(True)
            self.prediction_modes_layout.addWidget(checkbox)
            row += 1

    def _get_prediction_modes(self):
        modes = []
        for i in range(self.prediction_modes_layout.count()):
            widget = self.prediction_modes_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox) and widget.isChecked():
                mode_key = widget.property("mode_key") or widget.text()
                modes.append(str(mode_key))
        return modes
    def _get_prediction_param_values(self):
        values = []
        for i in range(self.prediction_params_layout.count()):
            item = self.prediction_params_layout.itemAt(i)
            row = item.widget()
            if not row:
                continue
            edits = row.findChildren(QLineEdit)
            if not edits:
                continue
            text = edits[0].text().strip()
            try:
                values.append(float(text))
            except ValueError:
                values.append(0.0)
        return values

    def _plot_calibration_results(self):
        if not self.latest_optimizer or not self.latest_result:
            return
        optimizer = self.latest_optimizer
        plot_params = dict(zip(optimizer.param_names, self.latest_result.x))

        if hasattr(self, "calib_dataset_label"):
            dataset_text = self._format_calibration_dataset_label()
            self.calib_dataset_label.setText(dataset_text)
            self.calib_dataset_label.setToolTip(dataset_text)

        self.calib_canvas.reset_axes()

        colors_calib = {"UT": "#2980b9", "ET": "#c0392b", "PS": "#27ae60", "SS": "#16a085", "CSS": "#16a085", "BT": "#8e44ad"}
        calib_data = optimizer.data
        bt_only = self._plot_dataset(self.calib_canvas.ax, calib_data, optimizer.solver, plot_params, colors_calib, "Exp", "Fit")

        if not bt_only:
            non_bt_modes = [d["mode"] for d in calib_data if d["mode"] != "BT"]
            self.calib_canvas.ax.set_xlabel(choose_xlabel_for_modes(non_bt_modes))
            stress_types = {d.get("stress_type", "PK1") for d in calib_data if d["mode"] != "BT"}
            has_multi = any(
                d["mode"] != "BT" and (np.ndim(d.get("stress_exp")) > 1 or d.get("component"))
                for d in calib_data
            )
            if non_bt_modes and all(m in ("SS", "CSS") for m in non_bt_modes) and not has_multi:
                if len(stress_types) == 1:
                    y_label = get_shear_component_label(next(iter(stress_types)))
                else:
                    y_label = r"$P_{12}$"
            elif has_multi:
                if len(stress_types) == 1:
                    stress_type = next(iter(stress_types))
                    y_label = r"$\sigma$" if stress_type == "cauchy" else r"$P$"
                else:
                    y_label = r"$P$"
            elif len(stress_types) == 1:
                y_label = get_uniaxial_component_label(next(iter(stress_types)))
            else:
                y_label = r"$P_{11}$"
            self.calib_canvas.ax.set_ylabel(y_label)
            self.calib_canvas.ax.legend(fontsize=7)
            self.calib_canvas.figure.tight_layout()
        r2_total = getattr(self.latest_result, "r2_total", None)
        if r2_total is None and hasattr(optimizer, "compute_r2"):
            try:
                r2_total, _ = optimizer.compute_r2(self.latest_result.x)
                self.latest_result.r2_total = r2_total
            except Exception:
                r2_total = None
        if r2_total is not None and np.isfinite(r2_total):
            text_color = QApplication.palette().color(QPalette.WindowText)
            color = (text_color.redF(), text_color.greenF(), text_color.blueF(), 1.0)
            self.calib_canvas.figure.text(
                0.98,
                0.02,
                f"$R^2 = {r2_total * 100:.1f}\\%$",
                ha="right",
                va="bottom",
                fontsize=9,
                color=color,
            )
        if hasattr(self, "calib_save_btn"):
            self.calib_save_btn.setEnabled(True)
        self.calib_canvas.draw()

    def _reset_results(self):
        if self._optimization_running:
            self._abort_optimization(wait=False)
            self._set_optimization_controls(False)
        self.latest_optimizer = None
        self.latest_result = None
        self.latest_network = None
        self.opt_status.setText("")
        self.opt_status.setStyleSheet("color: palette(windowtext);")
        self.loss_label.setText("Loss: -")
        self.opt_next_btn.setEnabled(False)
        self._clear_prediction_selection()
        if hasattr(self, "calib_save_btn"):
            self.calib_save_btn.setEnabled(False)
        if hasattr(self, "calib_dataset_label"):
            self.calib_dataset_label.setText("Dataset: -")
            self.calib_dataset_label.setToolTip("Dataset: -")
        if hasattr(self, "opt_log"):
            self.opt_log.clear()
        self.calib_canvas.reset_axes()
        self.calib_canvas.draw()
        self.prediction_canvas.reset_axes()
        self.prediction_canvas.draw()
        if hasattr(self, "pred_save_btn"):
            self.pred_save_btn.setEnabled(False)
        self._clear_calibration_params()
        self._clear_prediction_params()
        self._refresh_opt_params_from_springs()
        self._update_workflow_cards()

    def _reset_prediction_results(self):
        if hasattr(self, "prediction_canvas"):
            self.prediction_canvas.reset_axes()
            self.prediction_canvas.draw()
        if hasattr(self, "pred_save_btn"):
            self.pred_save_btn.setEnabled(False)
        self._clear_prediction_params()

    def _invalidate_prediction_plot(self):
        if hasattr(self, "prediction_canvas"):
            self.prediction_canvas.reset_axes()
            self.prediction_canvas.draw()
        if hasattr(self, "pred_save_btn"):
            self.pred_save_btn.setEnabled(False)

    def _on_prediction_param_edited(self, _text):
        self._invalidate_prediction_plot()

    def _update_prediction_plot(self):
        if not self.latest_optimizer or not self.latest_result:
            if hasattr(self, "pred_save_btn"):
                self.pred_save_btn.setEnabled(False)
            return
        optimizer = self.latest_optimizer
        param_values = self._get_prediction_param_values()
        if len(param_values) != len(optimizer.param_names):
            param_values = list(self.latest_result.x)
        plot_params = dict(zip(optimizer.param_names, param_values))

        self.prediction_canvas.reset_axes()

        colors_pred = {"UT": "#3498db", "ET": "#e74c3c", "PS": "#2ecc71", "SS": "#1abc9c", "CSS": "#1abc9c", "BT": "#a569bd"}
        pred_modes = self._get_prediction_modes()
        if pred_modes:
            author = self.author_combo.currentText()
            pred_configs = [{"author": author, "mode": m} for m in pred_modes]
            pred_data = load_experimental_data(pred_configs)
            bt_only = self._plot_dataset(self.prediction_canvas.ax, pred_data, optimizer.solver, plot_params, colors_pred, "Pred", "PredFit")
        else:
            bt_only = False

        if not bt_only:
            if pred_modes:
                non_bt_modes = [d["mode"] for d in pred_data if d["mode"] != "BT"]
                self.prediction_canvas.ax.set_xlabel(choose_xlabel_for_modes(non_bt_modes))
                stress_types = {d.get("stress_type", "PK1") for d in pred_data if d["mode"] != "BT"}
                has_multi = any(
                    d["mode"] != "BT" and (np.ndim(d.get("stress_exp")) > 1 or d.get("component"))
                    for d in pred_data
                )
                if non_bt_modes and all(m in ("SS", "CSS") for m in non_bt_modes) and not has_multi:
                    if len(stress_types) == 1:
                        y_label = get_shear_component_label(next(iter(stress_types)))
                    else:
                        y_label = r"$P_{12}$"
                elif has_multi:
                    if len(stress_types) == 1:
                        stress_type = next(iter(stress_types))
                        y_label = r"$\sigma$" if stress_type == "cauchy" else r"$P$"
                    else:
                        y_label = r"$P$"
                elif len(stress_types) == 1:
                    y_label = get_uniaxial_component_label(next(iter(stress_types)))
                else:
                    y_label = r"$P_{11}$"
                self.prediction_canvas.ax.set_ylabel(y_label)
            else:
                self.prediction_canvas.ax.set_xlabel(r"$\lambda_1$")
                self.prediction_canvas.ax.set_ylabel(r"$P_{11}$")
            self.prediction_canvas.ax.legend(fontsize=7)
            self.prediction_canvas.figure.tight_layout()
        self.prediction_canvas.draw()
        if hasattr(self, "pred_save_btn"):
            self.pred_save_btn.setEnabled(bool(pred_modes))

    def _plot_bt_preview(self, canvas, data):
        use_diff = all(d.get("bt_component") == "diff" for d in data)
        use_component = any(d.get("component") for d in data)
        fig = canvas.figure
        fig.clear()
        if use_diff:
            ax = fig.add_subplot(111)
            canvas.set_axes([ax])
            canvas.apply_theme()
            markers = ["o", "s", "v", "^", "D", "P", "X"]
            colors = ["#000000", "#7f8c8d", "#2980b9", "#c0392b", "#27ae60", "#8e44ad", "#d35400"]
            for idx, d in enumerate(data):
                stress = d["stress_exp"]
                if np.ndim(stress) == 1:
                    y = stress
                else:
                    y = stress[:, 0] - stress[:, 1]
                lam2 = parse_lambda2(d.get("mode_raw", ""))
                label = f"λ₂={lam2}" if lam2 else "BT"
                marker = markers[idx % len(markers)]
                color = colors[idx % len(colors)]
                ax.plot(d["stretch"], y, marker, color=color, label=label, linestyle="None")
            ax.set_xlabel(r"$\lambda_1$")
            diff_label = get_bt_diff_label(data[0].get("stress_type", "PK1"))
            ax.set_ylabel(diff_label)
            ax.legend(fontsize=8)
            fig.tight_layout()
            return

        if use_component:
            ax = fig.add_subplot(111)
            canvas.set_axes([ax])
            canvas.apply_theme()
            stress_type = data[0].get("stress_type", "PK1")
            markers = ["s", "v", "^", "D", "o", "P", "X"]
            colors = ["#7f8c8d", "#2980b9", "#c0392b", "#27ae60", "#8e44ad", "#d35400", "#2c3e50"]
            for idx, d in enumerate(data):
                component = d.get("component")
                comp_label = get_component_label(stress_type, component)
                marker = markers[idx % len(markers)]
                color = colors[idx % len(colors)]
                ax.plot(d["stretch"], d["stress_exp"], marker, color=color, label=comp_label, linestyle="None")
            ax.set_xlabel(r"$\lambda_1$")
            ax.set_ylabel(r"$\sigma$" if stress_type == "cauchy" else r"$P$")
            ax.legend(fontsize=8)
            fig.tight_layout()
            return

        stress_type = data[0].get("stress_type", "PK1")
        comp_11, comp_22 = get_bt_component_labels(stress_type)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)
        canvas.set_axes([ax1, ax2])
        canvas.apply_theme()
        markers = ["s", "v", "^", "D", "o", "P", "X"]
        colors = ["#7f8c8d", "#2980b9", "#c0392b", "#27ae60", "#8e44ad", "#d35400", "#2c3e50"]
        for idx, d in enumerate(data):
            stress = d["stress_exp"]
            lam2 = parse_lambda2(d.get("mode_raw", ""))
            label = f"λ₂={lam2}" if lam2 else "BT"
            marker = markers[idx % len(markers)]
            color = colors[idx % len(colors)]
            if np.ndim(stress) == 1:
                ax1.plot(d["stretch"], stress, marker, color=color, label=label, linestyle="None")
            else:
                ax1.plot(d["stretch"], stress[:, 0], marker, color=color, label=label, linestyle="None")
                ax2.plot(d["stretch"], stress[:, 1], marker, color=color, label=label, linestyle="None")
        ax1.set_ylabel(comp_11)
        ax2.set_ylabel(comp_22)
        ax2.set_xlabel(r"$\lambda_1$")
        ax1.legend(fontsize=8)
        ax2.legend(fontsize=8)
        fig.tight_layout()

    def _plot_bt_dataset(self, canvas, data, solver, params, colors, exp_label, fit_label):
        use_diff = all(d.get("bt_component") == "diff" for d in data)
        use_component = any(d.get("component") for d in data)
        use_single_axis = all(d.get("author") == "Katashima_2012" for d in data)
        fig = canvas.figure
        fig.clear()
        if use_diff:
            ax = fig.add_subplot(111)
            canvas.set_axes([ax])
            canvas.apply_theme()
            markers = ["o", "s", "v", "^", "D", "P", "X"]
            colors_seq = ["#000000", "#7f8c8d", "#2980b9", "#c0392b", "#27ae60", "#8e44ad", "#d35400"]
            for idx, d in enumerate(data):
                stress = d["stress_exp"]
                if np.ndim(stress) == 1:
                    y = stress
                else:
                    y = stress[:, 0] - stress[:, 1]
                lam2 = parse_lambda2(d.get("mode_raw", ""))
                label = f"λ₂={lam2}" if lam2 else "BT"
                marker = markers[idx % len(markers)]
                color = colors_seq[idx % len(colors_seq)]
                ax.plot(d["stretch"], y, marker, color=color, label=label, linestyle="None")
                smooth = np.linspace(min(d["stretch"]), max(d["stretch"]), 120)
                model = []
                for lam in smooth:
                    lam2_val = float(d["stretch_secondary"][0])
                    F = get_deformation_gradient((lam, lam2_val), "BT")
                    stress_type = d.get("stress_type", "PK1")
                    if stress_type == "cauchy":
                        stress_tensor = solver.get_Cauchy_stress(F, params)
                    else:
                        stress_tensor = solver.get_1st_PK_stress(F, params)
                    comps = get_stress_components(stress_tensor, "BT")
                    model.append(comps[0] - comps[1])
                ax.plot(smooth, model, "-", color=color, label="_nolegend_")
            ax.set_xlabel(r"$\lambda_1$")
            diff_label = get_bt_diff_label(data[0].get("stress_type", "PK1"))
            ax.set_ylabel(diff_label)
            ax.legend(fontsize=8)
            fig.tight_layout()
            return
        if use_component:
            ax = fig.add_subplot(111)
            canvas.set_axes([ax])
            canvas.apply_theme()
            stress_type = data[0].get("stress_type", "PK1")
            markers = ["s", "v", "^", "D", "o", "P", "X"]
            colors_seq = ["#7f8c8d", "#2980b9", "#c0392b", "#27ae60", "#8e44ad", "#d35400", "#2c3e50"]
            for idx, d in enumerate(data):
                component = d.get("component")
                comp_label = get_component_label(stress_type, component)
                marker = markers[idx % len(markers)]
                color = colors_seq[idx % len(colors_seq)]
                ax.plot(d["stretch"], d["stress_exp"], marker, color=color, label=f"{exp_label} {comp_label}", linestyle="None")
                model = []
                for F in d["F_list"]:
                    if stress_type == "cauchy":
                        stress_tensor = solver.get_Cauchy_stress(F, params)
                    else:
                        stress_tensor = solver.get_1st_PK_stress(F, params)
                    comps = get_stress_components(stress_tensor, "BT")
                    model.append(comps[1] if component == "22" else comps[0])
                ax.plot(d["stretch"], model, "-", color=color, label=f"{fit_label} {comp_label}")
            ax.set_xlabel(r"$\lambda_1$")
            ax.set_ylabel(r"$\sigma$" if stress_type == "cauchy" else r"$P$")
            ax.legend(fontsize=8)
            fig.tight_layout()
            return
        if use_single_axis:
            ax = fig.add_subplot(111)
            canvas.set_axes([ax])
            canvas.apply_theme()
            stress_type = data[0].get("stress_type", "PK1")
            comp_11, comp_22 = get_bt_component_labels(stress_type)
            markers = ["s", "v", "^", "D", "o", "P", "X"]
            colors_seq = ["#7f8c8d", "#2980b9", "#c0392b", "#27ae60", "#8e44ad", "#d35400", "#2c3e50"]
            for idx, d in enumerate(data):
                stress = d["stress_exp"]
                lam2 = parse_lambda2(d.get("mode_raw", ""))
                label = f"λ₂={lam2}" if lam2 else "BT"
                marker = markers[idx % len(markers)]
                color = colors_seq[idx % len(colors_seq)]
                if np.ndim(stress) == 1:
                    ax.plot(d["stretch"], stress, marker, color=color, label=f"{exp_label} {label} {comp_11}", linestyle="None")
                else:
                    ax.plot(d["stretch"], stress[:, 0], marker, color=color, label=f"{exp_label} {label} {comp_11}", linestyle="None")
                    ax.plot(d["stretch"], stress[:, 1], "^", color=color, label=f"{exp_label} {label} {comp_22}", linestyle="None")

                model_11 = []
                model_22 = []
                for F in d["F_list"]:
                    if stress_type == "cauchy":
                        stress_tensor = solver.get_Cauchy_stress(F, params)
                    else:
                        stress_tensor = solver.get_1st_PK_stress(F, params)
                    comps = get_stress_components(stress_tensor, "BT")
                    model_11.append(comps[0])
                    if len(comps) > 1:
                        model_22.append(comps[1])
                ax.plot(d["stretch"], model_11, "-", color=color, label=f"{fit_label} {label} {comp_11}")
                if model_22 and np.ndim(stress) > 1:
                    ax.plot(d["stretch"], model_22, "--", color=color, label=f"{fit_label} {label} {comp_22}")

            ax.set_xlabel(r"$\lambda_1$")
            ax.set_ylabel(r"$\sigma$" if stress_type == "cauchy" else r"$P$")
            ax.legend(fontsize=8)
            fig.tight_layout()
            return

        stress_type = data[0].get("stress_type", "PK1")
        comp_11, comp_22 = get_bt_component_labels(stress_type)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)
        canvas.set_axes([ax1, ax2])
        canvas.apply_theme()
        markers = ["s", "v", "^", "D", "o", "P", "X"]
        colors_seq = ["#7f8c8d", "#2980b9", "#c0392b", "#27ae60", "#8e44ad", "#d35400", "#2c3e50"]
        for idx, d in enumerate(data):
            stress = d["stress_exp"]
            lam2 = parse_lambda2(d.get("mode_raw", ""))
            label = f"λ₂={lam2}" if lam2 else "BT"
            marker = markers[idx % len(markers)]
            color = colors_seq[idx % len(colors_seq)]
            if np.ndim(stress) == 1:
                ax1.plot(d["stretch"], stress, marker, color=color, label=label, linestyle="None")
            else:
                ax1.plot(d["stretch"], stress[:, 0], marker, color=color, label=label, linestyle="None")
                ax2.plot(d["stretch"], stress[:, 1], marker, color=color, label=label, linestyle="None")

            smooth = np.linspace(min(d["stretch"]), max(d["stretch"]), 120)
            model_11 = []
            model_22 = []
            for lam in smooth:
                lam2_val = float(d["stretch_secondary"][0])
                F = get_deformation_gradient((lam, lam2_val), "BT")
                if stress_type == "cauchy":
                    stress_tensor = solver.get_Cauchy_stress(F, params)
                else:
                    stress_tensor = solver.get_1st_PK_stress(F, params)
                comps = get_stress_components(stress_tensor, "BT")
                model_11.append(comps[0])
                model_22.append(comps[1])
            ax1.plot(smooth, model_11, "-", color=color, label="_nolegend_")
            ax2.plot(smooth, model_22, "-", color=color, label="_nolegend_")

        ax1.set_ylabel(comp_11)
        ax2.set_ylabel(comp_22)
        ax2.set_xlabel(r"$\lambda_1$")
        ax1.legend(fontsize=8)
        ax2.legend(fontsize=8)
        fig.tight_layout()

    def _plot_dataset(self, ax, data, solver, params, colors, exp_label, fit_label):
        if data and all(d["mode"] == "BT" for d in data):
            canvas = ax.figure.canvas
            if isinstance(canvas, MatplotlibCanvas):
                self._plot_bt_dataset(canvas, data, solver, params, colors, exp_label, fit_label)
                return True
        css_palette = ["#1abc9c", "#2980b9", "#c0392b", "#8e44ad", "#d35400", "#2c3e50"]
        css_index = 0
        for d in data:
            mode = d["mode"]
            stress_type = d.get("stress_type", "PK1")
            stretch = d["stretch"]
            stress = d["stress_exp"]
            label = d.get("label") or format_mode_label(d.get("mode_raw", mode))
            component = d.get("component")
            has_two = np.ndim(stress) > 1 and component is None

            color = colors.get(mode, "black")
            if mode == "CSS":
                color = css_palette[css_index % len(css_palette)]
                css_index += 1
            if mode == "BT":
                bt_diff = d.get("bt_component") == "diff"
                if bt_diff:
                    diff_label = get_bt_diff_label(stress_type)
                    if np.ndim(stress) == 1:
                        exp_values = stress
                    else:
                        exp_values = stress[:, 0] - stress[:, 1]
                    ax.plot(stretch, exp_values, "o", color=color, label=f"{exp_label} {label} {diff_label}")
                    smooth = np.linspace(min(stretch), max(stretch), 120)
                    model = []
                    for lam in smooth:
                        lam2 = float(d["stretch_secondary"][0])
                        F = get_deformation_gradient((lam, lam2), mode)
                        if stress_type == "cauchy":
                            stress_tensor = solver.get_Cauchy_stress(F, params)
                        else:
                            stress_tensor = solver.get_1st_PK_stress(F, params)
                        comps = get_stress_components(stress_tensor, mode)
                        model.append(comps[0] - comps[1])
                    ax.plot(smooth, model, "-", color=color, label=f"{fit_label} {label} {diff_label}")
                    continue
                comp_11, comp_22 = get_bt_component_labels(stress_type)
                if component:
                    comp_label = get_component_label(stress_type, component)
                    ax.plot(stretch, stress, "o", color=color, label=f"{exp_label} {label} {comp_label}")
                    model = []
                    for F in d["F_list"]:
                        if stress_type == "cauchy":
                            stress_tensor = solver.get_Cauchy_stress(F, params)
                        else:
                            stress_tensor = solver.get_1st_PK_stress(F, params)
                        comps = get_stress_components(stress_tensor, mode)
                        model.append(comps[1] if component == "22" else comps[0])
                    ax.plot(stretch, model, "-", color=color, label=f"{fit_label} {label} {comp_label}")
                    continue
                if np.ndim(stress) == 1:
                    ax.plot(stretch, stress, "o", color=color, label=f"{exp_label} {label} {comp_11}")
                else:
                    ax.plot(stretch, stress[:, 0], "o", color=color, label=f"{exp_label} {label} {comp_11}")
                    ax.plot(stretch, stress[:, 1], "^", color=color, label=f"{exp_label} {label} {comp_22}")
                bt_variable = False
                lam2_vals = d.get("stretch_secondary")
                if lam2_vals is not None:
                    try:
                        bt_variable = np.ptp(lam2_vals) > 1e-6
                    except Exception:
                        bt_variable = False
                if d.get("author") == "Katashima_2012":
                    bt_variable = True
                if bt_variable:
                    model = []
                    model2 = []
                    for F in d["F_list"]:
                        if stress_type == "cauchy":
                            stress_tensor = solver.get_Cauchy_stress(F, params)
                        else:
                            stress_tensor = solver.get_1st_PK_stress(F, params)
                        comps = get_stress_components(stress_tensor, mode)
                        model.append(comps[0])
                        if len(comps) > 1:
                            model2.append(comps[1])
                    ax.plot(stretch, model, "-", color=color, label=f"{fit_label} {label} {comp_11}")
                    if model2 and np.ndim(stress) > 1:
                        ax.plot(stretch, model2, "--", color=color, label=f"{fit_label} {label} {comp_22}")
                    continue
            else:
                if mode in ("SS", "CSS"):
                    shear_label = get_shear_component_label(stress_type)
                    ax.plot(stretch, stress, "o", color=color, label=f"{exp_label} {label} {shear_label}")
                elif component:
                    comp_label = get_component_label(stress_type, component)
                    ax.plot(stretch, stress, "o", color=color, label=f"{exp_label} {label} {comp_label}")
                elif has_two:
                    comp_11, comp_22 = get_bt_component_labels(stress_type)
                    ax.plot(stretch, stress[:, 0], "o", color=color, label=f"{exp_label} {label} {comp_11}")
                    ax.plot(stretch, stress[:, 1], "^", color=color, label=f"{exp_label} {label} {comp_22}")
                else:
                    ax.plot(stretch, stress, "o", color=color, label=f"{exp_label} {label}")
                stretch_secondary = d.get("stretch_secondary")
                if stretch_secondary is not None or component or mode in ("SS", "CSS"):
                    model = []
                    model2 = []
                    for F in d["F_list"]:
                        if stress_type == "cauchy":
                            stress_tensor = solver.get_Cauchy_stress(F, params)
                        else:
                            stress_tensor = solver.get_1st_PK_stress(F, params)
                        comps = get_stress_components(stress_tensor, mode)
                        if component == "22":
                            model.append(comps[1])
                        else:
                            model.append(comps[0])
                        if has_two and len(comps) > 1:
                            model2.append(comps[1])
                    if component:
                        comp_label = get_component_label(stress_type, component)
                        ax.plot(stretch, model, "-", color=color, label=f"{fit_label} {label} {comp_label}")
                    elif mode in ("SS", "CSS"):
                        shear_label = get_shear_component_label(stress_type)
                        ax.plot(stretch, model, "-", color=color, label=f"{fit_label} {label} {shear_label}")
                    elif has_two:
                        comp_11, comp_22 = get_bt_component_labels(stress_type)
                        ax.plot(stretch, model, "-", color=color, label=f"{fit_label} {label} {comp_11}")
                        if model2:
                            ax.plot(stretch, model2, "--", color=color, label=f"{fit_label} {label} {comp_22}")
                    else:
                        ax.plot(stretch, model, "-", color=color, label=f"{fit_label} {label}")
                    continue

            smooth = np.linspace(min(stretch), max(stretch), 120)
            model = []
            model2 = []
            for lam in smooth:
                if mode == "BT":
                    lam2 = float(d["stretch_secondary"][0])
                    F = get_deformation_gradient((lam, lam2), mode)
                else:
                    F = get_deformation_gradient(lam, mode)
                if stress_type == "cauchy":
                    stress_tensor = solver.get_Cauchy_stress(F, params)
                else:
                    stress_tensor = solver.get_1st_PK_stress(F, params)
                comps = get_stress_components(stress_tensor, mode)
                if component == "22":
                    model.append(comps[1])
                else:
                    model.append(comps[0])
                if (mode == "BT" or has_two) and len(comps) > 1:
                    model2.append(comps[1])
            if mode == "BT":
                ax.plot(smooth, model, "-", color=color, label=f"{fit_label} {label} {comp_11}")
            else:
                if component:
                    comp_label = get_component_label(stress_type, component)
                    ax.plot(smooth, model, "-", color=color, label=f"{fit_label} {label} {comp_label}")
                elif mode in ("SS", "CSS"):
                    shear_label = get_shear_component_label(stress_type)
                    ax.plot(smooth, model, "-", color=color, label=f"{fit_label} {label} {shear_label}")
                elif has_two:
                    comp_11, comp_22 = get_bt_component_labels(stress_type)
                    ax.plot(smooth, model, "-", color=color, label=f"{fit_label} {label} {comp_11}")
                else:
                    ax.plot(smooth, model, "-", color=color, label=f"{fit_label} {label}")
            if model2:
                ax.plot(smooth, model2, "--", color=color, label=f"{fit_label} {label} {comp_22}")
        return False

    def closeEvent(self, event):
        self._abort_optimization(wait=True)
        for canvas_name in ("preview_canvas", "calib_canvas", "prediction_canvas"):
            canvas = getattr(self, canvas_name, None)
            if canvas:
                canvas.teardown()
        gc.collect()
        super().closeEvent(event)


def main():
    if multiprocessing.current_process().name != "MainProcess":
        return
    app = QApplication(sys.argv)
    app.setPalette(build_app_palette())
    app.setStyleSheet(build_app_stylesheet())
    app.setApplicationName("Calibration for Hyperelasticity")
    app.setApplicationDisplayName("Calibration for Hyperelasticity")
    app_icon_path = os.path.join(base_dir, "assets", "icons", "app.png")
    if os.path.exists(app_icon_path):
        app.setWindowIcon(QIcon(app_icon_path))
    app.setFont(select_app_font())
    progress = QProgressDialog("Loading libraries...", None, 0, 5)
    progress.setWindowTitle("Starting Calibration for Hyperelasticity")
    progress.setMinimumDuration(0)
    progress.setCancelButton(None)
    progress.setValue(0)
    progress.show()
    QApplication.processEvents()
    steps = [
        "Loading NumPy",
        "Loading Matplotlib",
        "Loading PySide6",
        "Loading datasets",
        "Finalizing",
    ]
    for idx, message in enumerate(steps, start=1):
        progress.setLabelText(message)
        progress.setValue(idx)
        QApplication.processEvents()
    window = MainWindow()
    window.show()
    progress.close()
    sys.exit(app.exec())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
