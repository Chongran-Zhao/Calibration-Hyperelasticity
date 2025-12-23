import os
import sys
import tempfile
import re
import warnings
import multiprocessing
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

cache_root = Path.home() / "Library" / "Caches" / "HyperelasticCalibration"
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
    "BT": "Biaxial Tension",
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

MODEL_REFERENCES = {
    "NeoHookean": ("Treloar 1943, Rubber Chemistry and Technology", "https://doi.org/10.5254/1.3540158"),
    "MooneyRivlin": ("Mooney 1940, Journal of Applied Physics", "https://doi.org/10.1063/1.1712836"),
    "Yeoh": ("Yeoh 1993, Rubber Chemistry and Technology", "https://doi.org/10.5254/1.3538343"),
    "ArrudaBoyce": ("Arruda & Boyce 1993, J. Mech. Phys. Solids", "https://doi.org/10.1016/0022-5096(93)90013-6"),
    "Ogden": ("Ogden 1972, Proc. R. Soc. A", "https://doi.org/10.1098/rspa.1972.0026"),
    "Hill": ("Seth-Hill generalized strains 2005, Int. J. Numer. Meth. Eng.", "https://doi.org/10.1002/cnm.752"),
}


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
    return MODE_DISPLAY_MAP.get(mode_key, mode_key)


def get_stress_type_label(stress_type):
    return "Cauchy stress" if stress_type == "cauchy" else "Nominal stress"

def get_bt_component_labels(stress_type):
    if stress_type == "cauchy":
        return r"$\sigma_{11}$", r"$\sigma_{22}$"
    return r"$P_{11}$", r"$P_{22}$"

def get_uniaxial_component_label(stress_type):
    return r"$\sigma_{11}$" if stress_type == "cauchy" else r"$P_{11}$"

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


def get_model_list():
    models = []
    for attr_name in dir(MaterialModels):
        attr = getattr(MaterialModels, attr_name)
        if hasattr(attr, "model_type") and hasattr(attr, "category"):
            if hasattr(attr, "param_names") and attr.param_names:
                models.append(attr_name)
    models.append("Hill")
    return sorted(models)


class LatexLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._latex_image_data = None
        self.setMinimumHeight(48)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet("background: transparent;")
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def set_latex(self, latex):
        if not latex:
            self.clear()
            self._latex_image_data = None
            return
        fig = Figure(figsize=(5.2, 0.7), dpi=150)
        fig.patch.set_alpha(0.0)
        fig.patch.set_facecolor("none")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.set_facecolor("none")
        ax.patch.set_alpha(0.0)
        text_color = QApplication.palette().color(QPalette.WindowText)
        color = (text_color.redF(), text_color.greenF(), text_color.blueF(), 1.0)
        ax.text(0.0, 0.5, f"${latex}$", fontsize=12, va="center", ha="left", color=color)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = canvas.buffer_rgba()
        from PySide6.QtGui import QImage, QPixmap
        self._latex_image_data = bytes(image)
        qimage = QImage(
            self._latex_image_data,
            int(width),
            int(height),
            int(width) * 4,
            QImage.Format_RGBA8888,
        )
        self.setPixmap(QPixmap.fromImage(qimage))
        self.setMinimumHeight(int(height))
        fig.clear()
        del fig


@dataclass
class SpringConfig:
    model_name: str
    strain_name: Optional[str] = None
    ogden_terms: int = 1
    param_values: Optional[List[float]] = None


class OptimizerWorker(QThread):
    finished = Signal(object, object)
    failed = Signal(str)

    def __init__(self, optimizer, initial_guess, bounds, method):
        super().__init__()
        self.optimizer = optimizer
        self.initial_guess = initial_guess
        self.bounds = bounds
        self.method = method

    def run(self):
        try:
            result = self.optimizer.fit(self.initial_guess, self.bounds, method=self.method)
            self.finished.emit(result, self.optimizer)
        except Exception as exc:
            self.failed.emit(str(exc))


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

    def apply_theme(self):
        self.figure.patch.set_facecolor("none")
        for ax in self.axes:
            apply_theme_to_axis(ax)
            ax.tick_params(labelsize=10)


class SpringWidget(QGroupBox):
    def __init__(self, index, parent=None, on_change=None):
        super().__init__("")
        self.index = index
        self.on_change = on_change
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Select..."] + get_model_list())
        self.model_label = QLabel("Model")
        self.strain_label = QLabel("Strain")
        self.strain_combo = QComboBox()
        self.strain_combo.addItems(list(STRAIN_CONFIGS.keys()))
        self.strain_combo.setEnabled(False)
        self.strain_combo.setVisible(False)
        self.strain_label.setVisible(False)
        self.ogden_label = QLabel("Ogden terms")
        self.ogden_terms = QSpinBox()
        self.ogden_terms.setRange(1, 6)
        self.ogden_terms.setValue(1)
        self.ogden_terms.setEnabled(False)
        self.ogden_terms.setVisible(False)
        self.ogden_label.setVisible(False)

        self.formula_label = LatexLabel()
        self.strain_formula_title = QLabel("Generalized Strain")
        self.strain_formula_title.setFont(QFont("Helvetica", 10, QFont.Bold))
        self.strain_formula_title.setVisible(False)
        self.strain_formula_label = LatexLabel()
        self.reference_title = QLabel("Reference")
        self.reference_title.setFont(QFont("Helvetica", 10, QFont.Bold))
        self.reference_title.setVisible(False)
        self.reference_label = QLabel()
        self.reference_label.setTextFormat(Qt.RichText)
        self.reference_label.setOpenExternalLinks(True)
        self.reference_label.setWordWrap(True)
        self.params_layout = QGridLayout()
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        self.params_layout.setHorizontalSpacing(10)
        self.params_layout.setVerticalSpacing(6)
        self.params_layout.setColumnStretch(1, 1)
        self.params_layout.setColumnStretch(3, 1)
        self.params_layout.setColumnStretch(5, 1)

        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        header = QLabel(f"Spring {index}")
        header.setFont(QFont("Helvetica", 12, QFont.Bold))
        layout.addWidget(header)

        controls = QGridLayout()
        controls.setHorizontalSpacing(10)
        controls.setVerticalSpacing(6)
        controls.addWidget(self.model_label, 0, 0)
        controls.addWidget(self.model_combo, 0, 1)
        controls.addWidget(self.strain_label, 0, 2)
        controls.addWidget(self.strain_combo, 0, 3)
        controls.addWidget(self.ogden_label, 0, 4)
        controls.addWidget(self.ogden_terms, 0, 5)
        controls.setColumnStretch(1, 2)
        controls.setColumnStretch(3, 2)
        layout.addLayout(controls)

        content = QGridLayout()
        content.setHorizontalSpacing(16)
        content.setVerticalSpacing(6)
        content.setContentsMargins(0, 0, 0, 0)

        params_block = QVBoxLayout()
        params_label = QLabel("Parameters")
        params_label.setFont(QFont("Helvetica", 11, QFont.Bold))
        params_block.addWidget(params_label)
        self.params_widget = QWidget()
        self.params_widget.setLayout(self.params_layout)
        self.params_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        params_block.addWidget(self.params_widget)

        formula_block = QVBoxLayout()
        formula_label = QLabel("Formula")
        formula_label.setFont(QFont("Helvetica", 11, QFont.Bold))
        formula_block.addWidget(formula_label)
        formula_block.addWidget(self.formula_label)
        formula_block.addWidget(self.strain_formula_title)
        formula_block.addWidget(self.strain_formula_label)
        formula_block.addWidget(self.reference_title)
        formula_block.addWidget(self.reference_label)

        content.addLayout(params_block, 0, 0)
        content.addLayout(formula_block, 0, 1)
        content.setColumnStretch(0, 2)
        content.setColumnStretch(1, 3)
        layout.addLayout(content)

        self.setLayout(layout)
        self.setFlat(True)
        self.setStyleSheet(
            "QGroupBox { border: 1px solid palette(mid); border-radius: 8px; }"
        )

        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.strain_combo.currentTextChanged.connect(self._on_model_changed)
        self.ogden_terms.valueChanged.connect(self._on_model_changed)
        self.param_edits = []
        self._param_prefix = f"{self.model_combo.currentText()}_{self.index}_"
        self._model_name = "Select..."

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
        match = re.match(r"^([A-Za-z]+)(\d+)$", name)
        if match:
            base, idx = match.groups()
            if self._model_name == "Hill" and base.lower() in ("m", "n"):
                return base
            if base.lower() == "mu":
                return f"&mu;<sub>{idx}</sub>"
            if base.lower() == "alpha":
                return f"&alpha;<sub>{idx}</sub>"
            return f"{base}<sub>{idx}</sub>"
        if name.lower() == "mu":
            return "&mu;"
        if name.lower() == "alpha":
            return "&alpha;"
        return name

    def _on_model_changed(self):
        model_name = self.model_combo.currentText()
        self._model_name = model_name
        is_hill = model_name == "Hill"
        is_ogden = model_name == "Ogden"
        self.strain_combo.setEnabled(is_hill)
        self.strain_combo.setVisible(is_hill)
        self.strain_label.setVisible(is_hill)
        self.ogden_terms.setEnabled(is_ogden)
        self.ogden_terms.setVisible(is_ogden)
        self.ogden_label.setVisible(is_ogden)
        self._clear_params()

        if model_name == "Select...":
            self.formula_label.clear()
            self.strain_formula_label.clear()
            self.strain_formula_title.setVisible(False)
            self.reference_title.setVisible(False)
            self.reference_label.clear()
            if self.on_change:
                self.on_change()
            return

        if model_name == "Hill":
            strain_name = self.strain_combo.currentText()
            func = MaterialModels.create_hill_model(strain_name)
        elif model_name == "Ogden":
            func = MaterialModels.create_ogden_model(self.ogden_terms.value())
        else:
            func = getattr(MaterialModels, model_name)

        self._param_prefix = f"{model_name}_{self.index}_"
        formula = getattr(func, "formula", "")
        self.formula_label.set_latex(formula)
        strain_formula = getattr(func, "strain_formula", "")
        if strain_formula:
            self.strain_formula_title.setVisible(True)
            self.strain_formula_label.set_latex(strain_formula)
        else:
            self.strain_formula_title.setVisible(False)
            self.strain_formula_label.clear()

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
        for idx, (name, default) in enumerate(zip(temp_net.param_names, temp_net.initial_guess)):
            row = idx
            col = 0
            label = QLabel()
            label.setTextFormat(Qt.RichText)
            label.setText(self._format_param_label(name))
            edit = QLineEdit()
            text_value = f"{float(default):.4g}"
            edit.setText(text_value)
            edit.setPlaceholderText(text_value)
            edit.setMinimumWidth(140)
            edit.setMinimumHeight(26)
            edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.params_layout.addWidget(label, row, col)
            self.params_layout.addWidget(edit, row, col + 1)
            self.param_edits.append((name, edit, default))
        if self.on_change:
            self.on_change(changed=True)

    def is_valid(self):
        return self.model_combo.currentText() != "Select..."

    def build_config(self):
        model_name = self.model_combo.currentText()
        if model_name == "Hill":
            strain_name = self.strain_combo.currentText()
            func = MaterialModels.create_hill_model(strain_name)
        elif model_name == "Ogden":
            func = MaterialModels.create_ogden_model(self.ogden_terms.value())
        else:
            func = getattr(MaterialModels, model_name)

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

    def get_state(self):
        return {
            "model": self.model_combo.currentText(),
            "strain": self.strain_combo.currentText(),
            "ogden_terms": self.ogden_terms.value(),
            "params": [edit.text().strip() for _, edit, _ in self.param_edits],
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
        # Apply params after widgets are built
        values = state.get("params", [])
        for i, (_, edit, _) in enumerate(self.param_edits):
            if i < len(values) and values[i]:
                edit.setText(values[i])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hyperelastic Calibration (Desktop)")
        self.setMinimumSize(1400, 900)
        os.environ["CALIBRATION_DATA_DIR"] = os.path.join(base_dir, "data")
        self.step_names = ["Experimental Data", "Model Architecture", "Optimization", "Prediction"]
        self.current_step = 0
        self.section_widgets = {}
        self.latest_optimizer = None
        self.latest_result = None
        self.latest_network = None
        self._bt_mix_warned = False

        root = QWidget()
        root_layout = QHBoxLayout(root)

        sidebar = self._build_sidebar()
        root_layout.addWidget(sidebar)

        self.content = self._build_content()
        root_layout.addWidget(self.content, 1)

        self.setCentralWidget(root)

        self.datasets = get_available_datasets()
        self._populate_authors()

    def _build_sidebar(self):
        sidebar = QGroupBox("Navigation")
        layout = QVBoxLayout()

        about_box = QGroupBox("About")
        about_layout = QVBoxLayout()
        about_layout.addWidget(QLabel("Chongran Zhao"))
        email = QLabel("<a href='mailto:chongranzhao@outlook.com'>chongranzhao@outlook.com</a>")
        email.setTextFormat(Qt.RichText)
        email.setOpenExternalLinks(True)
        about_layout.addWidget(email)
        site = QLabel("<a href='https://chongran-zhao.github.io'>chongran-zhao.github.io</a>")
        site.setTextFormat(Qt.RichText)
        site.setOpenExternalLinks(True)
        about_layout.addWidget(site)
        about_box.setLayout(about_layout)
        layout.addWidget(about_box)

        workflow_box = QGroupBox("Workflow")
        workflow_layout = QVBoxLayout()
        self.step_list = QListWidget()
        self.step_list.addItems(self.step_names)
        self.step_list.setCurrentRow(0)
        self.step_list.currentRowChanged.connect(self._on_step_selected)
        workflow_layout.addWidget(self.step_list)
        workflow_box.setLayout(workflow_layout)
        layout.addWidget(workflow_box)

        layout.addStretch()
        sidebar.setLayout(layout)
        sidebar.setMaximumWidth(240)
        return sidebar

    def _build_content(self):
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)

        header = QLabel("Hyperelastic Calibration")
        header.setFont(QFont("Helvetica", 20, QFont.Bold))
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

    def _build_data_section(self):
        self.data_box = QGroupBox("1. Experimental Data")
        layout = QVBoxLayout()

        form = QFormLayout()
        self.author_combo = QComboBox()
        self.author_combo.currentTextChanged.connect(self._on_author_changed)
        form.addRow("Author / Dataset", self.author_combo)
        layout.addLayout(form)

        self.modes_grid = QGridLayout()
        layout.addLayout(self.modes_grid)

        self.reference_label = QLabel()
        self.reference_label.setTextFormat(Qt.RichText)
        self.reference_label.setOpenExternalLinks(True)
        layout.addWidget(self.reference_label)

        self.preview_canvas = MatplotlibCanvas(width=8.8, height=4.6)
        self.preview_canvas.setMinimumHeight(360)
        layout.addWidget(self.preview_canvas)

        preview_actions = QHBoxLayout()
        preview_actions.addStretch()
        self.preview_save_btn = QPushButton("Save Preview Plot")
        self.preview_save_btn.clicked.connect(lambda: self._save_figure(self.preview_canvas, "preview_plot"))
        preview_actions.addWidget(self.preview_save_btn)
        layout.addLayout(preview_actions)

        self.data_next_btn = QPushButton("Next: Model Architecture")
        self.data_next_btn.setEnabled(False)
        self.data_next_btn.clicked.connect(lambda: self._set_step(1))
        layout.addWidget(self.data_next_btn, alignment=Qt.AlignRight)

        self.data_box.setLayout(layout)
        return self.data_box

    def _build_model_section(self):
        self.model_box = QGroupBox("2. Model Architecture")
        layout = QVBoxLayout()

        form = QFormLayout()
        self.springs_spin = QSpinBox()
        self.springs_spin.setRange(1, 6)
        self.springs_spin.setValue(1)
        self.springs_spin.valueChanged.connect(self._rebuild_springs)
        form.addRow("Parallel Springs", self.springs_spin)
        layout.addLayout(form)

        self.spring_container = QVBoxLayout()
        self.spring_container.setSpacing(12)
        layout.addLayout(self.spring_container)

        self.model_next_btn = QPushButton("Next: Optimization")
        self.model_next_btn.setEnabled(False)
        self.model_next_btn.clicked.connect(lambda: self._set_step(2))
        layout.addWidget(self.model_next_btn, alignment=Qt.AlignRight)

        self.model_box.setLayout(layout)

        self._rebuild_springs(self.springs_spin.value())
        return self.model_box

    def _build_optimization_section(self):
        self.opt_box = QGroupBox("3. Optimization")
        layout = QVBoxLayout()

        self.method_combo = QComboBox()
        self.method_combo.addItems(["L-BFGS-B"])
        layout.addWidget(self.method_combo)

        self.run_button = QPushButton("Start Calibration")
        self.run_button.clicked.connect(self._run_optimization)
        layout.addWidget(self.run_button)

        self.opt_status = QLabel("")
        layout.addWidget(self.opt_status)

        self.loss_label = QLabel("Final Loss: -")
        layout.addWidget(self.loss_label)

        self.calib_results_box = QGroupBox("Calibration Results")
        calib_layout = QHBoxLayout()
        self.calib_params_box = QGroupBox("Optimized Parameters")
        params_layout = QVBoxLayout()
        self.calib_params_area = QScrollArea()
        self.calib_params_area.setWidgetResizable(True)
        self.calib_params_widget = QWidget()
        self.calib_params_layout = QVBoxLayout(self.calib_params_widget)
        self.calib_params_layout.setSpacing(8)
        self.calib_params_layout.addStretch()
        self.calib_params_area.setWidget(self.calib_params_widget)
        params_layout.addWidget(self.calib_params_area)
        self.calib_params_box.setLayout(params_layout)
        calib_layout.addWidget(self.calib_params_box, 1)

        self.calib_canvas = MatplotlibCanvas(width=8.0, height=4.4)
        self.calib_canvas.setMinimumHeight(340)
        calib_plot_container = QWidget()
        calib_plot_layout = QVBoxLayout(calib_plot_container)
        calib_plot_layout.setContentsMargins(0, 0, 0, 0)
        calib_plot_layout.addWidget(self.calib_canvas, 1)
        calib_actions = QHBoxLayout()
        calib_actions.addStretch()
        self.calib_save_btn = QPushButton("Save Calibration Plot")
        self.calib_save_btn.clicked.connect(lambda: self._save_figure(self.calib_canvas, "calibration_plot"))
        calib_actions.addWidget(self.calib_save_btn)
        calib_plot_layout.addLayout(calib_actions)
        calib_layout.addWidget(calib_plot_container, 2)
        self.calib_results_box.setLayout(calib_layout)
        self.calib_results_box.setVisible(False)
        layout.addWidget(self.calib_results_box)

        self.opt_next_btn = QPushButton("Next: Prediction")
        self.opt_next_btn.setEnabled(False)
        self.opt_next_btn.clicked.connect(lambda: self._set_step(3))
        layout.addWidget(self.opt_next_btn, alignment=Qt.AlignRight)

        self.opt_box.setLayout(layout)
        return self.opt_box

    def _build_prediction_section(self):
        self.prediction_box = QGroupBox("4. Prediction")
        layout = QHBoxLayout()

        left_panel = QVBoxLayout()
        self.prediction_params_box = QGroupBox("Parameters")
        pred_params_layout = QVBoxLayout()
        self.prediction_params_area = QScrollArea()
        self.prediction_params_area.setWidgetResizable(True)
        self.prediction_params_widget = QWidget()
        self.prediction_params_layout = QVBoxLayout(self.prediction_params_widget)
        self.prediction_params_layout.setSpacing(8)
        self.prediction_params_layout.addStretch()
        self.prediction_params_area.setWidget(self.prediction_params_widget)
        pred_params_layout.addWidget(self.prediction_params_area)
        self.prediction_params_box.setLayout(pred_params_layout)
        left_panel.addWidget(self.prediction_params_box)

        self.prediction_modes_box = QGroupBox("Prediction Data")
        modes_layout = QVBoxLayout()
        self.prediction_modes_widget = QWidget()
        self.prediction_modes_layout = QGridLayout(self.prediction_modes_widget)
        modes_layout.addWidget(self.prediction_modes_widget)
        self.prediction_button = QPushButton("Update Prediction")
        self.prediction_button.clicked.connect(self._update_prediction_plot)
        modes_layout.addWidget(self.prediction_button)
        self.prediction_modes_box.setLayout(modes_layout)
        left_panel.addWidget(self.prediction_modes_box)
        left_panel.addStretch()

        layout.addLayout(left_panel, 1)

        self.prediction_canvas = MatplotlibCanvas(width=8.0, height=4.4)
        self.prediction_canvas.setMinimumHeight(340)
        pred_plot_container = QWidget()
        pred_plot_layout = QVBoxLayout(pred_plot_container)
        pred_plot_layout.setContentsMargins(0, 0, 0, 0)
        pred_plot_layout.addWidget(self.prediction_canvas, 1)
        pred_actions = QHBoxLayout()
        pred_actions.addStretch()
        self.pred_save_btn = QPushButton("Save Prediction Plot")
        self.pred_save_btn.clicked.connect(lambda: self._save_figure(self.prediction_canvas, "prediction_plot"))
        pred_actions.addWidget(self.pred_save_btn)
        pred_plot_layout.addLayout(pred_actions)
        layout.addWidget(pred_plot_container, 2)

        self.prediction_box.setLayout(layout)
        return self.prediction_box

    def _save_figure(self, canvas, default_name):
        from PySide6.QtWidgets import QFileDialog

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

    def _populate_authors(self):
        self.author_combo.blockSignals(True)
        self.author_combo.clear()
        self.author_combo.addItem("Select...")
        for author in sorted(self.datasets.keys()):
            self.author_combo.addItem(author)
        self.author_combo.blockSignals(False)

    def _on_author_changed(self, text):
        self._reset_results()
        self._bt_mix_warned = False
        for i in reversed(range(self.modes_grid.count())):
            widget = self.modes_grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        if text == "Select..." or text not in self.datasets:
            self.reference_label.setText("")
            self.preview_canvas.ax.clear()
            self.preview_canvas.apply_theme()
            self.preview_canvas.draw()
            self.data_next_btn.setEnabled(False)
            return

        modes = self.datasets[text]
        for idx, mode in enumerate(modes):
            checkbox = QCheckBox(format_mode_label(mode))
            checkbox.stateChanged.connect(self._update_preview)
            row = idx // 3
            col = idx % 3
            self.modes_grid.addWidget(checkbox, row, col)

        reference = DATASET_REFERENCES.get(text)
        if reference:
            label, url = reference
            self.reference_label.setText(f"<a href='{url}'>{label}</a>")
        else:
            self.reference_label.setText("")

        self._update_preview()

    def _selected_modes(self):
        modes = []
        for i in range(self.modes_grid.count()):
            widget = self.modes_grid.itemAt(i).widget()
            if isinstance(widget, QCheckBox) and widget.isChecked():
                modes.append(widget.text())
        # Map displayed labels back to mode keys
        mapping = {format_mode_label(m): m for m in self.datasets.get(self.author_combo.currentText(), [])}
        return [mapping.get(m, m) for m in modes]

    def _update_preview(self):
        self._reset_results()
        author = self.author_combo.currentText()
        if author == "Select...":
            return
        modes = self._selected_modes()
        if not modes:
            self.preview_canvas.figure.clear()
            ax = self.preview_canvas.figure.add_subplot(111)
            self.preview_canvas.set_axes([ax])
            self.preview_canvas.apply_theme()
            self.preview_canvas.draw()
            self.data_next_btn.setEnabled(False)
            return
        configs = [{"author": author, "mode": m} for m in modes]
        data = load_experimental_data(configs)
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
            return

        self.preview_canvas.figure.clear()
        ax = self.preview_canvas.figure.add_subplot(111)
        self.preview_canvas.set_axes([ax])
        self.preview_canvas.apply_theme()
        for idx, d in enumerate(data):
            stretch = d["stretch"]
            stress = d["stress_exp"]
            label = format_mode_label(d.get("mode_raw", d["mode"]))
            if d["mode"] == "BT":
                comp_11, comp_22 = get_bt_component_labels(d.get("stress_type", "PK1"))
                if np.ndim(stress) == 1:
                    ax.plot(stretch, stress, "o", label=f"{label} {comp_11}")
                else:
                    ax.plot(stretch, stress[:, 0], "o", label=f"{label} {comp_11}")
                    ax.plot(stretch, stress[:, 1], "^", label=f"{label} {comp_22}")
                ax.set_xlabel(r"$\lambda_1$")
            else:
                ax.plot(stretch, stress, "o", label=label)
                ax.set_xlabel(r"$\lambda_1$")
            ax.set_ylabel(get_uniaxial_component_label(d.get("stress_type", "PK1")))
        ax.legend(fontsize=8)
        self.preview_canvas.figure.tight_layout()
        self.preview_canvas.draw()
        self.data_next_btn.setEnabled(True)

    def _rebuild_springs(self, count):
        previous_states = []
        for i in range(self.spring_container.count()):
            widget = self.spring_container.itemAt(i).widget()
            if widget:
                previous_states.append(widget.get_state())
        while self.spring_container.count():
            item = self.spring_container.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        for i in range(1, count + 1):
            spring_widget = SpringWidget(i, on_change=self._on_spring_config_changed)
            if i <= len(previous_states):
                spring_widget.apply_state(previous_states[i - 1])
            self.spring_container.addWidget(spring_widget)
        self._on_spring_config_changed()

    def _run_optimization(self):
        author = self.author_combo.currentText()
        if author == "Select...":
            QMessageBox.warning(self, "Missing data", "Select a dataset.")
            return
        modes = self._selected_modes()
        if not modes:
            QMessageBox.warning(self, "Missing data", "Select at least one mode.")
            return

        configs = [{"author": author, "mode": m} for m in modes]
        exp_data = load_experimental_data(configs)

        execution_network = ParallelNetwork()
        initial_guess = []
        for idx in range(self.spring_container.count()):
            spring = self.spring_container.itemAt(idx).widget()
            if not spring.is_valid():
                QMessageBox.warning(self, "Model error", "Select a model for each spring.")
                return
            func, params = spring.build_config()
            execution_network.add_model(func, f"{func.__name__}_{idx+1}")
            initial_guess.extend(params)

        solver = Kinematics(execution_network, execution_network.param_names)
        optimizer = MaterialOptimizer(solver, exp_data)
        bounds = execution_network.bounds

        method = self.method_combo.currentText()
        self.opt_status.setText("Running optimization...")
        self.run_button.setEnabled(False)
        self._set_step(2)

        self.worker = OptimizerWorker(optimizer, initial_guess, bounds, method)
        self.worker.finished.connect(self._on_optimization_finished)
        self.worker.failed.connect(self._on_optimization_failed)
        self.worker.start()

    def _on_optimization_finished(self, result, optimizer):
        self.run_button.setEnabled(True)
        if not result.success:
            self.opt_status.setText(f"Optimization failed: {result.message}")
            return
        self.opt_status.setText("Optimization completed.")
        self.loss_label.setText(f"Final Loss: {result.fun:.6f}")
        self.opt_next_btn.setEnabled(True)
        self.latest_optimizer = optimizer
        self.latest_result = result
        self.latest_network = optimizer.solver.energy_function
        self._populate_calibration_params(optimizer.param_names, result.x)
        self._populate_prediction_params(optimizer.param_names, result.x)
        self._refresh_prediction_modes()
        self._plot_calibration_results()
        self.calib_results_box.setVisible(True)

    def _on_optimization_failed(self, message):
        self.run_button.setEnabled(True)
        self.opt_status.setText(f"Optimization failed: {message}")
        self.opt_next_btn.setEnabled(False)

    def _on_step_selected(self, index):
        if index < 0:
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

    def _set_step(self, index):
        if index == self.current_step:
            return
        self.current_step = index
        self.step_list.blockSignals(True)
        self.step_list.setCurrentRow(index)
        self.step_list.blockSignals(False)
        step_name = self.step_names[index]
        widget = self.section_widgets.get(step_name)
        if widget:
            self._update_section_visibility()
            if step_name == "Prediction":
                if self.latest_result is not None and self.latest_optimizer is not None:
                    self._populate_prediction_params(self.latest_optimizer.param_names, self.latest_result.x)
                self._refresh_prediction_modes()
            self.scroll.ensureWidgetVisible(widget, 0, 20)

    def _update_section_visibility(self):
        for idx, step_name in enumerate(self.step_names):
            widget = self.section_widgets.get(step_name)
            if widget:
                widget.setVisible(idx == self.current_step)

    def _on_spring_config_changed(self, changed=False):
        if changed:
            self._reset_results()
        springs = [self.spring_container.itemAt(i).widget() for i in range(self.spring_container.count())]
        if springs and all(spring.is_valid() for spring in springs):
            self.model_next_btn.setEnabled(True)
        else:
            self.model_next_btn.setEnabled(False)

    def _format_result_param_label(self, name):
        parts = name.split("_")
        short = parts[-1] if parts else name
        match = re.match(r"^([A-Za-z]+)(\d+)$", short)
        if match:
            base, idx = match.groups()
            if base.lower() == "mu":
                return f"&mu;<sub>{idx}</sub>"
            if base.lower() == "alpha":
                return f"&alpha;<sub>{idx}</sub>"
            return f"{base}<sub>{idx}</sub>"
        if short.lower() == "mu":
            return "&mu;"
        if short.lower() == "alpha":
            return "&alpha;"
        return short

    def _clear_calibration_params(self):
        while self.calib_params_layout.count():
            item = self.calib_params_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def _populate_calibration_params(self, param_names, values):
        self._clear_calibration_params()
        for name, value in zip(param_names, values):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            label = QLabel()
            label.setTextFormat(Qt.RichText)
            label.setText(self._format_result_param_label(name))
            edit = QLineEdit(f"{value:.6g}")
            edit.setReadOnly(True)
            edit.setMinimumWidth(140)
            edit.setMinimumHeight(26)
            edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row_layout.addWidget(label)
            row_layout.addWidget(edit, 1)
            self.calib_params_layout.addWidget(row)
        self.calib_params_layout.addStretch()

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
            label.setText(self._format_result_param_label(name))
            edit = QLineEdit(f"{value:.6g}")
            edit.setMinimumWidth(140)
            edit.setMinimumHeight(26)
            edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row_layout.addWidget(label)
            row_layout.addWidget(edit, 1)
            self.prediction_params_layout.addWidget(row)
        self.prediction_params_layout.addStretch()

    def _refresh_prediction_modes(self):
        for i in reversed(range(self.prediction_modes_layout.count())):
            widget = self.prediction_modes_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        author = self.author_combo.currentText()
        if author == "Select...":
            return
        available = self.datasets.get(author, [])
        for idx, mode in enumerate(available):
            checkbox = QCheckBox(format_mode_label(mode))
            checkbox.setProperty("mode_key", mode)
            checkbox.stateChanged.connect(self._on_prediction_mode_toggled)
            row = idx // 2
            col = idx % 2
            self.prediction_modes_layout.addWidget(checkbox, row, col)
        self._apply_prediction_mode_rules()

    def _on_prediction_mode_toggled(self, _state):
        self._apply_prediction_mode_rules()

    def _apply_prediction_mode_rules(self):
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

        for checkbox, is_bt in checkboxes:
            if bt_selected and not is_bt and not checkbox.isChecked():
                checkbox.setEnabled(False)
            elif non_bt_selected and is_bt and not checkbox.isChecked():
                checkbox.setEnabled(False)
            else:
                checkbox.setEnabled(True)

    def _get_prediction_modes(self):
        modes = []
        mapping = {format_mode_label(m): m for m in self.datasets.get(self.author_combo.currentText(), [])}
        for i in range(self.prediction_modes_layout.count()):
            widget = self.prediction_modes_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox) and widget.isChecked():
                modes.append(mapping.get(widget.text(), widget.text()))
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

        self.calib_canvas.figure.clear()
        ax = self.calib_canvas.figure.add_subplot(111)
        self.calib_canvas.set_axes([ax])
        self.calib_canvas.apply_theme()

        colors_calib = {"UT": "#2980b9", "ET": "#c0392b", "PS": "#27ae60", "BT": "#8e44ad"}
        calib_data = optimizer.data
        bt_only = self._plot_dataset(self.calib_canvas.ax, calib_data, optimizer.solver, plot_params, colors_calib, "Exp", "Fit")

        if not bt_only:
            self.calib_canvas.ax.set_xlabel(r"$\lambda_1$")
            stress_types = {d.get("stress_type", "PK1") for d in calib_data if d["mode"] != "BT"}
            if len(stress_types) == 1:
                self.calib_canvas.ax.set_ylabel(get_uniaxial_component_label(stress_types.pop()))
            else:
                self.calib_canvas.ax.set_ylabel(r"$P_{11}$")
            self.calib_canvas.ax.legend(fontsize=7)
            self.calib_canvas.figure.tight_layout()
        self.calib_canvas.draw()

    def _reset_results(self):
        self.latest_optimizer = None
        self.latest_result = None
        self.latest_network = None
        self.loss_label.setText("Final Loss: -")
        self.opt_next_btn.setEnabled(False)
        self.calib_results_box.setVisible(False)
        self.calib_canvas.figure.clear()
        ax = self.calib_canvas.figure.add_subplot(111)
        self.calib_canvas.set_axes([ax])
        self.calib_canvas.apply_theme()
        self.calib_canvas.draw()
        self.prediction_canvas.figure.clear()
        ax = self.prediction_canvas.figure.add_subplot(111)
        self.prediction_canvas.set_axes([ax])
        self.prediction_canvas.apply_theme()
        self.prediction_canvas.draw()
        self._clear_calibration_params()
        self._clear_prediction_params()

    def _update_prediction_plot(self):
        if not self.latest_optimizer or not self.latest_result:
            return
        optimizer = self.latest_optimizer
        param_values = self._get_prediction_param_values()
        if len(param_values) != len(optimizer.param_names):
            param_values = list(self.latest_result.x)
        plot_params = dict(zip(optimizer.param_names, param_values))

        self.prediction_canvas.figure.clear()
        ax = self.prediction_canvas.figure.add_subplot(111)
        self.prediction_canvas.set_axes([ax])
        self.prediction_canvas.apply_theme()

        colors_pred = {"UT": "#3498db", "ET": "#e74c3c", "PS": "#2ecc71", "BT": "#a569bd"}
        pred_modes = self._get_prediction_modes()
        if pred_modes:
            author = self.author_combo.currentText()
            pred_configs = [{"author": author, "mode": m} for m in pred_modes]
            pred_data = load_experimental_data(pred_configs)
            bt_only = self._plot_dataset(self.prediction_canvas.ax, pred_data, optimizer.solver, plot_params, colors_pred, "Pred", "PredFit")
        else:
            bt_only = False

        if not bt_only:
            self.prediction_canvas.ax.set_xlabel(r"$\lambda_1$")
            if pred_modes:
                stress_types = {d.get("stress_type", "PK1") for d in pred_data if d["mode"] != "BT"}
                if len(stress_types) == 1:
                    self.prediction_canvas.ax.set_ylabel(get_uniaxial_component_label(stress_types.pop()))
                else:
                    self.prediction_canvas.ax.set_ylabel(r"$P_{11}$")
            else:
                self.prediction_canvas.ax.set_ylabel(r"$P_{11}$")
            self.prediction_canvas.ax.legend(fontsize=7)
            self.prediction_canvas.figure.tight_layout()
        self.prediction_canvas.draw()

    def _plot_bt_preview(self, canvas, data):
        author = self.author_combo.currentText()
        is_jones = author == "Jones_1975"
        fig = canvas.figure
        fig.clear()
        if is_jones:
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
                    y = stress[:, 0]
                lam2 = parse_lambda2(d.get("mode_raw", ""))
                label = f"λ₂={lam2}" if lam2 else "BT"
                marker = markers[idx % len(markers)]
                color = colors[idx % len(colors)]
                ax.plot(d["stretch"], y, marker, color=color, label=label, linestyle="None")
            ax.set_xlabel(r"$\lambda_1$")
            ax.set_ylabel(r"$\sigma_{11}-\sigma_{22}$")
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
        author = self.author_combo.currentText()
        is_jones = author == "Jones_1975"
        fig = canvas.figure
        fig.clear()
        if is_jones:
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
                    y = stress[:, 0]
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
                    stress_tensor = solver.get_Cauchy_stress(F, params)
                    comps = get_stress_components(stress_tensor, "BT")
                    model.append(comps[0] - comps[1])
                ax.plot(smooth, model, "-", color=color, label="_nolegend_")
            ax.set_xlabel(r"$\lambda_1$")
            ax.set_ylabel(r"$\sigma_{11}-\sigma_{22}$")
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
        for d in data:
            mode = d["mode"]
            stress_type = d.get("stress_type", "PK1")
            stretch = d["stretch"]
            stress = d["stress_exp"]
            mode_label = format_mode_label(d.get("mode_raw", mode))
            label = mode_label

            color = colors.get(mode, "black")
            if mode == "BT":
                comp_11, comp_22 = get_bt_component_labels(stress_type)
                if np.ndim(stress) == 1:
                    ax.plot(stretch, stress, "o", color=color, label=f"{exp_label} {label} {comp_11}")
                else:
                    ax.plot(stretch, stress[:, 0], "o", color=color, label=f"{exp_label} {label} {comp_11}")
                    ax.plot(stretch, stress[:, 1], "^", color=color, label=f"{exp_label} {label} {comp_22}")
            else:
                ax.plot(stretch, stress, "o", color=color, label=f"{exp_label} {label}")

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
                model.append(comps[0])
                if len(comps) > 1:
                    model2.append(comps[1])
            if mode == "BT":
                ax.plot(smooth, model, "-", color=color, label=f"{fit_label} {label} {comp_11}")
            else:
                ax.plot(smooth, model, "-", color=color, label=f"{fit_label} {label}")
            if model2:
                ax.plot(smooth, model2, "--", color=color, label=f"{fit_label} {label} {comp_22}")
        return False


def main():
    if multiprocessing.current_process().name != "MainProcess":
        return
    app = QApplication(sys.argv)
    app.setFont(QFont("Helvetica", 13))
    progress = QProgressDialog("Loading libraries...", None, 0, 5)
    progress.setWindowTitle("Starting Hyperelastic Calibration")
    progress.setMinimumDuration(0)
    progress.setCancelButton(None)
    progress.setValue(0)
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
