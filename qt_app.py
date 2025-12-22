import os
import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
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
    QLineEdit,
    QMainWindow,
    QMessageBox,
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
        return f"Biaxial Tension, λ2={lam2}"
    return MODE_DISPLAY_MAP.get(mode_key, mode_key)


def get_stress_type_label(stress_type):
    return "Cauchy stress" if stress_type == "cauchy" else "Nominal stress"


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
        self.setMinimumHeight(40)

    def set_latex(self, latex):
        fig = Figure(figsize=(4.5, 0.6), dpi=150)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(0.0, 0.5, f"${latex}$", fontsize=11, va="center", ha="left")
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = canvas.buffer_rgba()
        from PySide6.QtGui import QImage, QPixmap
        self._latex_image_data = bytes(image)
        qimage = QImage(self._latex_image_data, int(width), int(height), QImage.Format_RGBA8888)
        self.setPixmap(QPixmap.fromImage(qimage))


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
        super().__init__(fig)


class SpringWidget(QGroupBox):
    def __init__(self, index, parent=None):
        super().__init__(f"Spring {index}")
        self.index = index
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Select..."] + get_model_list())
        self.strain_combo = QComboBox()
        self.strain_combo.addItems(list(STRAIN_CONFIGS.keys()))
        self.strain_combo.setEnabled(False)
        self.ogden_terms = QSpinBox()
        self.ogden_terms.setRange(1, 6)
        self.ogden_terms.setValue(1)
        self.ogden_terms.setEnabled(False)

        self.formula_label = LatexLabel()
        self.strain_formula_label = LatexLabel()
        self.params_layout = QGridLayout()

        layout = QVBoxLayout()
        form = QFormLayout()
        form.addRow("Model", self.model_combo)
        form.addRow("Strain", self.strain_combo)
        form.addRow("Ogden terms", self.ogden_terms)
        layout.addLayout(form)

        params_box = QGroupBox("Parameters")
        params_box.setLayout(self.params_layout)
        layout.addWidget(params_box)

        formula_box = QGroupBox("Formula")
        formula_layout = QVBoxLayout()
        formula_layout.addWidget(self.formula_label)
        formula_layout.addWidget(self.strain_formula_label)
        formula_box.setLayout(formula_layout)
        layout.addWidget(formula_box)

        self.setLayout(layout)

        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.strain_combo.currentTextChanged.connect(self._on_model_changed)
        self.ogden_terms.valueChanged.connect(self._on_model_changed)
        self.param_edits = []

    def _clear_params(self):
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.param_edits = []

    def _on_model_changed(self):
        model_name = self.model_combo.currentText()
        self.strain_combo.setEnabled(model_name == "Hill")
        self.ogden_terms.setEnabled(model_name == "Ogden")
        self._clear_params()

        if model_name == "Select...":
            self.formula_label.clear()
            self.strain_formula_label.clear()
            return

        if model_name == "Hill":
            strain_name = self.strain_combo.currentText()
            func = MaterialModels.create_hill_model(strain_name)
        elif model_name == "Ogden":
            func = MaterialModels.create_ogden_model(self.ogden_terms.value())
        else:
            func = getattr(MaterialModels, model_name)

        formula = getattr(func, "formula", "")
        self.formula_label.set_latex(formula)
        strain_formula = getattr(func, "strain_formula", "")
        if strain_formula:
            self.strain_formula_label.set_latex(strain_formula)
        else:
            self.strain_formula_label.clear()

        temp_net = ParallelNetwork()
        temp_net.add_model(func, f"{model_name}_{self.index}")
        for idx, (name, default) in enumerate(zip(temp_net.param_names, temp_net.initial_guess)):
            row = idx // 3
            col = (idx % 3) * 2
            label = QLabel(name)
            edit = QLineEdit()
            edit.setText(f"{float(default):.4g}")
            self.params_layout.addWidget(label, row, col)
            self.params_layout.addWidget(edit, row, col + 1)
            self.param_edits.append((name, edit, default))

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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hyperelastic Calibration (Desktop)")
        self.setMinimumSize(1200, 800)
        os.environ["CALIBRATION_DATA_DIR"] = os.path.join(base_dir, "data")

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
        sidebar = QGroupBox("About")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Chongran Zhao"))
        layout.addWidget(QLabel("chongranzhao@outlook.com"))
        layout.addWidget(QLabel("chongran-zhao.github.io"))
        layout.addStretch()
        sidebar.setLayout(layout)
        sidebar.setMaximumWidth(220)
        return sidebar

    def _build_content(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)

        header = QLabel("Hyperelastic Calibration")
        header.setFont(QFont("Helvetica", 20, QFont.Bold))
        layout.addWidget(header)

        layout.addWidget(self._build_data_section())
        layout.addWidget(self._build_model_section())
        layout.addWidget(self._build_optimization_section())
        layout.addWidget(self._build_results_section())
        layout.addStretch()

        scroll.setWidget(container)
        return scroll

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

        self.preview_canvas = MatplotlibCanvas(width=6.5, height=2.0)
        layout.addWidget(self.preview_canvas)

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
        layout.addLayout(self.spring_container)
        self.model_box.setLayout(layout)

        self._rebuild_springs(self.springs_spin.value())
        return self.model_box

    def _build_optimization_section(self):
        self.opt_box = QGroupBox("3. Optimization")
        layout = QVBoxLayout()

        self.method_combo = QComboBox()
        self.method_combo.addItems(["L-BFGS-B", "trust-constr", "CG", "Newton-CG"])
        layout.addWidget(self.method_combo)

        self.run_button = QPushButton("Start Calibration")
        self.run_button.clicked.connect(self._run_optimization)
        layout.addWidget(self.run_button)

        self.opt_status = QLabel("")
        layout.addWidget(self.opt_status)

        self.opt_box.setLayout(layout)
        return self.opt_box

    def _build_results_section(self):
        self.results_box = QGroupBox("Results")
        layout = QVBoxLayout()

        self.loss_label = QLabel("Final Loss: -")
        layout.addWidget(self.loss_label)

        self.results_canvas = MatplotlibCanvas(width=6.5, height=3.4)
        layout.addWidget(self.results_canvas)

        self.results_box.setLayout(layout)
        return self.results_box

    def _populate_authors(self):
        self.author_combo.blockSignals(True)
        self.author_combo.clear()
        self.author_combo.addItem("Select...")
        for author in sorted(self.datasets.keys()):
            self.author_combo.addItem(author)
        self.author_combo.blockSignals(False)

    def _on_author_changed(self, text):
        for i in reversed(range(self.modes_grid.count())):
            widget = self.modes_grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        if text == "Select..." or text not in self.datasets:
            self.reference_label.setText("")
            self.preview_canvas.ax.clear()
            self.preview_canvas.draw()
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
        author = self.author_combo.currentText()
        if author == "Select...":
            return
        modes = self._selected_modes()
        if not modes:
            self.preview_canvas.ax.clear()
            self.preview_canvas.draw()
            return
        configs = [{"author": author, "mode": m} for m in modes]
        data = load_experimental_data(configs)

        self.preview_canvas.ax.clear()
        for idx, d in enumerate(data):
            stretch = d["stretch"]
            stress = d["stress_exp"]
            label = format_mode_label(d.get("mode_raw", d["mode"]))
            if d["mode"] == "BT":
                if np.ndim(stress) == 1:
                    self.preview_canvas.ax.plot(stretch, stress, "o", label=label)
                else:
                    self.preview_canvas.ax.plot(stretch, stress[:, 0], "o", label=f"{label} P11")
                    self.preview_canvas.ax.plot(stretch, stress[:, 1], "^", label=f"{label} P22")
                self.preview_canvas.ax.set_xlabel("lambda_1")
            else:
                self.preview_canvas.ax.plot(stretch, stress, "o", label=label)
                self.preview_canvas.ax.set_xlabel("lambda")
            self.preview_canvas.ax.set_ylabel(get_stress_type_label(d.get("stress_type", "PK1")))
        self.preview_canvas.ax.legend(fontsize=8)
        self.preview_canvas.ax.grid(True, linestyle="--", alpha=0.3)
        self.preview_canvas.draw()

    def _rebuild_springs(self, count):
        while self.spring_container.count():
            item = self.spring_container.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        for i in range(1, count + 1):
            spring_widget = SpringWidget(i)
            self.spring_container.addWidget(spring_widget)

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

        # Plot results
        plot_params = dict(zip(optimizer.param_names, result.x))
        data = optimizer.data
        self.results_canvas.ax.clear()

        colors = {"UT": "#2980b9", "ET": "#c0392b", "PS": "#27ae60", "BT": "#8e44ad"}
        for d in data:
            mode = d["mode"]
            stress_type = d.get("stress_type", "PK1")
            stretch = d["stretch"]
            stress = d["stress_exp"]
            label = f"{d['tag']} ({'Cauchy' if stress_type == 'cauchy' else 'Nominal'})"

            if mode == "BT":
                if np.ndim(stress) == 1:
                    self.results_canvas.ax.plot(stretch, stress, "o", label=f"Exp {label}")
                else:
                    self.results_canvas.ax.plot(stretch, stress[:, 0], "o", label=f"Exp {label} P11")
                    self.results_canvas.ax.plot(stretch, stress[:, 1], "^", label=f"Exp {label} P22")
            else:
                self.results_canvas.ax.plot(stretch, stress, "o", label=f"Exp {label}")

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
                    stress_tensor = optimizer.solver.get_Cauchy_stress(F, plot_params)
                else:
                    stress_tensor = optimizer.solver.get_1st_PK_stress(F, plot_params)
                comps = get_stress_components(stress_tensor, mode)
                model.append(comps[0])
                if len(comps) > 1:
                    model2.append(comps[1])
            self.results_canvas.ax.plot(smooth, model, "-", color=colors.get(mode, "black"))
            if model2:
                self.results_canvas.ax.plot(smooth, model2, "--", color=colors.get(mode, "black"))

        self.results_canvas.ax.set_xlabel("lambda")
        self.results_canvas.ax.set_ylabel("stress")
        self.results_canvas.ax.legend(fontsize=7)
        self.results_canvas.ax.grid(True, linestyle="--", alpha=0.3)
        self.results_canvas.draw()

    def _on_optimization_failed(self, message):
        self.run_button.setEnabled(True)
        self.opt_status.setText(f"Optimization failed: {message}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
