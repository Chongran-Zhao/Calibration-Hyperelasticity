"""Theme primitives for the PySide desktop GUI."""

import re
import sys

from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontMetrics, QPalette
from PySide6.QtWidgets import QApplication


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
        "QWidget { color: palette(windowtext); font-family: \"Helvetica Neue\", \"Segoe UI\", Arial, sans-serif; }"
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


def strip_html(text):
    plain = re.sub(r"<[^>]*>", "", text or "")
    plain = plain.replace("&mu;", "mu").replace("&alpha;", "alpha").replace("&lambda;", "lambda")
    return plain


def label_width_for_text(widget, text, padding=24, minimum=70, maximum=280):
    metrics = QFontMetrics(widget.font())
    width = metrics.horizontalAdvance(strip_html(text)) + padding
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
