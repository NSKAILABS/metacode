"""
Application theme and styling.
Modern dark theme with accent colors for the MetaOpticsAI application.
"""

DARK_STYLESHEET = """
QMainWindow {
    background-color: #1a1b2e;
}
QWidget {
    background-color: #1a1b2e;
    color: #e0e0e8;
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 13px;
}
QLabel {
    color: #e0e0e8;
    background: transparent;
}
QLabel#title {
    font-size: 22px;
    font-weight: bold;
    color: #7eb8ff;
}
QLabel#subtitle {
    font-size: 14px;
    color: #8890a8;
}
QLabel#section_header {
    font-size: 16px;
    font-weight: bold;
    color: #a0c4ff;
    padding: 8px 0;
}
QLabel#param_label {
    font-size: 13px;
    color: #b0b8d0;
}
QLabel#status_ok {
    color: #6ecf6e;
    font-weight: bold;
}
QLabel#status_error {
    color: #ff6b6b;
    font-weight: bold;
}
QLabel#metric_value {
    font-size: 28px;
    font-weight: bold;
    color: #7eb8ff;
}

/* Push Buttons */
QPushButton {
    background-color: #2d3055;
    border: 1px solid #404580;
    border-radius: 6px;
    padding: 8px 20px;
    color: #d0d4e8;
    font-weight: bold;
    font-size: 13px;
    min-height: 24px;
}
QPushButton:hover {
    background-color: #3a3f70;
    border-color: #5060a0;
}
QPushButton:pressed {
    background-color: #252850;
}
QPushButton:disabled {
    background-color: #1e2040;
    color: #505570;
    border-color: #303355;
}
QPushButton#primary {
    background-color: #3060c0;
    border-color: #4080e0;
    color: #ffffff;
}
QPushButton#primary:hover {
    background-color: #3870d8;
}
QPushButton#accent {
    background-color: #2a8060;
    border-color: #40a080;
    color: #ffffff;
}
QPushButton#accent:hover {
    background-color: #35a070;
}

/* Input fields */
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #22253e;
    border: 1px solid #3a3f65;
    border-radius: 5px;
    padding: 6px 10px;
    color: #e0e0e8;
    font-size: 13px;
    min-height: 22px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #5070c0;
}

/* Combo boxes */
QComboBox {
    background-color: #22253e;
    border: 1px solid #3a3f65;
    border-radius: 5px;
    padding: 6px 10px;
    color: #e0e0e8;
    min-height: 22px;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox QAbstractItemView {
    background-color: #22253e;
    border: 1px solid #3a3f65;
    selection-background-color: #3060c0;
    color: #e0e0e8;
}

/* Radio buttons and checkboxes */
QRadioButton, QCheckBox {
    color: #c8cce0;
    spacing: 8px;
    font-size: 13px;
}
QRadioButton::indicator, QCheckBox::indicator {
    width: 18px;
    height: 18px;
}

/* Group boxes */
QGroupBox {
    border: 1px solid #303560;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    color: #a0a8c0;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 16px;
    padding: 0 8px;
    color: #a0c4ff;
}

/* Tab widget */
QTabWidget::pane {
    border: 1px solid #303560;
    border-radius: 0 0 8px 8px;
    background: #1e2040;
}
QTabBar::tab {
    background: #22253e;
    border: 1px solid #303560;
    border-bottom: none;
    padding: 10px 24px;
    margin-right: 2px;
    border-radius: 6px 6px 0 0;
    color: #8890a8;
    font-weight: bold;
}
QTabBar::tab:selected {
    background: #1e2040;
    color: #7eb8ff;
    border-bottom: 2px solid #5080e0;
}
QTabBar::tab:hover:!selected {
    background: #282b48;
    color: #c0c8e0;
}

/* Progress bar */
QProgressBar {
    border: 1px solid #303560;
    border-radius: 6px;
    background-color: #1a1d35;
    text-align: center;
    color: #e0e0e8;
    min-height: 22px;
}
QProgressBar::chunk {
    background-color: #3070d0;
    border-radius: 5px;
}

/* Scroll areas */
QScrollArea {
    border: none;
    background: transparent;
}
QScrollBar:vertical {
    background: #1a1d35;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #404580;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

/* Stacked widget pages */
QStackedWidget {
    background: transparent;
}

/* Separator */
QFrame[frameShape="4"], QFrame[frameShape="5"] {
    color: #303560;
}

/* Table */
QTableWidget {
    background-color: #1e2040;
    border: 1px solid #303560;
    gridline-color: #303560;
    color: #e0e0e8;
    border-radius: 6px;
}
QTableWidget::item {
    padding: 6px;
}
QTableWidget::item:selected {
    background-color: #3060c0;
}
QHeaderView::section {
    background-color: #252850;
    color: #a0c4ff;
    border: 1px solid #303560;
    padding: 6px;
    font-weight: bold;
}
"""

# Step indicator colors
STEP_ACTIVE_BG = "#3060c0"
STEP_ACTIVE_FG = "#ffffff"
STEP_DONE_BG = "#2a8060"
STEP_DONE_FG = "#ffffff"
STEP_INACTIVE_BG = "#22253e"
STEP_INACTIVE_FG = "#606880"