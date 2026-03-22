"""
Application theme and styling.
Modern dark theme with accent colors for the MetaOpticsAI application.

Updated to support:
- pyqtgraph live plots
- QDockWidget panels
- Optimization dashboard
"""

# =============================================================================
# COLOR CONSTANTS
# =============================================================================

# Background colors (darkest to lightest)
C_BG_DARKEST = "#0d0f1a"      # Deepest background (title bar)
C_BG_DARK = "#1a1b2e"          # Dark background (main areas) - matches original
C_BG_MID = "#1e2040"           # Mid background (panels)
C_BG_SURFACE = "#22253e"       # Surface background (input fields)
C_BG_ELEVATED = "#2d3055"      # Elevated elements (buttons)

# Accent colors
C_ACCENT_PRIMARY = "#3060c0"   # Primary blue (matches original)
C_ACCENT_SECONDARY = "#7eb8ff" # Light blue (matches original accent)
C_ACCENT_TERTIARY = "#a0c4ff"  # Soft blue for headers

# Status colors
C_SUCCESS = "#6ecf6e"          # Success green (matches original)
C_WARNING = "#f59e0b"          # Warning amber
C_ERROR = "#ff6b6b"            # Error red (matches original)
C_INFO = "#7eb8ff"             # Info blue

# Text colors
C_TEXT_PRIMARY = "#e0e0e8"     # Primary text (matches original)
C_TEXT_SECONDARY = "#b0b8d0"   # Secondary text
C_TEXT_MUTED = "#8890a8"       # Muted text (matches original subtitle)
C_TEXT_ACCENT = "#7eb8ff"      # Accent text (matches original)

# Border colors
C_BORDER_DARK = "#303560"      # Dark border (matches original)
C_BORDER_MID = "#3a3f65"       # Mid border
C_BORDER_LIGHT = "#404580"     # Light border
C_BORDER_ACCENT = "#5080e0"    # Accent border (focus states)

# =============================================================================
# PLOT COLORS FOR PYQTGRAPH
# =============================================================================

PLOT_COLORS = {
    # Named semantic colors for optimization dashboard
    "loss": "#ef4444",          # Red - loss curve
    "target": "#10b981",        # Green - target spectrum
    "current": "#3b82f6",       # Blue - current spectrum
    "best": "#6ecf6e",          # Light green - best loss line
    
    # Generic indexed colors
    "primary": "#3b82f6",       # Blue
    "secondary": "#10b981",     # Green
    "tertiary": "#f59e0b",      # Amber
    "quaternary": "#ef4444",    # Red
    "quinary": "#8b5cf6",       # Purple
    "senary": "#06b6d4",        # Cyan
    
    # Gradient for multi-line plots
    "gradient": [
        "#3b82f6",  # Blue
        "#10b981",  # Green
        "#f59e0b",  # Amber
        "#ef4444",  # Red
        "#8b5cf6",  # Purple
        "#06b6d4",  # Cyan
        "#ec4899",  # Pink
        "#14b8a6",  # Teal
    ]
}

# =============================================================================
# PYQTGRAPH THEME CONFIGURATION
# =============================================================================

def configure_pyqtgraph_theme():
    """
    Configure pyqtgraph to use our dark theme.
    Call this BEFORE creating any pyqtgraph widgets.
    """
    try:
        import pyqtgraph as pg
        
        # Set dark background
        pg.setConfigOptions(
            background=C_BG_MID,
            foreground=C_TEXT_PRIMARY,
            antialias=True,
            useOpenGL=False,  # More stable on Windows
        )
        
        # Configure default plot appearance
        pg.setConfigOption('leftButtonPan', False)
        
    except ImportError:
        print("Warning: pyqtgraph not installed. Install with: pip install pyqtgraph")

# =============================================================================
# DARK STYLESHEET (ORIGINAL + DOCK WIDGETS + NEW COMPONENTS)
# =============================================================================

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
QPushButton#accent, QPushButton#success {
    background-color: #2a8060;
    border-color: #40a080;
    color: #ffffff;
}
QPushButton#accent:hover, QPushButton#success:hover {
    background-color: #35a070;
}
QPushButton#warning {
    background-color: #b45309;
    border-color: #d97706;
    color: #ffffff;
}
QPushButton#warning:hover {
    background-color: #d97706;
}
QPushButton#danger {
    background-color: #dc2626;
    border-color: #ef4444;
    color: #ffffff;
}
QPushButton#danger:hover {
    background-color: #ef4444;
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

/* Optimize checkbox special styling */
QCheckBox[objectName="optimize_checkbox"] {
    color: #8890a8;
    font-size: 11px;
}
QCheckBox[objectName="optimize_checkbox"]:checked {
    color: #6ecf6e;
    font-weight: bold;
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
QScrollBar:horizontal {
    background: #1a1d35;
    height: 10px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background: #404580;
    border-radius: 5px;
    min-width: 30px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
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

/* =========================================================================
   DOCK WIDGETS (NEW)
   ========================================================================= */

QDockWidget {
    background-color: #1a1b2e;
    color: #e0e0e8;
    font-weight: bold;
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}

QDockWidget::title {
    background-color: #0d0f1a;
    padding: 8px 12px;
    text-align: left;
    font-size: 12px;
    font-weight: bold;
    color: #7eb8ff;
    border-bottom: 1px solid #303560;
}

QDockWidget::close-button, QDockWidget::float-button {
    background: transparent;
    border: none;
    padding: 2px;
}

QDockWidget::close-button:hover, QDockWidget::float-button:hover {
    background-color: #3a3f70;
    border-radius: 3px;
}

/* Splitter handles between docks */
QSplitter::handle {
    background-color: #303560;
}
QSplitter::handle:horizontal {
    width: 2px;
}
QSplitter::handle:vertical {
    height: 2px;
}
QSplitter::handle:hover {
    background-color: #5080e0;
}

/* Text edit for logs */
QTextEdit {
    background-color: #22253e;
    border: 1px solid #303560;
    border-radius: 6px;
    color: #e0e0e8;
    selection-background-color: #3060c0;
}

/* Menu bar */
QMenuBar {
    background-color: #0d0f1a;
    color: #e0e0e8;
    padding: 4px;
    border-bottom: 1px solid #303560;
}
QMenuBar::item {
    padding: 6px 12px;
    background: transparent;
}
QMenuBar::item:selected {
    background-color: #3060c0;
    border-radius: 4px;
}
QMenu {
    background-color: #1e2040;
    border: 1px solid #303560;
    border-radius: 6px;
    padding: 4px;
}
QMenu::item {
    padding: 8px 24px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #3060c0;
}
QMenu::separator {
    height: 1px;
    background-color: #303560;
    margin: 4px 8px;
}

/* Status bar */
QStatusBar {
    background-color: #0d0f1a;
    color: #8890a8;
    border-top: 1px solid #303560;
}
QStatusBar::item {
    border: none;
}

/* Sliders */
QSlider::groove:horizontal {
    border: 1px solid #303560;
    height: 6px;
    background: #22253e;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #5080e0;
    border: none;
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #7eb8ff;
}
"""

# Step indicator colors (kept for backward compatibility with legacy wizard)
STEP_ACTIVE_BG = "#3060c0"
STEP_ACTIVE_FG = "#ffffff"
STEP_DONE_BG = "#2a8060"
STEP_DONE_FG = "#ffffff"
STEP_INACTIVE_BG = "#22253e"
STEP_INACTIVE_FG = "#606880"
