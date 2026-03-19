"""
Main application window with step-by-step wizard navigation.

Workflow order:
  1. RCWA Sim      — Unit cell simulation (generates phase library)
  2. Wavelength    — Select design wavelength (or use RCWA data)
  3. Phase Curve   — View FDTD / RCWA phase data
  4. Phase Design  — Design phase mask
  5. GDSII Export  — Generate fabrication layout
  6. Analysis      — Optical performance metrics
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QStackedWidget, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont

from core.meta_data import MetaDataStore
from core.design import DesignState
from gui.theme import (
    DARK_STYLESHEET, STEP_ACTIVE_BG, STEP_ACTIVE_FG,
    STEP_DONE_BG, STEP_DONE_FG, STEP_INACTIVE_BG, STEP_INACTIVE_FG
)
from gui.rcwa_page import RCWAPage
from gui.wavelength_page import WavelengthPage
from gui.phase_curve_page import PhaseCurvePage
from gui.phase_design_page import PhaseDesignPage
from gui.gds_page import GDSPage
from gui.analysis_page import AnalysisPage


STEPS = [
    ("RCWA Sim", "Unit cell electromagnetic simulation"),
    ("Wavelength", "Select design wavelength"),
    ("Phase Curve", "View phase-vs-dimension data"),
    ("Phase Design", "Design phase mask"),
    ("GDSII Export", "Generate fabrication layout"),
    ("Analysis", "Optical performance metrics"),
]


class StepIndicator(QWidget):
    """Horizontal step progress indicator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.step_labels = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 8, 20, 8)
        layout.setSpacing(4)

        for idx, (name, desc) in enumerate(STEPS):
            lbl = QLabel(f"  {idx+1}. {name}  ")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            lbl.setMinimumHeight(36)
            lbl.setStyleSheet(
                f"background: {STEP_INACTIVE_BG}; color: {STEP_INACTIVE_FG}; "
                f"border-radius: 6px; padding: 4px 12px;"
            )
            self.step_labels.append(lbl)
            layout.addWidget(lbl)

            if idx < len(STEPS) - 1:
                arrow = QLabel("›")
                arrow.setStyleSheet("color: #404580; font-size: 18px; background: transparent;")
                arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(arrow)

    def set_active(self, step_idx: int):
        for i, lbl in enumerate(self.step_labels):
            if i < step_idx:
                lbl.setStyleSheet(
                    f"background: {STEP_DONE_BG}; color: {STEP_DONE_FG}; "
                    f"border-radius: 6px; padding: 4px 12px;"
                )
            elif i == step_idx:
                lbl.setStyleSheet(
                    f"background: {STEP_ACTIVE_BG}; color: {STEP_ACTIVE_FG}; "
                    f"border-radius: 6px; padding: 4px 12px;"
                )
            else:
                lbl.setStyleSheet(
                    f"background: {STEP_INACTIVE_BG}; color: {STEP_INACTIVE_FG}; "
                    f"border-radius: 6px; padding: 4px 12px;"
                )


class MetaOpticsMainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MetaOpticsAI — Photonics Design Platform")
        self.setMinimumSize(1050, 720)
        self.resize(1150, 780)

        self.data_store = MetaDataStore()
        self.design = DesignState()
        self.current_step = 0

        self._setup_ui()
        self._connect_signals()
        self._update_step(0)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Title bar
        title_bar = QWidget()
        title_bar.setStyleSheet("background: #12132a; padding: 8px;")
        tb_layout = QHBoxLayout(title_bar)
        tb_layout.setContentsMargins(20, 6, 20, 6)

        app_title = QLabel("MetaOpticsAI")
        app_title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        app_title.setStyleSheet("color: #7eb8ff; background: transparent;")
        tb_layout.addWidget(app_title)

        subtitle = QLabel("AI-Automated Photonics Design")
        subtitle.setStyleSheet("color: #606888; font-size: 13px; background: transparent;")
        tb_layout.addWidget(subtitle)
        tb_layout.addStretch()

        main_layout.addWidget(title_bar)

        # Step indicator
        self.step_indicator = StepIndicator()
        self.step_indicator.setStyleSheet("background: #16172e;")
        main_layout.addWidget(self.step_indicator)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #303560;")
        main_layout.addWidget(sep)

        # Stacked pages — NEW ORDER: RCWA first
        self.pages = QStackedWidget()

        self.rcwa_page = RCWAPage(self.design)          # Step 1
        self.wl_page = WavelengthPage(self.data_store)   # Step 2
        self.pc_page = PhaseCurvePage(self.data_store)    # Step 3
        self.pd_page = PhaseDesignPage(self.design)       # Step 4
        self.gds_page = GDSPage(self.design)              # Step 5
        self.analysis_page = AnalysisPage(self.design)    # Step 6

        self.pages.addWidget(self.rcwa_page)
        self.pages.addWidget(self.wl_page)
        self.pages.addWidget(self.pc_page)
        self.pages.addWidget(self.pd_page)
        self.pages.addWidget(self.gds_page)
        self.pages.addWidget(self.analysis_page)

        main_layout.addWidget(self.pages, stretch=1)

        # Navigation bar
        nav_bar = QWidget()
        nav_bar.setStyleSheet("background: #12132a;")
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(20, 8, 20, 8)

        self.btn_prev = QPushButton("← Previous")
        self.btn_prev.clicked.connect(self._go_prev)
        nav_layout.addWidget(self.btn_prev)

        self.btn_start_new = QPushButton("Start New")
        self.btn_start_new.clicked.connect(self._start_new)
        nav_layout.addWidget(self.btn_start_new)

        nav_layout.addStretch()

        self.step_desc = QLabel("")
        self.step_desc.setStyleSheet("color: #606888; font-size: 12px; background: transparent;")
        nav_layout.addWidget(self.step_desc)

        nav_layout.addStretch()

        self.btn_next = QPushButton("Next →")
        self.btn_next.setObjectName("primary")
        self.btn_next.clicked.connect(self._go_next)
        nav_layout.addWidget(self.btn_next)

        main_layout.addWidget(nav_bar)

    def _connect_signals(self):
        self.wl_page.wavelength_selected.connect(self._on_wavelength_selected)
        self.pd_page.design_ready.connect(self._on_design_ready)
        self.rcwa_page.sweep_data_ready.connect(self._on_rcwa_data_ready)

    def _on_wavelength_selected(self, wl: int):
        entry = self.data_store.get_entry(wl)
        if entry:
            self.design.wavelength = wl
            self.design.shape = entry.shape
            self.design.material = entry.material
            self.design.height = entry.height
            self.design.period = entry.period
            self.design.dimensions = entry.dimensions
            self.design.phases = entry.phases
            self.design.transmissions = entry.transmissions
            self.design.cross_width = entry.cross_width or 0
            self.design.fin_width = entry.fin_width or 0
            self.design.fin_length = entry.fin_length or 0

    def _on_design_ready(self):
        pass  # Design state already updated by PhaseDesignPage

    def _on_rcwa_data_ready(self):
        """RCWA sweep data pushed into DesignState — can skip wavelength page."""
        pass

    def _update_step(self, step: int):
        self.current_step = step
        self.pages.setCurrentIndex(step)
        self.step_indicator.set_active(step)
        self.step_desc.setText(STEPS[step][1])
        self.btn_prev.setEnabled(step > 0)

        if step == len(STEPS) - 1:
            self.btn_next.setText("Finish")
        else:
            self.btn_next.setText("Next →")

    def _go_next(self):
        # Step-specific logic (0-indexed with new order)
        if self.current_step == 0:
            # RCWA → Wavelength: RCWA is optional, can skip
            pass

        elif self.current_step == 1:
            # Wavelength → Phase Curve
            if self.design.wavelength == 0 and self.design.dimensions is None:
                return  # must either select wavelength or have RCWA data
            if self.design.wavelength > 0:
                self.pc_page.update_for_wavelength(self.design.wavelength)

        elif self.current_step == 2:
            # Phase Curve → Phase Design
            pass

        elif self.current_step == 3:
            # Phase Design → GDS Export
            if self.design.phase_mask is None:
                return  # must design phase mask

        elif self.current_step == 4:
            # GDS Export → Analysis
            self.gds_page.refresh_summary()

        if self.current_step < len(STEPS) - 1:
            next_step = self.current_step + 1
            if next_step == 4:
                self.gds_page.refresh_summary()
            self._update_step(next_step)

    def _go_prev(self):
        if self.current_step > 0:
            self._update_step(self.current_step - 1)

    def _start_new(self):
        self.design.reset()
        self._update_step(0)