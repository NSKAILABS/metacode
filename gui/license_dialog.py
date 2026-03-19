"""
License Activation & Validation Dialog for MetaOpticsAI.

Provides:
  - Activation dialog (enter key + email, or purchase)
  - Validation splash screen (startup check with progress)
  - Expiry warning dialog (< 7 days remaining)
  - License info display in settings
"""

import webbrowser
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGroupBox, QGridLayout, QMessageBox,
    QProgressBar, QFrame, QApplication, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QFont, QIcon, QPixmap

from core.license_validator import LicenseValidator, ValidationResult


# ═══════════════════════════════════════════════════════════════════════
# Background validation worker
# ═══════════════════════════════════════════════════════════════════════

class ValidationWorker(QThread):
    """Run license validation in background thread."""
    finished = pyqtSignal(object)  # ValidationResult

    def __init__(self, validator: LicenseValidator, license_key: str = "",
                 email: str = "", mode: str = "validate"):
        super().__init__()
        self.validator = validator
        self.license_key = license_key
        self.email = email
        self.mode = mode  # "validate" or "activate"

    def run(self):
        if self.mode == "activate":
            result = self.validator.activate(self.license_key, self.email)
        else:
            result = self.validator.validate()
        self.finished.emit(result)


# ═══════════════════════════════════════════════════════════════════════
# Startup Validation Splash
# ═══════════════════════════════════════════════════════════════════════

class LicenseCheckSplash(QDialog):
    """
    Shown on startup while validating the license.
    Displays progress indicator and transitions to main app or activation dialog.
    """

    validation_complete = pyqtSignal(object)  # ValidationResult

    def __init__(self, validator: LicenseValidator, parent=None):
        super().__init__(parent)
        self.validator = validator
        self._result: ValidationResult = None
        self.setWindowTitle("MetaOpticsAI")
        self.setFixedSize(420, 280)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1b2e;
                border: 2px solid #3060c0;
                border-radius: 12px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(16)

        # App title
        title = QLabel("MetaOpticsAI")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #7eb8ff; background: transparent;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("AI-Automated Photonics Design Platform")
        subtitle.setStyleSheet("font-size: 12px; color: #8890a8; background: transparent;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(12)

        # Status
        self.status_label = QLabel("Verifying license...")
        self.status_label.setStyleSheet("font-size: 13px; color: #c0c8e0; background: transparent;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # Indeterminate
        self.progress.setFixedHeight(6)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 3px;
                background-color: #22253e;
            }
            QProgressBar::chunk {
                background-color: #3070d0;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress)

        layout.addStretch()

        # Version
        ver = QLabel(f"v{self.validator.app_version}")
        ver.setStyleSheet("font-size: 10px; color: #505570; background: transparent;")
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(ver)

    def start_validation(self):
        """Begin the background validation."""
        self._worker = ValidationWorker(self.validator, mode="validate")
        self._worker.finished.connect(self._on_validation_done)
        self._worker.start()

    def _on_validation_done(self, result: ValidationResult):
        self._result = result
        self.progress.setRange(0, 100)
        self.progress.setValue(100)

        if result.is_valid:
            self.status_label.setText("✓  " + result.message)
            self.status_label.setStyleSheet("font-size: 13px; color: #6ecf6e; background: transparent;")
            QTimer.singleShot(800, lambda: self._finish(result))
        else:
            self.status_label.setText(result.message)
            self.status_label.setStyleSheet("font-size: 12px; color: #ff8060; background: transparent;")
            QTimer.singleShot(1200, lambda: self._finish(result))

    def _finish(self, result: ValidationResult):
        self.validation_complete.emit(result)
        self.accept()

    @property
    def result(self) -> ValidationResult:
        return self._result


# ═══════════════════════════════════════════════════════════════════════
# Activation Dialog
# ═══════════════════════════════════════════════════════════════════════

class LicenseActivationDialog(QDialog):
    """
    License activation dialog — enter key, activate, or purchase.
    Shown when no valid license is found.
    """

    activated = pyqtSignal(object)  # ValidationResult

    PURCHASE_URL = "https://yourcompany.com/pricing"

    def __init__(self, validator: LicenseValidator,
                 message: str = "", parent=None):
        super().__init__(parent)
        self.validator = validator
        self._initial_message = message
        self._worker = None
        self.setWindowTitle("Activate MetaOpticsAI")
        self.setMinimumSize(520, 480)
        self.setMaximumSize(600, 560)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(14)

        # Header
        header = QLabel("Activate Your License")
        header.setStyleSheet("font-size: 20px; font-weight: bold; color: #7eb8ff;")
        layout.addWidget(header)

        desc = QLabel(
            "Enter your license key to activate MetaOpticsAI.\n"
            "Don't have a license? Purchase one below."
        )
        desc.setStyleSheet("font-size: 12px; color: #8890a8;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Error/info message
        if self._initial_message:
            msg_frame = QFrame()
            msg_frame.setStyleSheet("""
                QFrame {
                    background-color: #2a1520;
                    border: 1px solid #804050;
                    border-radius: 8px;
                    padding: 10px;
                }
            """)
            msg_layout = QVBoxLayout(msg_frame)
            msg_layout.setContentsMargins(12, 8, 12, 8)
            msg_label = QLabel(self._initial_message)
            msg_label.setStyleSheet("font-size: 12px; color: #ff8060; background: transparent;")
            msg_label.setWordWrap(True)
            msg_layout.addWidget(msg_label)
            layout.addWidget(msg_frame)

        # Input group
        input_group = QGroupBox("License Information")
        ig_layout = QGridLayout()
        ig_layout.setSpacing(10)

        ig_layout.addWidget(QLabel("License Key:"), 0, 0)
        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX")
        self.key_input.setMaxLength(39)  # 32 hex + 7 dashes
        self.key_input.setFont(QFont("Consolas", 12))
        self.key_input.setMinimumHeight(36)
        ig_layout.addWidget(self.key_input, 0, 1)

        ig_layout.addWidget(QLabel("Email:"), 1, 0)
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("your@email.com")
        self.email_input.setMinimumHeight(36)
        ig_layout.addWidget(self.email_input, 1, 1)

        input_group.setLayout(ig_layout)
        layout.addWidget(input_group)

        # Activate button
        self.activate_btn = QPushButton("  Activate License")
        self.activate_btn.setObjectName("primary")
        self.activate_btn.setMinimumHeight(44)
        self.activate_btn.setStyleSheet("""
            QPushButton {
                background-color: #3060c0;
                border: 1px solid #4080e0;
                border-radius: 8px;
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover { background-color: #3870d8; }
            QPushButton:disabled { background-color: #1e2040; color: #505570; }
        """)
        self.activate_btn.clicked.connect(self._on_activate)
        layout.addWidget(self.activate_btn)

        # Progress
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedHeight(4)
        self.progress.setVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar { border: none; background-color: #22253e; border-radius: 2px; }
            QProgressBar::chunk { background-color: #3070d0; border-radius: 2px; }
        """)
        layout.addWidget(self.progress)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-size: 12px; color: #c0c8e0;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #303560;")
        layout.addWidget(sep)

        # Purchase section
        purchase_layout = QHBoxLayout()
        purchase_label = QLabel("Don't have a license?")
        purchase_label.setStyleSheet("color: #8890a8;")
        purchase_layout.addWidget(purchase_label)

        purchase_btn = QPushButton("  Purchase License")
        purchase_btn.setObjectName("accent")
        purchase_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a8060;
                border: 1px solid #40a080;
                border-radius: 6px;
                color: #ffffff;
                font-weight: bold;
                padding: 8px 20px;
            }
            QPushButton:hover { background-color: #35a070; }
        """)
        purchase_btn.clicked.connect(self._on_purchase)
        purchase_layout.addWidget(purchase_btn)

        purchase_layout.addStretch()
        layout.addLayout(purchase_layout)

        layout.addStretch()

        # Machine info (small, for support reference)
        machine_id = self.validator.get_machine_id()[:16] + "..."
        machine_label = QLabel(f"Machine ID: {machine_id}")
        machine_label.setStyleSheet("font-size: 9px; color: #404560;")
        machine_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(machine_label)

    def _on_activate(self):
        key = self.key_input.text().strip()
        email = self.email_input.text().strip()

        if not key:
            self.status_label.setText("Please enter your license key.")
            self.status_label.setStyleSheet("font-size: 12px; color: #ff8060;")
            return

        # Validate key format (32 hex chars with optional dashes)
        clean_key = key.replace("-", "")
        if len(clean_key) != 32:
            self.status_label.setText("Invalid key format. Expected 32 hex characters (XXXX-XXXX-...).")
            self.status_label.setStyleSheet("font-size: 12px; color: #ff8060;")
            return

        try:
            int(clean_key, 16)
        except ValueError:
            self.status_label.setText("Invalid key format. Key must contain only hexadecimal characters.")
            self.status_label.setStyleSheet("font-size: 12px; color: #ff8060;")
            return

        # Start activation
        self.activate_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.status_label.setText("Contacting license server...")
        self.status_label.setStyleSheet("font-size: 12px; color: #c0c8e0;")

        self._worker = ValidationWorker(self.validator, license_key=key,
                                         email=email, mode="activate")
        self._worker.finished.connect(self._on_activation_result)
        self._worker.start()

    def _on_activation_result(self, result: ValidationResult):
        self.progress.setVisible(False)
        self.activate_btn.setEnabled(True)

        if result.is_valid:
            self.status_label.setText("✓  " + result.message)
            self.status_label.setStyleSheet("font-size: 12px; color: #6ecf6e;")
            self.activated.emit(result)
            QTimer.singleShot(1200, self.accept)
        else:
            self.status_label.setText("✗  " + result.message)
            self.status_label.setStyleSheet("font-size: 12px; color: #ff6b6b;")

    def _on_purchase(self):
        webbrowser.open(self.PURCHASE_URL)


# ═══════════════════════════════════════════════════════════════════════
# Expiry Warning Dialog
# ═══════════════════════════════════════════════════════════════════════

class ExpiryWarningDialog(QDialog):
    """Shown when license is valid but expiring soon (< 7 days)."""

    RENEW_URL = "https://yourcompany.com/renew"

    def __init__(self, days_remaining: int, plan_type: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("License Expiring Soon")
        self.setFixedSize(400, 220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        icon_label = QLabel("⚠")
        icon_label.setStyleSheet("font-size: 32px; color: #f0a030;")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)

        msg = QLabel(
            f"Your <b>{plan_type}</b> subscription expires in "
            f"<b>{days_remaining}</b> day{'s' if days_remaining != 1 else ''}.\n\n"
            f"Renew now to avoid interruption."
        )
        msg.setStyleSheet("font-size: 13px; color: #e0e0e8;")
        msg.setWordWrap(True)
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(msg)

        btn_layout = QHBoxLayout()

        renew_btn = QPushButton("Renew Now")
        renew_btn.setObjectName("primary")
        renew_btn.clicked.connect(lambda: webbrowser.open(self.RENEW_URL))
        btn_layout.addWidget(renew_btn)

        later_btn = QPushButton("Remind Me Later")
        later_btn.clicked.connect(self.accept)
        btn_layout.addWidget(later_btn)

        layout.addLayout(btn_layout)


# ═══════════════════════════════════════════════════════════════════════
# License Info Widget (for Settings page)
# ═══════════════════════════════════════════════════════════════════════

class LicenseInfoWidget(QGroupBox):
    """Embeddable widget showing current license status. For use in a Settings page."""

    def __init__(self, validator: LicenseValidator, parent=None):
        super().__init__("License Information", parent)
        self.validator = validator
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        layout = QGridLayout()
        layout.setSpacing(8)

        self.status_label = QLabel("—")
        self.status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(QLabel("Status:"), 0, 0)
        layout.addWidget(self.status_label, 0, 1)

        self.plan_label = QLabel("—")
        layout.addWidget(QLabel("Plan:"), 1, 0)
        layout.addWidget(self.plan_label, 1, 1)

        self.expiry_label = QLabel("—")
        layout.addWidget(QLabel("Expires:"), 2, 0)
        layout.addWidget(self.expiry_label, 2, 1)

        self.days_label = QLabel("—")
        layout.addWidget(QLabel("Days left:"), 3, 0)
        layout.addWidget(self.days_label, 3, 1)

        self.email_label = QLabel("—")
        layout.addWidget(QLabel("Email:"), 4, 0)
        layout.addWidget(self.email_label, 4, 1)

        self.machine_label = QLabel("—")
        self.machine_label.setStyleSheet("font-size: 10px; color: #606880;")
        layout.addWidget(QLabel("Machine:"), 5, 0)
        layout.addWidget(self.machine_label, 5, 1)

        # Deactivate button
        deactivate_btn = QPushButton("Deactivate License")
        deactivate_btn.setStyleSheet("""
            QPushButton {
                background-color: #402025;
                border: 1px solid #804050;
                border-radius: 6px;
                color: #ff8060;
                font-weight: bold;
                padding: 6px 16px;
            }
            QPushButton:hover { background-color: #502535; }
        """)
        deactivate_btn.clicked.connect(self._on_deactivate)
        layout.addWidget(deactivate_btn, 6, 0, 1, 2)

        self.setLayout(layout)

    def refresh(self):
        info = self.validator.current_license
        if not info:
            result = self.validator.check_local_license()
            info = result.license_info

        if info:
            days = self.validator.get_days_remaining(info)
            expired = self.validator.is_expired(info)

            self.status_label.setText("Expired" if expired else "Active")
            self.status_label.setStyleSheet(
                f"font-weight: bold; color: {'#ff6b6b' if expired else '#6ecf6e'};"
            )
            self.plan_label.setText(info.plan_type.capitalize())
            self.expiry_label.setText(info.expiry_date[:10] if info.expiry_date else "—")
            self.days_label.setText(str(days))
            self.email_label.setText(info.user_email or "—")
            mid = info.machine_id or self.validator.get_machine_id()
            self.machine_label.setText(mid[:20] + "...")
        else:
            self.status_label.setText("Not activated")
            self.status_label.setStyleSheet("font-weight: bold; color: #ff8060;")

    def _on_deactivate(self):
        reply = QMessageBox.question(
            self, "Deactivate License",
            "This will remove the license from this machine.\n"
            "You will need to re-enter your key to use MetaOpticsAI.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.validator.deactivate()
            self.refresh()


# ═══════════════════════════════════════════════════════════════════════
# Integration helper
# ═══════════════════════════════════════════════════════════════════════

def run_license_check(validator: LicenseValidator, app: QApplication) -> bool:
    """
    Run the full startup license check flow.
    Returns True if the app should proceed, False if it should exit.

    Call this from main.py before showing the main window:

        from core.license_validator import LicenseValidator
        from gui.license_dialog import run_license_check

        validator = LicenseValidator(
            license_server_url="https://license.yourcompany.com",
            app_version="1.0.0"
        )
        if not run_license_check(validator, app):
            sys.exit(1)
    """

    # Phase 1: Show splash and validate
    splash = LicenseCheckSplash(validator)

    result_holder = [None]

    def on_complete(result):
        result_holder[0] = result

    splash.validation_complete.connect(on_complete)
    splash.show()
    splash.start_validation()
    splash.exec()

    result = result_holder[0]

    if result is None:
        # Dialog was closed without completing
        return False

    # Phase 2: Handle result
    if result.is_valid:
        # Check if expiring soon (< 7 days)
        if result.days_remaining < 7 and result.days_remaining > 0:
            info = result.license_info
            warn = ExpiryWarningDialog(
                result.days_remaining,
                info.plan_type if info else "subscription"
            )
            warn.exec()
        return True

    # Phase 3: Not valid — show activation dialog
    if result.needs_activation or not result.is_valid:
        max_attempts = 3
        for attempt in range(max_attempts):
            dialog = LicenseActivationDialog(
                validator, message=result.message
            )

            activated = [False]

            def on_activated(act_result):
                activated[0] = True

            dialog.activated.connect(on_activated)
            dialog_result = dialog.exec()

            if activated[0]:
                return True

            if dialog_result == QDialog.DialogCode.Rejected:
                # User closed the dialog without activating
                if attempt < max_attempts - 1:
                    retry = QMessageBox.question(
                        None, "License Required",
                        "MetaOpticsAI requires a valid license to run.\n\n"
                        "Would you like to try again?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    )
                    if retry == QMessageBox.StandardButton.No:
                        return False
                    # Update message for retry
                    result = ValidationResult(
                        is_valid=False,
                        message="Please enter a valid license key.",
                        needs_activation=True,
                    )
                else:
                    return False

    return False