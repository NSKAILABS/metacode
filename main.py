#!/usr/bin/env python3
"""
MetaOpticsAI — AI-Automated Photonics Design Platform

Entry point for the application.
Run: python main.py
     python main.py --skip-license   (development mode, skips license check)
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
from gui.main_window import MetaOpticsMainWindow
from gui.theme import DARK_STYLESHEET

# ── Application metadata ──
APP_VERSION = "1.0.0"
LICENSE_SERVER_URL = "http://localhost:8000"


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MetaOpticsAI")
    app.setOrganizationName("NSK AI Labs")

    app.setStyleSheet(DARK_STYLESHEET)

    font = QFont("Segoe UI", 11)
    app.setFont(font)

    skip_license = "--skip-license" in sys.argv

    if not skip_license:
        from core.license_validator import LicenseValidator
        from gui.license_dialog import run_license_check

        validator = LicenseValidator(
            license_server_url=LICENSE_SERVER_URL,
            app_version=APP_VERSION,
        )

        if not run_license_check(validator, app):
            sys.exit(1)

    window = MetaOpticsMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()