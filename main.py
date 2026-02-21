#!/usr/bin/env python3
"""
MetaOpticsAI — AI-Automated Photonics Design Platform

Entry point for the application.
Run: python main.py
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
from gui.main_window import MetaOpticsMainWindow
from gui.theme import DARK_STYLESHEET


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MetaOpticsAI")
    app.setOrganizationName("NSK AI Labs")

    # Apply dark theme
    app.setStyleSheet(DARK_STYLESHEET)

    # Set default font
    font = QFont("Segoe UI", 11)
    app.setFont(font)

    window = MetaOpticsMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()