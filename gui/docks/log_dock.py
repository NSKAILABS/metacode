"""
Log Dock
========

QDockWidget providing a console-style log output for the application.

Features:
    - Color-coded messages (info, success, warning, error, debug)
    - Timestamps
    - Clear and export functionality
    - Auto-scroll toggle
"""

from datetime import datetime
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QCheckBox, QFileDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QTextCursor

from gui.theme import (
    C_BG_SURFACE, C_TEXT_PRIMARY, C_SUCCESS, C_WARNING, C_ERROR,
    C_ACCENT_SECONDARY, C_TEXT_MUTED
)


class LogDock(QWidget):
    """
    Console-style log output widget.
    
    Displays timestamped, color-coded messages with support for:
        - info: Normal messages (white)
        - success: Success messages (green)
        - warning: Warning messages (amber)
        - error: Error messages (red)
        - debug: Debug messages (muted gray)
    
    Includes Clear, Export, and Auto-scroll controls.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._auto_scroll = True
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Create the dock widget layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {C_BG_SURFACE};
                color: {C_TEXT_PRIMARY};
                border: none;
                border-radius: 4px;
            }}
        """)
        main_layout.addWidget(self.log_text, stretch=1)
        
        # Controls row
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setMaximumWidth(80)
        controls_layout.addWidget(self.clear_btn)
        
        self.export_btn = QPushButton("Export...")
        self.export_btn.setMaximumWidth(80)
        controls_layout.addWidget(self.export_btn)
        
        controls_layout.addStretch()
        
        self.auto_scroll_check = QCheckBox("Auto-scroll")
        self.auto_scroll_check.setChecked(True)
        controls_layout.addWidget(self.auto_scroll_check)
        
        main_layout.addLayout(controls_layout)
    
    def _connect_signals(self):
        """Connect internal signals."""
        self.clear_btn.clicked.connect(self.clear)
        self.export_btn.clicked.connect(self._export_log)
        self.auto_scroll_check.toggled.connect(self._set_auto_scroll)
    
    def _set_auto_scroll(self, enabled: bool):
        """Toggle auto-scroll behavior."""
        self._auto_scroll = enabled
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("[%H:%M:%S]")
    
    def _append_html(self, html: str):
        """Append HTML content and optionally scroll to bottom."""
        self.log_text.append(html)
        
        if self._auto_scroll:
            cursor = self.log_text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.log_text.setTextCursor(cursor)
    
    def append_message(self, message: str, msg_type: str = "info"):
        """
        Append a message to the log.
        
        Args:
            message: The message text
            msg_type: One of 'info', 'success', 'warning', 'error', 'debug'
        """
        colors = {
            'info': C_TEXT_PRIMARY,
            'success': C_SUCCESS,
            'warning': C_WARNING,
            'error': C_ERROR,
            'debug': C_TEXT_MUTED,
        }
        color = colors.get(msg_type, C_TEXT_PRIMARY)
        
        timestamp = self._get_timestamp()
        html = f'<span style="color: {C_TEXT_MUTED};">{timestamp}</span> '
        html += f'<span style="color: {color};">{message}</span>'
        
        self._append_html(html)
    
    def log_info(self, message: str):
        """Log an info message."""
        self.append_message(message, "info")
    
    def log_success(self, message: str):
        """Log a success message."""
        self.append_message(message, "success")
    
    def log_warning(self, message: str):
        """Log a warning message."""
        self.append_message(message, "warning")
    
    def log_error(self, message: str):
        """Log an error message."""
        self.append_message(message, "error")
    
    def log_debug(self, message: str):
        """Log a debug message."""
        self.append_message(message, "debug")
    
    def log_optimization_start(self, params: dict):
        """Log optimization start with parameters summary."""
        self.append_message("━" * 50, "info")
        self.append_message("OPTIMIZATION STARTED", "success")
        
        if 'optimizer' in params:
            self.append_message(f"  Optimizer: {params['optimizer']}", "info")
        if 'learning_rate' in params:
            self.append_message(f"  Learning rate: {params['learning_rate']}", "info")
        if 'max_iterations' in params:
            self.append_message(f"  Max iterations: {params['max_iterations']}", "info")
        
        self.append_message("━" * 50, "info")
    
    def log_iteration(self, iteration: int, loss: float, best_loss: float):
        """Log an optimization iteration."""
        improvement = "" if loss >= best_loss else " ★"
        self.append_message(
            f"Iter {iteration:4d}: Loss = {loss:.6f}, Best = {best_loss:.6f}{improvement}",
            "info"
        )
    
    def log_optimization_complete(
        self, 
        final_loss: float, 
        iterations: int, 
        converged: bool
    ):
        """Log optimization completion."""
        self.append_message("━" * 50, "info")
        
        if converged:
            self.append_message("✓ OPTIMIZATION CONVERGED", "success")
        else:
            self.append_message("OPTIMIZATION COMPLETE", "warning")
        
        self.append_message(f"  Final loss: {final_loss:.6f}", "info")
        self.append_message(f"  Iterations: {iterations}", "info")
        self.append_message("━" * 50, "info")
    
    def clear(self):
        """Clear all log messages."""
        self.log_text.clear()
    
    def _export_log(self):
        """Export log to a text file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Log",
            f"optimization_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                # Get plain text (strip HTML)
                plain_text = self.log_text.toPlainText()
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(plain_text)
                self.log_success(f"Log exported to: {file_path}")
            except Exception as e:
                self.log_error(f"Export failed: {e}")
    
    def get_text(self) -> str:
        """Get all log text as plain text."""
        return self.log_text.toPlainText()
