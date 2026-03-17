"""Main application entry point."""

import signal
import sys

from PySide6.QtWidgets import QApplication

from open_broadcast.ui.tray import SystemTrayApp
from open_broadcast.pipeline.manager import PipelineManager


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    app.setApplicationName("Open Broadcast")
    app.setQuitOnLastWindowClosed(False)

    manager = PipelineManager()
    tray = SystemTrayApp(manager)
    tray.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
