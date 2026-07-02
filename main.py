from __future__ import annotations

import os
import sys


def setup_runtime_environment() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    cpu_count = os.cpu_count() or 8

    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_count))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_count))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_count))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_count))


def main() -> int:
    setup_runtime_environment()

    from PySide6.QtWidgets import QApplication
    from ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())