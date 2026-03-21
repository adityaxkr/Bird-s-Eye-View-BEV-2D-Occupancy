# logger/custom_logger.py
# ══════════════════════════════════════════════════════
# Custom logger for BEV Occupancy Project
# Logs to both terminal and a log file
# Same pattern as Document Portal CustomLogger
# ══════════════════════════════════════════════════════

import logging
import os
from datetime import datetime
from pathlib import Path


class CustomLogger:
    """
    Creates a logger that writes to:
      1. Terminal   — colored, readable
      2. logs/ dir  — timestamped .log file

    Usage:
        from logger.custom_logger import CustomLogger
        logger = CustomLogger().get_logger(__name__)

        logger.info("Dataset loaded", extra={"samples": 404})
        logger.error("Image failed to load")
    """

    def __init__(self, logs_dir: str = "logs"):
        # ── Create logs directory ───────────────────────
        self.logs_dir = Path(os.getcwd()) / logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # ── Log file name = timestamp ───────────────────
        timestamp          = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
        self.log_file_path = self.logs_dir / f"{timestamp}.log"

    def get_logger(self, name: str) -> logging.Logger:
        """
        Returns a configured logger for the given module name.

        Args:
            name: pass __name__ from the calling module

        Returns:
            logging.Logger instance
        """
        # Use module name as logger name (avoids duplicate handlers)
        logger_name = os.path.basename(name).replace(".py", "")
        logger      = logging.getLogger(logger_name)

        # ── Avoid adding duplicate handlers ────────────
        if logger.handlers:
            return logger

        logger.setLevel(logging.DEBUG)

        # ── Format ─────────────────────────────────────
        fmt = logging.Formatter(
            fmt=(
                "%(asctime)s | %(levelname)-8s | "
                "%(name)s | %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # ── Handler 1: Terminal output ──────────────────
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(fmt)

        # ── Handler 2: File output ──────────────────────
        file_handler = logging.FileHandler(
            self.log_file_path, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        # ── Don't propagate to root logger ──────────────
        logger.propagate = False

        return logger