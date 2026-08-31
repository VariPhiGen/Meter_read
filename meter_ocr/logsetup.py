"""Console + rotating file logging, shared by every entry point."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import ROOT


def setup(name: str = "meter_ocr", level: int = logging.INFO) -> None:
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    if root.handlers and getattr(root, "_meter_ocr_configured", False):
        return
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    fh = RotatingFileHandler(
        log_dir / f"{name}.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # These libraries are extremely chatty at INFO level.
    logging.getLogger("easyocr").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    root._meter_ocr_configured = True
    _remember(name, level)


_LAST = {"name": "meter_ocr", "level": logging.INFO}


def _remember(name: str, level: int) -> None:
    _LAST["name"], _LAST["level"] = name, level


def ensure() -> None:
    """Re-attach our handlers if a library has torn them off the root logger.

    paddlex/paddleocr reconfigure the root logger when a model is constructed,
    which silently removes both the console and the file handler. Everything
    after the first OCR call then vanishes - including, in a 24/7 watch, every
    reading that follows. Called after engine construction to put them back.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    root._meter_ocr_configured = False
    setup(_LAST["name"], _LAST["level"])
