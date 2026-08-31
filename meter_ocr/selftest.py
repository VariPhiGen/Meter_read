"""End-to-end check with no camera and no real photo required.

Generates a synthetic meter image, pushes it through the real preprocessing,
OCR, parsing, validation and CSV code, and reports what broke. This is what to
run first inside a freshly built image to prove the install works.
"""

import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np

from . import ocr, preprocess, storage
from .config import Config

log = logging.getLogger(__name__)

EXPECTED = "12345.67"


def make_meter_image(text: str = EXPECTED) -> np.ndarray:
    """A dark panel with light digits, roughly like a meter display."""
    img = np.full((160, 640, 3), 35, dtype=np.uint8)
    cv2.rectangle(img, (20, 30), (620, 130), (15, 15, 15), -1)
    cv2.putText(img, text, (55, 105), cv2.FONT_HERSHEY_SIMPLEX,
                2.2, (235, 235, 235), 5, cv2.LINE_AA)
    return img


def run(cfg: Config) -> int:
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  - {detail}" if detail else ""))

    print("\nMeter OCR self-test\n" + "-" * 60)

    # 1. Dependencies
    try:
        import easyocr  # noqa: F401
        import torch
        check("imports", True,
              f"torch {torch.__version__}, cuda={torch.cuda.is_available()}")
    except ImportError as exc:
        check("imports", False, str(exc))
        return _summary(checks)

    # 2. Parsing, independent of any OCR
    check("parse '12345.67' -> 12345.67", ocr.parse_value("12345.67", None) == 12345.67)
    check("parse '1234567' with decimals=2 -> 12345.67",
          ocr.parse_value("1234567", 2) == 12345.67)
    check("parse junk -> None", ocr.parse_value("", None) is None)

    # 3. Validation
    status, _ = storage.validate(100.0, "100.0", 90.0, cfg.validation)
    check("validate: rising reading is OK", status == "OK", status)
    status, note = storage.validate(80.0, "80.0", 90.0, {"expect_increasing": True})
    check("validate: falling reading is SUSPECT", status == "SUSPECT", note)

    # 4. Preprocessing
    img = make_meter_image()
    variants = preprocess.variants(img, None, cfg.preprocess,
                                   float(cfg.ocr.get("upscale", 3.0)),
                                   cfg.ocr.get("target_width"))
    check("preprocessing produces variants", len(variants) > 0, f"{len(variants)} variants")

    # 5. The real OCR path
    engine = str(cfg.ocr.get("engine", "easyocr"))
    try:
        reader = ocr.get_reader(list(cfg.ocr.get("languages", ["en"])),
                                bool(cfg.ocr.get("gpu", True)), engine)
    except (ImportError, ValueError) as exc:
        check(f"load engine '{engine}'", False, str(exc))
        return _summary(checks)
    check(f"load engine '{engine}'", True)

    result = ocr.read_meter(reader, variants,
                            str(cfg.ocr.get("allowlist", "0123456789.")),
                            float(cfg.ocr.get("min_confidence", 0.35)), None)
    check(f"OCR reads the synthetic panel as {EXPECTED}",
          result.text == EXPECTED,
          f"read '{result.text}' (conf {result.confidence:.2f}, votes {result.votes}/{result.total_frames})")

    # 6. CSV round-trip, in a temp dir so the real readings file is untouched
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "readings.csv"
        storage.append_reading(csv_path, {
            "timestamp_local": "2026-01-01 00:00:00", "source": "selftest",
            "camera_id": "selftest", "reading": "12345.67", "status": "OK",
        })
        check("CSV write + read back",
              storage.last_reading(csv_path, "selftest") == 12345.67)

    return _summary(checks)


def _summary(checks) -> int:
    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print("-" * 60)
    print(f"{passed}/{total} checks passed\n")
    if passed < total:
        print("Failed:")
        for name, ok, detail in checks:
            if not ok:
                print(f"  - {name}" + (f": {detail}" if detail else ""))
        print()
    return 0 if passed == total else 1
