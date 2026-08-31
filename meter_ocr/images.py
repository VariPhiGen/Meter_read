"""Read a meter from a still image file.

The camera path in pipeline.py votes across five consecutive frames. A still
has one frame, so this path votes across preprocessing variants instead (see
preprocess.variants) and is otherwise the same OCR, parsing, validation and CSV
code - a number read from a dropped-in JPG goes through exactly what a number
read from an RTSP stream goes through.
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from . import autocrop, ocr, pages, preprocess, storage
from .config import ROOT, Config

log = logging.getLogger(__name__)

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in SUFFIXES


def load(path: Path, rotate: int = 0) -> np.ndarray:
    """Read an image off disk. cv2.imread returns None rather than raising."""
    # np.fromfile, not cv2.imread(str(path)): imread cannot open paths with
    # non-ASCII characters on Windows and returns None with no explanation.
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"cannot read {path.name}: {exc}") from exc

    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError(f"not a decodable image: {path.name}")

    if rotate in _ROTATIONS:
        img = cv2.rotate(img, _ROTATIONS[rotate])
    return img


def read_image(
    cfg: Config,
    path: Path,
    reader,
    roi: Optional[List[int]] = None,
    decimals: Optional[int] = None,
    rotate: int = 0,
    source_id: str = "image",
    annotate_dir: Optional[Path] = None,
) -> dict:
    """Produce one CSV row for one image file. Never raises."""
    _local_dt, ts_local, ts_utc = storage.now_pair()

    row = {
        "timestamp_local": ts_local,
        "timestamp_utc": ts_utc,
        "source": path.name,
        "camera_id": source_id,
        "camera_name": source_id,
        "reading": "",
        "raw_text": "",
        "confidence": "",
        "votes": "",
        "frames": "",
        "delta": "",
        "status": "FAILED",
        "note": "",
        "snapshot": "",
    }

    try:
        img = load(path, rotate)
    except ValueError as exc:
        log.error("[%s] %s", path.name, exc)
        row["note"] = str(exc)
        return row

    upscale = float(cfg.ocr.get("upscale", 3.0))
    target_w = cfg.ocr.get("target_width")
    # "auto" has to be resolved against this image, not once for the batch:
    # handheld shots frame the panel differently every time.
    roi = autocrop.resolve(img, roi, path.name)
    prepared = preprocess.variants(img, roi, cfg.preprocess, upscale, target_w)
    if not prepared:
        row["note"] = "preprocessing produced no usable image"
        return row

    # Page filtering happens before the digits are read: on a multi-page meter
    # the kW page's number is not the same quantity as the kWh page's.
    want_page = cfg.watch.get("page")
    if want_page:
        page = pages.detect(reader, prepared[0],
                            float(cfg.ocr.get("min_confidence", 0.35)) * 0.7)
        if not pages.wanted(page, want_page,
                            bool(cfg.watch.get("read_unknown_page", True))):
            log.info("[%s] showing the %s page, not %s - skipped",
                     path.name, page, want_page)
            row["status"] = "SKIPPED"
            row["note"] = f"{page} page, wanted {want_page}"
            return row

    result = ocr.read_meter(
        reader,
        prepared,
        allowlist=str(cfg.ocr.get("allowlist", "0123456789.")),
        min_conf=float(cfg.ocr.get("min_confidence", 0.35)),
        decimals=decimals,
        min_height_frac=float(cfg.ocr.get("min_height_frac", 0.0) or 0.0),
    )

    previous = storage.last_reading(cfg.csv_path, source_id)
    status, note = storage.validate(result.value, result.text, previous, cfg.validation)

    row.update(
        {
            "reading": "" if result.value is None else f"{result.value}",
            "raw_text": result.text,
            "confidence": f"{result.confidence:.3f}",
            "votes": f"{result.votes}/{result.total_frames}",
            "frames": str(len(prepared)),
            "delta": ""
            if (result.value is None or previous is None)
            else f"{result.value - previous:.3f}",
            "status": status,
            "note": note,
        }
    )

    if annotate_dir is not None:
        try:
            row["snapshot"] = annotate(
                cfg, img, prepared[0], roi, reader, annotate_dir, path
            )
        except Exception as exc:  # noqa: BLE001 - a drawing failure must not lose the reading
            log.warning("[%s] annotation failed: %s", path.name, exc)

    if status == "OK":
        log.info("[%s] %s (conf %.2f, votes %s)", path.name, row["reading"],
                 result.confidence, row["votes"])
    else:
        log.warning("[%s] %s - %s (candidates: %s)", path.name, status,
                    note or "unreadable", result.raw_candidates)

    return row


def annotate(
    cfg: Config,
    original: np.ndarray,
    prepared: np.ndarray,
    roi: Optional[List[int]],
    reader,
    out_dir: Path,
    src: Path,
) -> str:
    """Write a copy of the crop with every detection boxed and labelled.

    This is the ROI feedback loop: if the box is in the wrong place or the
    digits are too small, it is visible here in one glance, and the fix is the
    `roi` numbers in config.yaml. It replaces the interactive picker, which
    needed a GUI window the container does not have.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    canvas = prepared if prepared.ndim == 3 else cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)
    canvas = canvas.copy()

    detections = ocr.tallest_only(
        ocr.detect(
            reader,
            prepared,
            str(cfg.ocr.get("allowlist", "0123456789.")),
            float(cfg.ocr.get("min_confidence", 0.35)),
        ),
        float(cfg.ocr.get("min_height_frac", 0.0) or 0.0),
    )

    for box, text, conf in detections:
        pts = np.array(box, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, (0, 255, 0), 3)
        x, y = pts[0][0]
        cv2.putText(canvas, f"{text} {conf:.2f}", (int(x), max(24, int(y) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)

    if roi is None:
        cv2.putText(canvas, "no ROI set - reading the whole frame", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    path = out_dir / f"{src.stem}_annotated.jpg"
    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise ValueError("could not encode annotated image")
    buf.tofile(str(path))

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
