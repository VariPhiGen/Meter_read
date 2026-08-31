"""CSV persistence, snapshot files, and sanity checks against the last reading."""

import csv
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import ROOT

log = logging.getLogger(__name__)

FIELDS = [
    "timestamp_local",
    "timestamp_utc",
    "source",
    "camera_id",
    "camera_name",
    "reading",
    "raw_text",
    "confidence",
    "votes",
    "frames",
    "delta",
    "status",
    "note",
    "snapshot",
]


def append_reading(csv_path: Path, row: dict) -> None:
    """Append one row, writing the header first if the file is new."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists() or csv_path.stat().st_size == 0

    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def last_reading(csv_path: Path, camera_id: str) -> Optional[float]:
    """Most recent successfully parsed value for a camera, for delta checks."""
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as fh:
            rows = [
                r
                for r in csv.DictReader(fh)
                if r.get("camera_id") == camera_id and r.get("reading")
            ]
        for row in reversed(rows):
            try:
                return float(row["reading"])
            except (ValueError, TypeError):
                continue
    except Exception as exc:  # noqa: BLE001 - never let history reading break a capture
        log.warning("Could not read history from CSV: %s", exc)
    return None


def save_snapshot(
    snapshot_dir: Path, camera_id: str, frame: np.ndarray, stamp: datetime
) -> str:
    """Write a JPG of the frame so any reading can be checked by eye later."""
    day_dir = snapshot_dir / stamp.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"{camera_id}_{stamp.strftime('%H%M%S')}.jpg"
    cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    try:
        return str(path.relative_to(ROOT))
    except ValueError:  # snapshot_dir configured outside the project
        return str(path)


def prune_snapshots(snapshot_dir: Path, retention_days: int) -> int:
    """Delete snapshot day-folders older than the retention window."""
    if not retention_days or retention_days <= 0 or not snapshot_dir.exists():
        return 0

    cutoff = time.time() - retention_days * 86400
    removed = 0
    for day_dir in snapshot_dir.iterdir():
        if not day_dir.is_dir():
            continue
        try:
            age = datetime.strptime(day_dir.name, "%Y-%m-%d").timestamp()
        except ValueError:
            continue
        if age < cutoff:
            for f in day_dir.glob("*.jpg"):
                f.unlink(missing_ok=True)
                removed += 1
            try:
                day_dir.rmdir()
            except OSError:
                pass
    if removed:
        log.info("Pruned %d snapshot(s) older than %d days", removed, retention_days)
    return removed


def validate(value: Optional[float], text: str, previous: Optional[float], rules: dict):
    """Return (status, note). status is OK, SUSPECT or FAILED."""
    if value is None or not text:
        return "FAILED", "no readable digits"

    digits = len(text.replace(".", ""))
    dmin = int(rules.get("digits_min", 0) or 0)
    dmax = int(rules.get("digits_max", 0) or 0)
    if dmin and digits < dmin:
        return "SUSPECT", f"only {digits} digits, expected >= {dmin}"
    if dmax and digits > dmax:
        return "SUSPECT", f"{digits} digits, expected <= {dmax}"

    if previous is not None:
        delta = value - previous
        if rules.get("expect_increasing", True) and delta < 0:
            return "SUSPECT", f"reading decreased by {abs(delta):.3f}"
        max_delta = float(rules.get("max_delta_per_hour", 0) or 0)
        if max_delta and delta > max_delta:
            return "SUSPECT", f"jumped {delta:.3f}, above limit {max_delta}"

    return "OK", ""


def now_pair():
    """(local datetime, local iso string, utc iso string)."""
    local = datetime.now()
    utc = datetime.now(timezone.utc)
    return local, local.strftime("%Y-%m-%d %H:%M:%S"), utc.strftime("%Y-%m-%d %H:%M:%S")
