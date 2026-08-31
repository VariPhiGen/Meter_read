"""One full cycle: for every enabled camera, capture -> OCR -> validate -> CSV."""

import logging
from typing import List, Optional

from . import capture, ocr, preprocess, storage
from .config import Camera, Config

log = logging.getLogger(__name__)


def read_camera(cfg: Config, cam: Camera, reader) -> dict:
    """Produce one CSV row for one camera. Never raises - failures become rows."""
    local_dt, ts_local, ts_utc = storage.now_pair()

    row = {
        "timestamp_local": ts_local,
        "timestamp_utc": ts_utc,
        "source": "rtsp",
        "camera_id": cam.id,
        "camera_name": cam.name,
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

    cap_cfg = cfg.capture
    try:
        frames = capture.grab_frames_with_retry(
            cam.url,
            count=int(cap_cfg.get("frames_per_reading", 5)),
            warmup=int(cap_cfg.get("warmup_frames", 15)),
            transport=str(cap_cfg.get("rtsp_transport", "tcp")),
            open_timeout_sec=int(cap_cfg.get("open_timeout_sec", 15)),
            rotate=cam.rotate,
            retries=int(cap_cfg.get("retries", 3)),
            retry_delay_sec=int(cap_cfg.get("retry_delay_sec", 10)),
        )
    except Exception as exc:  # noqa: BLE001 - camera down must not stop other cameras
        log.error("[%s] capture failed: %s", cam.id, exc)
        row["note"] = f"capture failed: {exc}"
        return row

    prepared = [
        preprocess.prepare(
            f, cam.roi, cfg.preprocess,
            float(cfg.ocr.get("upscale", 3.0)), cfg.ocr.get("target_width"),
        )
        for f in frames
    ]

    result = ocr.read_meter(
        reader,
        prepared,
        allowlist=str(cfg.ocr.get("allowlist", "0123456789.")),
        min_conf=float(cfg.ocr.get("min_confidence", 0.35)),
        decimals=cam.decimals,
        min_height_frac=float(cfg.ocr.get("min_height_frac", 0.0) or 0.0),
    )

    previous = storage.last_reading(cfg.csv_path, cam.id)
    status, note = storage.validate(result.value, result.text, previous, cfg.validation)

    row.update(
        {
            "reading": "" if result.value is None else f"{result.value}",
            "raw_text": result.text,
            "confidence": f"{result.confidence:.3f}",
            "votes": f"{result.votes}/{result.total_frames}",
            "frames": str(len(frames)),
            "delta": ""
            if (result.value is None or previous is None)
            else f"{result.value - previous:.3f}",
            "status": status,
            "note": note,
        }
    )

    if cfg.storage.get("save_snapshots", True):
        # Snapshot the sharpest frame - that's the one worth auditing against.
        best = max(frames, key=capture.sharpness)
        try:
            row["snapshot"] = storage.save_snapshot(
                cfg.snapshot_dir, cam.id, best, local_dt
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] snapshot failed: %s", cam.id, exc)

    if status == "OK":
        log.info("[%s] %s (conf %.2f, votes %s)", cam.id, row["reading"],
                 result.confidence, row["votes"])
    else:
        log.warning("[%s] %s - %s (candidates: %s)", cam.id, status, note or "unreadable",
                    result.raw_candidates)

    return row


def run_cycle(cfg: Config, only: Optional[List[str]] = None) -> List[dict]:
    """Read every enabled camera once and append the results to the CSV."""
    cams = cfg.enabled_cameras()
    if only:
        cams = [c for c in cams if c.id in only]
    if not cams:
        log.warning("No cameras to read")
        return []

    reader = ocr.get_reader(
        list(cfg.ocr.get("languages", ["en"])),
        bool(cfg.ocr.get("gpu", True)),
        str(cfg.ocr.get("engine", "easyocr")),
    )

    rows = []
    for cam in cams:
        row = read_camera(cfg, cam, reader)
        storage.append_reading(cfg.csv_path, row)
        rows.append(row)

    storage.prune_snapshots(
        cfg.snapshot_dir, int(cfg.storage.get("snapshot_retention_days", 0) or 0)
    )

    ok = sum(1 for r in rows if r["status"] == "OK")
    log.info("Cycle complete: %d/%d OK -> %s", ok, len(rows), cfg.csv_path)
    return rows
