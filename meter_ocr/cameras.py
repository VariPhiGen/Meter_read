"""Check every camera is reachable and see what OCR reads right now.

Writes nothing to the readings CSV - run it freely while tuning an ROI.
"""

import logging

import cv2

from . import capture, ocr, preprocess
from .config import Config, resolve

log = logging.getLogger(__name__)


def run(cfg: Config, only=None, save: bool = False) -> int:
    cams = cfg.enabled_cameras()
    if only:
        cams = [c for c in cams if c.id in only]
    if not cams:
        log.error("No enabled cameras in config.yaml - nothing to test")
        return 1

    reader = None
    failures = 0

    for cam in cams:
        print(f"\n{cam.id} ({cam.name})")
        print("-" * 60)
        try:
            frames = capture.grab_frames(
                cam.url,
                count=1,
                warmup=int(cfg.capture.get("warmup_frames", 15)),
                transport=str(cfg.capture.get("rtsp_transport", "tcp")),
                open_timeout_sec=int(cfg.capture.get("open_timeout_sec", 15)),
                rotate=cam.rotate,
            )
        except Exception as exc:  # noqa: BLE001 - one dead camera must not stop the rest
            print(f"  UNREACHABLE  {exc}")
            failures += 1
            continue

        frame = frames[0]
        h, w = frame.shape[:2]
        focus = capture.sharpness(frame)
        print(f"  reachable    yes")
        print(f"  resolution   {w}x{h}")
        print(f"  focus score  {focus:.0f}" + ("  (blurry - adjust the lens)" if focus < 100 else ""))
        print(f"  roi          {cam.roi or 'not set - OCR sees the whole frame'}")

        if reader is None:
            reader = ocr.get_reader(
                list(cfg.ocr.get("languages", ["en"])),
                bool(cfg.ocr.get("gpu", True)),
                str(cfg.ocr.get("engine", "easyocr")),
            )

        prepared = preprocess.variants(frame, cam.roi, cfg.preprocess,
                                       float(cfg.ocr.get("upscale", 3.0)),
                                       cfg.ocr.get("target_width"))
        result = ocr.read_meter(reader, prepared,
                                str(cfg.ocr.get("allowlist", "0123456789.")),
                                float(cfg.ocr.get("min_confidence", 0.35)),
                                cam.decimals,
                                float(cfg.ocr.get("min_height_frac", 0.0) or 0.0))
        print(f"  reads as     {result.text or '(nothing)'}"
              f"  (conf {result.confidence:.2f}, votes {result.votes}/{result.total_frames})")

        if save:
            path = resolve(f"data/test_{cam.id}_full.jpg")
            path.parent.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                buf.tofile(str(path))
                print(f"  saved        {path}")

    print()
    return 0 if failures == 0 else 2
