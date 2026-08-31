"""Watch a folder and OCR every image dropped into it, forever.

This is the container's default mode: `docker run` it once, leave it running,
and drop images into the mounted folder whenever you want another reading.

Files are tracked by (name, size, mtime) in a small state file rather than
being moved or renamed, so the input folder can be mounted read-only and
nothing you drop in is ever modified or deleted. Re-saving an image changes its
mtime, so it is read again - which is what you want when you are re-testing a
tweaked crop.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import images, ocr, storage
from .config import Config

log = logging.getLogger(__name__)


def _key(path: Path) -> str:
    st = path.stat()
    return f"{path.name}:{st.st_size}:{int(st.st_mtime)}"


def _load_state(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not read watch state (%s) - starting fresh", exc)
        return {}


def _save_state(path: Path, state: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=1)
        tmp.replace(path)
    except OSError as exc:
        log.warning("Could not persist watch state: %s", exc)


def _settled(path: Path, min_age_sec: float) -> bool:
    """True once the file has stopped being written to.

    A large JPG copied into the folder appears on disk before it is complete;
    OCR-ing it half-written produces a spurious FAILED row for an image that is
    actually fine.
    """
    try:
        return (time.time() - path.stat().st_mtime) >= min_age_sec
    except OSError:
        return False


def pending(watch_dir: Path, state: Dict[str, str], min_age_sec: float) -> List[Path]:
    if not watch_dir.exists():
        return []
    out = []
    for path in sorted(watch_dir.iterdir()):
        if not path.is_file() or not images.is_image(path):
            continue
        if not _settled(path, min_age_sec):
            continue
        try:
            if state.get(path.name) == _key(path):
                continue
        except OSError:
            continue
        out.append(path)
    return out


def run(
    cfg: Config,
    once: bool = False,
    roi: Optional[List[int]] = None,
    decimals: Optional[int] = None,
) -> int:
    """Loop until interrupted, reading each new image. Returns rows processed."""
    wcfg = cfg.watch
    watch_dir = cfg.watch_dir
    annotate_dir = cfg.annotated_dir if wcfg.get("annotate", True) else None
    poll = float(wcfg.get("poll_seconds", 5) or 5)
    min_age = float(wcfg.get("min_file_age_seconds", 2) or 0)
    source_id = str(wcfg.get("source_id", "image"))

    roi = roi if roi is not None else wcfg.get("roi")
    decimals = decimals if decimals is not None else wcfg.get("decimals")
    rotate = int(wcfg.get("rotate", 0) or 0)

    state_path = cfg.state_path
    state = _load_state(state_path)

    watch_dir.mkdir(parents=True, exist_ok=True)

    reader = ocr.get_reader(
        list(cfg.ocr.get("languages", ["en"])),
        bool(cfg.ocr.get("gpu", True)),
        str(cfg.ocr.get("engine", "easyocr")),
    )

    log.info("Watching %s every %.0fs (engine=%s, roi=%s)",
             watch_dir, poll, cfg.ocr.get("engine", "easyocr"), roi or "whole frame")
    log.info("Readings -> %s", cfg.csv_path)

    processed = 0
    try:
        while True:
            todo = pending(watch_dir, state, min_age)
            for path in todo:
                row = images.read_image(
                    cfg, path, reader,
                    roi=roi, decimals=decimals, rotate=rotate,
                    source_id=source_id, annotate_dir=annotate_dir,
                )
                storage.append_reading(cfg.csv_path, row)
                try:
                    state[path.name] = _key(path)
                except OSError:
                    pass
                processed += 1

            if todo:
                _save_state(state_path, state)
                log.info("%d image(s) read - waiting for more", len(todo))

            if once:
                if not processed:
                    log.info("No new images in %s", watch_dir)
                return processed

            time.sleep(poll)
    except KeyboardInterrupt:
        log.info("Watcher stopped after %d image(s)", processed)
        return processed
