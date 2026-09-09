"""Grabs fresh frames from an RTSP camera.

Two things make RTSP stills unreliable, and both are handled here:
  1. OpenCV buffers frames, so the first frame you read is often seconds stale.
     We drop `warmup_frames` before keeping anything.
  2. UDP transport drops packets and produces smeared/torn frames. We force TCP.
"""

import logging
import os
import time
from typing import List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _set_ffmpeg_options(transport: str, open_timeout_sec: int, url: str = "") -> None:
    """FFMPEG options must be set via env var before VideoCapture is created.

    The option set is per-protocol, and passing the wrong one is not ignored -
    FFmpeg rejects the open outright. `rtsp_transport` and `stimeout` belong to
    the RTSP demuxer; handing them to an http/HLS URL fails before a single
    segment is fetched. HLS also has to download a playlist AND a media segment
    before it can probe the codec, so it needs a probe budget that RTSP does not.
    """
    timeout_us = int(open_timeout_sec * 1_000_000)
    if url.lower().startswith("rtsp"):
        opts = (
            f"rtsp_transport;{transport}"
            f"|stimeout;{timeout_us}"
            f"|max_delay;500000"
            f"|buffer_size;1024000"
        )
    else:
        opts = (
            f"rw_timeout;{timeout_us}"
            f"|analyzeduration;10000000"
            f"|probesize;10000000"
        )
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian - higher means better focused."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def grab_frames(
    url: str,
    count: int = 5,
    warmup: int = 15,
    transport: str = "tcp",
    open_timeout_sec: int = 15,
    rotate: int = 0,
) -> List[np.ndarray]:
    """Open the stream, flush stale frames, and return `count` fresh frames."""
    _set_ffmpeg_options(transport, open_timeout_sec, url)

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        # Ask the driver for the smallest buffer it will give us.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not cap.isOpened():
            raise ConnectionError(f"Could not open stream: {_safe(url)}")

        for _ in range(max(0, warmup)):
            cap.grab()

        frames: List[np.ndarray] = []
        misses = 0
        deadline = time.time() + open_timeout_sec + count * 2

        while len(frames) < count and time.time() < deadline:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                misses += 1
                if misses > 30:
                    break
                time.sleep(0.05)
                continue
            if rotate in _ROTATIONS:
                frame = cv2.rotate(frame, _ROTATIONS[rotate])
            frames.append(frame)

        if not frames:
            raise ConnectionError(f"Stream opened but produced no frames: {_safe(url)}")

        return frames
    finally:
        cap.release()


def grab_frames_with_retry(
    url: str,
    count: int = 5,
    warmup: int = 15,
    transport: str = "tcp",
    open_timeout_sec: int = 15,
    rotate: int = 0,
    retries: int = 3,
    retry_delay_sec: int = 10,
) -> List[np.ndarray]:
    last: Optional[Exception] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            return grab_frames(url, count, warmup, transport, open_timeout_sec, rotate)
        except Exception as exc:  # noqa: BLE001 - retry on any stream failure
            last = exc
            log.warning("Capture attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(retry_delay_sec)
    raise ConnectionError(f"All {retries} capture attempts failed: {last}")


def _safe(url: str) -> str:
    """Strip credentials so they never reach a log file."""
    if "@" in url and "//" in url:
        scheme, rest = url.split("//", 1)
        return f"{scheme}//***@{rest.split('@', 1)[1]}"
    return url
