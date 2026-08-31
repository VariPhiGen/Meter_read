"""Turns raw OCR text into a meter reading.

The engine itself lives in engines.py; this module owns everything above it -
cleaning, parsing to a number, and the voting that turns a set of noisy reads
into one answer.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from . import engines, logsetup

log = logging.getLogger(__name__)


def get_reader(languages: List[str], gpu: bool, engine: str = "easyocr"):
    """The OCR engine, built once and reused across cameras and cycles."""
    reader = engines.get(engine, list(languages), bool(gpu))
    # Constructing a paddle model tears our handlers off the root logger.
    logsetup.ensure()
    return reader


@dataclass
class ReadResult:
    text: str                      # cleaned digits as read
    value: Optional[float]         # parsed numeric value, None if unparseable
    confidence: float              # mean OCR confidence of the winning read
    votes: int                     # how many frames agreed on this answer
    total_frames: int
    raw_candidates: List[str]      # every per-frame read, for debugging


def detect(reader, img: np.ndarray, allowlist: str, min_conf: float) -> List[tuple]:
    """Raw detections above the confidence floor: [(box, text, conf), ...].

    Kept separate from read_frame so the annotator can draw the same boxes the
    reading was actually built from, rather than re-running OCR with different
    settings and drawing something that never influenced the number.
    """
    return reader.detect(img, allowlist, min_conf)


def _height(box) -> float:
    ys = [float(p[1]) for p in box]
    return max(ys) - min(ys)


def tallest_only(detections: List[tuple], min_height_frac: float) -> List[tuple]:
    """Keep only the main register, discarding the panel's small print.

    The green glass carries more than the reading: "kWh", "kW", "MD", and a
    subscript decimal group. With a digits-only allowlist the recogniser is
    forced to return a digit for those too - "kW" comes back as "6", scored
    0.98, which is *higher* than the real digits score. Confidence therefore
    cannot separate them, but height can: the main register is roughly twice
    the height of everything else on the panel.
    """
    if not detections or min_height_frac <= 0:
        return detections
    tallest = max(_height(b) for b, _t, _c in detections)
    if tallest <= 0:
        return detections
    return [d for d in detections if _height(d[0]) >= min_height_frac * tallest]


def read_frame(
    reader,
    img: np.ndarray,
    allowlist: str,
    min_conf: float,
    min_height_frac: float = 0.0,
) -> Tuple[str, float]:
    """OCR one prepared image and return (concatenated digits, mean confidence)."""
    detections = tallest_only(detect(reader, img, allowlist, min_conf), min_height_frac)
    kept = [(t, c) for _box, t, c in detections]
    if not kept:
        return "", 0.0

    # Meter displays read left to right; both engines return in that order.
    text = "".join(t.strip() for t, _ in kept)
    mean_conf = float(np.mean([c for _, c in kept]))
    return _clean(text), mean_conf


def _clean(text: str) -> str:
    """Keep digits and a single decimal point; drop everything else."""
    text = re.sub(r"[^0-9.]", "", text)
    if text.count(".") > 1:  # keep the last dot, meters rarely have two
        head, _, tail = text.rpartition(".")
        text = head.replace(".", "") + "." + tail
    return text.strip(".")


def parse_value(text: str, decimals: Optional[int]) -> Optional[float]:
    """Turn the cleaned string into a number.

    If `decimals` is configured we trust the meter's known format over whatever
    dot EasyOCR thought it saw - a smudge read as a decimal point is a common
    failure and would otherwise change the reading by orders of magnitude.
    """
    if not text:
        return None

    digits = text.replace(".", "")
    if not digits.isdigit():
        return None

    try:
        if decimals is None:
            return float(text)
        if decimals == 0:
            return float(digits)
        if len(digits) <= decimals:
            digits = digits.zfill(decimals + 1)
        return float(f"{digits[:-decimals]}.{digits[-decimals:]}")
    except ValueError:
        return None


def read_meter(
    reader,
    images: List[np.ndarray],
    allowlist: str,
    min_conf: float,
    decimals: Optional[int] = None,
    min_height_frac: float = 0.0,
) -> ReadResult:
    """OCR every frame independently and let the most common answer win.

    A single frame can be spoiled by glare, a flicker or motion blur. Reading
    several and voting turns those one-off errors into outvoted minorities.
    """
    candidates: List[str] = []
    confidences: dict = {}

    for img in images:
        text, conf = read_frame(reader, img, allowlist, min_conf, min_height_frac)
        candidates.append(text)
        if text:
            confidences.setdefault(text, []).append(conf)

    valid = [c for c in candidates if c]
    if not valid:
        return ReadResult("", None, 0.0, 0, len(images), candidates)

    winner, votes = Counter(valid).most_common(1)[0]
    mean_conf = float(np.mean(confidences.get(winner, [0.0])))

    return ReadResult(
        text=winner,
        value=parse_value(winner, decimals),
        confidence=mean_conf,
        votes=votes,
        total_frames=len(images),
        raw_candidates=candidates,
    )
