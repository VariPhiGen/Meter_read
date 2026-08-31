"""Work out which page a multi-page meter is currently showing.

This meter cycles its display: a kWh energy page, and a kW / MD (maximum
demand) page. Logging both into one column would be meaningless - the kW page's
02.32 next to the kWh page's 13200 looks like the meter reset.

The discriminator is the unit label, read WITHOUT a digits-only allowlist so
letters can come back. On this meter the MD page labels itself "MD" clearly
(PaddleOCR reads it at 0.98) while the kWh page's label is faint, so the rule
that survives contact with reality is "MD present => not the kWh page", with
an unreadable label treated as readable-anyway rather than silently dropped.
"""

import logging
import re
from typing import Optional

import numpy as np

from . import ocr

log = logging.getLogger(__name__)

KWH = "kWh"
KW = "kW"
MD = "MD"


def detect(reader, img: np.ndarray, min_conf: float = 0.25) -> Optional[str]:
    """Which page is on screen: "kWh", "kW", "MD", or None if unreadable."""
    detections = ocr.detect(reader, img, "", min_conf)
    text = re.sub(r"[^a-z]", "", "".join(t for _b, t, _c in detections).lower())

    if "md" in text:
        return MD
    if "kwh" in text:
        return KWH
    if "kw" in text:
        return KW
    return None


def wanted(page: Optional[str], want: Optional[str], on_unknown_read: bool = True) -> bool:
    """Should a reading from this page be recorded?"""
    if not want:
        return True                      # no filtering configured
    if page is None:
        return on_unknown_read           # label unreadable - do not silently drop
    return page.lower() == str(want).lower()
