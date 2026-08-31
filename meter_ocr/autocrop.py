"""Find the meter's lit LCD panel in a photo, so the ROI does not have to be
fixed in advance.

A wall-mounted camera frames the meter identically every time, so a fixed
`roi: [x, y, w, h]` is ideal there. Handheld photos are different: every shot
is framed slightly differently, and one fixed box either clips the digits on
some frames or is loose enough to let the serial number back in on others.

The lit panel is the only strongly-saturated green object in an otherwise grey
meter, so an HSV mask isolates it with no model and no training. Set
`roi: auto` in config.yaml to use this instead of fixed numbers.
"""

import logging
from typing import List, Optional, Union

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Backlit-LCD green. Wide enough for the yellow-green and blue-green casts
# different backlights and white balances produce.
HUE_LO, HUE_HI = 35, 90
SAT_MIN, VAL_MIN = 60, 60

# The panel must be a believable fraction of the frame - this rejects a stray
# green reflection or an indicator LED being mistaken for the display.
MIN_AREA_FRAC = 0.002
MAX_AREA_FRAC = 0.60


def panel_roi(
    img: np.ndarray,
    pad_x: float = 0.03,
    pad_y: float = 0.02,
) -> Optional[List[int]]:
    """Bounding box of the lit panel as [x, y, w, h], or None if not found.

    The box is padded outwards, because the mask tracks the *backlight* and the
    outermost digit often sits right at its edge - without padding the leading
    digit gets clipped, which silently turns 02.32 into 2.32.

    The padding is deliberately small. Just outside the glass sit the meter's
    printed serial number and a barcode, and they are exactly the kind of long
    digit string that ruins a reading: at 12% padding this crop picked up
    "3779" and "250861" from the meter face and EasyOCR returned 177963288 for
    a display showing 13200.87. 3% clears the leading digit without reaching
    the print.
    """
    if img is None or img.ndim != 3:
        return None

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([HUE_LO, SAT_MIN, VAL_MIN], dtype=np.uint8),
        np.array([HUE_HI, 255, 255], dtype=np.uint8),
    )

    # Close first to bridge the unlit segment gaps inside the panel, then open
    # to drop speckle. Kernel size scales with the image so the same numbers
    # work on an 8MP phone photo and a 2MP camera still.
    k = max(3, int(min(h, w) * 0.004) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k * 2, k * 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = float(h * w)
    best = max(contours, key=cv2.contourArea)
    frac = cv2.contourArea(best) / frame_area
    if not (MIN_AREA_FRAC <= frac <= MAX_AREA_FRAC):
        log.debug("Rejected panel candidate covering %.3f%% of frame", frac * 100)
        return None

    x, y, cw, ch = cv2.boundingRect(best)
    px, py = int(cw * pad_x), int(ch * pad_y)
    x, y = max(0, x - px), max(0, y - py)
    cw, ch = min(w - x, cw + 2 * px), min(h - y, ch + 2 * py)
    return [x, y, cw, ch]


def resolve(
    img: np.ndarray, roi: Union[None, str, List[int]], label: str = ""
) -> Optional[List[int]]:
    """Turn a configured `roi` value into concrete pixels for THIS image.

    null       -> None      (OCR the whole frame)
    [x,y,w,h]  -> unchanged (fixed box, right for a mounted camera)
    "auto"     -> detected  (right for handheld photos)
    """
    if isinstance(roi, str):
        if roi.strip().lower() != "auto":
            log.warning("Unknown roi value %r - reading the whole frame", roi)
            return None
        found = panel_roi(img)
        if found is None:
            log.warning("%sNo lit panel found - falling back to the whole frame",
                        f"[{label}] " if label else "")
        else:
            log.info("%sauto ROI %s", f"[{label}] " if label else "", found)
        return found
    return roi
