"""Crops the meter display out of the frame and cleans it up for OCR.

OCR accuracy on meters is decided here far more than in the OCR engine itself.
The ROI crop is the single biggest win: it stops EasyOCR reading serial numbers,
labels and background clutter.
"""

from typing import List, Optional

import cv2
import numpy as np


def crop_roi(img: np.ndarray, roi: Optional[List[int]]) -> np.ndarray:
    """roi is [x, y, w, h]. Clamped to the frame so a bad box can't crash us."""
    if not roi or len(roi) != 4:
        return img
    h, w = img.shape[:2]
    x, y, rw, rh = (int(v) for v in roi)
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rw = max(1, min(rw, w - x))
    rh = max(1, min(rh, h - y))
    return img[y : y + rh, x : x + rw]


def scale_for(width: int, upscale: float, target_width: Optional[int]) -> float:
    """How much to resize a crop before OCR.

    `target_width` (when set) wins, and it scales in BOTH directions. That
    matters: a fixed upscale is right for a small crop off a 720p camera, but
    catastrophic for a panel crop out of an 8000px phone photo - 4593px wide
    times 3 is a 13779px image, which is slower by orders of magnitude and no
    more readable. Normalising every crop to roughly the same width makes one
    set of preprocessing numbers work on both.
    """
    if target_width and width > 0:
        return max(0.05, min(8.0, float(target_width) / float(width)))
    return float(upscale or 1.0)


def enhance(
    img: np.ndarray,
    cfg: dict,
    upscale: float = 3.0,
    target_width: Optional[int] = None,
) -> np.ndarray:
    """Apply the preprocessing chain from config to a cropped ROI."""
    out = img

    scale = scale_for(out.shape[1], upscale, target_width)
    if scale != 1.0:
        # INTER_AREA is the correct filter for shrinking; cubic ringing on a
        # downscale would put halos on the segment edges.
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=interp)

    if cfg.get("grayscale", True) and out.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    # Denoise before contrast work, otherwise CLAHE amplifies sensor noise.
    out = cv2.bilateralFilter(out, 9, 75, 75) if out.ndim == 2 else out

    if cfg.get("clahe", True) and out.ndim == 2:
        clahe = cv2.createCLAHE(
            clipLimit=float(cfg.get("clahe_clip", 2.0)), tileGridSize=(8, 8)
        )
        out = clahe.apply(out)

    if cfg.get("sharpen", True):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        out = cv2.filter2D(out, -1, kernel)

    if cfg.get("invert", False):
        out = cv2.bitwise_not(out)

    if cfg.get("threshold", False) and out.ndim == 2:
        out = cv2.adaptiveThreshold(
            out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )

    return out


def prepare(
    img: np.ndarray,
    roi: Optional[List[int]],
    cfg: dict,
    upscale: float = 3.0,
    target_width: Optional[int] = None,
) -> np.ndarray:
    return enhance(crop_roi(img, roi), cfg, upscale, target_width)


def flatfield(img: np.ndarray, kernel: int = 81) -> np.ndarray:
    """Remove uneven illumination (glare) while keeping stroke contrast.

    A morphological CLOSE with a kernel wider than any digit stroke erases the
    dark segments and leaves an estimate of the lighting itself; dividing it
    out flattens the panel. This matters more than any other single step on a
    photographed LCD: glare across the glass puts a brightness gradient over
    the display, so no global threshold fits the whole panel at once - one end
    is crushed to black while the other is still washed out. After flattening,
    lit segments sit around 195-210 and unlit ones around 240-250 *everywhere*
    on the panel, which is what makes them separable at all.
    """
    if img.ndim == 3:
        # The green channel carries a backlit-green LCD best: the digits are
        # darkest there, and it drops the chroma noise the other two add.
        img = img[:, :, 1]
    k = max(3, int(kernel) | 1)
    bg = cv2.morphologyEx(
        img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    flat = cv2.divide(img.astype(np.float32), bg.astype(np.float32) + 1e-6)
    return np.clip(flat * 255.0, 0, 255).astype(np.uint8)


def variants(
    img: np.ndarray,
    roi: Optional[List[int]],
    cfg: dict,
    upscale: float = 3.0,
    target_width: Optional[int] = None,
) -> List[np.ndarray]:
    """Several preprocessing chains over the same crop, for voting.

    A live camera gets robustness for free by reading five consecutive frames
    and voting. A still image has only one frame, so the variation has to come
    from the preprocessing instead: no single chain wins on every meter, and a
    crop that is unreadable under CLAHE is often clean under a hard threshold.
    Each variant votes on the digit string and the majority takes it.
    """
    crop = crop_roi(img, roi)

    chains = [
        dict(cfg),                                    # whatever config asks for
        {**cfg, "invert": not cfg.get("invert", False)},   # light-on-dark flipped
        {**cfg, "threshold": True},                   # hard black/white
    ]

    out: List[np.ndarray] = []
    for chain in chains:
        try:
            out.append(enhance(crop, chain, upscale, target_width))
        except cv2.error:  # a chain that cannot apply to this crop is just skipped
            continue

    # Flat-fielded variants, added last so they vote alongside the others.
    # On a glare-lit LCD these are usually the only legible ones.
    if cfg.get("flatfield", True):
        try:
            scale = scale_for(crop.shape[1], upscale, target_width)
            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            sized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=interp)
            flat = flatfield(sized, int(cfg.get("flatfield_kernel", 81)))
            out.append(flat)
            # and a hard threshold of it, which is what a segment display wants
            out.append(cv2.threshold(
                flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
        except cv2.error:
            pass

    return out
