"""Pluggable OCR backends behind one interface.

Both engines are wrapped so they return the same shape - [(box, text, conf)] -
which is what lets `compare` score them against the same ground truth on the
same crops. Neither is assumed to be better; the numbers decide.

Engines are expensive to construct (each loads detection + recognition models),
so instances are cached per (engine, languages, gpu).
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

log = logging.getLogger(__name__)

Detection = Tuple[list, str, float]

_cache: Dict[tuple, "Engine"] = {}


class Engine:
    name = "base"

    def detect(self, img: np.ndarray, allowlist: str, min_conf: float) -> List[Detection]:
        raise NotImplementedError


class EasyOCREngine(Engine):
    """EasyOCR. Strong on printed and mechanical/odometer digits.

    Known weakness: LCD/LED seven-segment glyphs. The segment gaps break the
    character shapes it was trained on, and it fails *confidently* - a wrong
    digit at 0.98 is normal, so confidence cannot be used to detect the error.
    """

    name = "easyocr"

    def __init__(self, languages: List[str], gpu: bool):
        import easyocr
        import torch

        use_gpu = bool(gpu) and torch.cuda.is_available()
        if gpu and not use_gpu:
            log.warning("GPU requested but CUDA is unavailable - falling back to CPU")

        log.info("Loading EasyOCR models (gpu=%s)...", use_gpu)
        self._reader = easyocr.Reader(languages, gpu=use_gpu)
        log.info("EasyOCR ready")

    def detect(self, img, allowlist, min_conf):
        out = self._reader.readtext(
            img, allowlist=allowlist or None, detail=1, paragraph=False
        )
        return [(b, t, float(c)) for b, t, c in out if c >= min_conf and t.strip()]


class PaddleOCREngine(Engine):
    """PaddleOCR. Stronger than EasyOCR on segmented/stylised digits, and much
    faster: 2.0s vs 23.4s per image on CPU over this project's test set.

    It has no allowlist, so filtering to digits happens afterwards in
    ocr._clean rather than inside the recogniser. That is a disadvantage on a
    cluttered frame and an advantage on a tight ROI, where forcing a
    digits-only vocabulary is what makes EasyOCR read "kW" as "6".

    Two incompatible generations are supported, because they ship different
    models and the newer one is worth having:

      paddleocr 2.x - .ocr(img, cls=True); lang="en" pulls a PP-OCRv3
                      *detector* with a PP-OCRv4 recogniser (there is no
                      en_PP-OCRv4_det), and detection was the weaker half.
      paddleocr 3.x - .predict(img); PP-OCRv5 for both halves.
    """

    name = "paddleocr"

    def __init__(self, languages: List[str], gpu: bool):
        from paddleocr import PaddleOCR
        import paddleocr as _pkg

        lang = "en" if not languages else str(languages[0])
        self._major = int(str(getattr(_pkg, "__version__", "2")).split(".")[0])

        use_gpu = bool(gpu) and self._cuda_available()
        if gpu and not use_gpu:
            log.warning("GPU requested but paddle reports no usable CUDA device "
                        "- falling back to CPU")

        if self._major >= 3:
            log.info("Loading PaddleOCR %s / PP-OCRv5 (gpu=%s)...",
                     getattr(_pkg, "__version__", "3.x"), use_gpu)
            # The document orientation/unwarping stages are for scanned pages.
            # On a cropped meter panel they cost time and can rotate the crop.
            self._reader = PaddleOCR(
                ocr_version="PP-OCRv5",
                lang=lang,
                device="gpu:0" if use_gpu else "cpu",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        else:
            log.info("Loading PaddleOCR %s / PP-OCRv4 rec + v3 det (gpu=%s)...",
                     getattr(_pkg, "__version__", "2.x"), use_gpu)
            self._reader = PaddleOCR(
                use_angle_cls=True, lang=lang, show_log=False, use_gpu=use_gpu
            )
        log.info("PaddleOCR ready")

    @staticmethod
    def _cuda_available() -> bool:
        """True only if this paddle build can actually reach a GPU.

        The CPU wheel and the GPU wheel are different packages with the same
        import name, so `gpu: true` against a CPU build must degrade quietly
        rather than raise halfway through a scheduled run.
        """
        try:
            import paddle

            return bool(paddle.is_compiled_with_cuda()) and                 paddle.device.cuda.device_count() > 0
        except Exception:  # noqa: BLE001 - any probe failure means "no GPU"
            return False

    def detect(self, img, allowlist, min_conf):
        # Both generations want 3-channel input; the preprocessing chain often
        # hands back a single-channel image.
        import cv2

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if self._major >= 3:
            return self._detect_v3(img, min_conf)
        return self._detect_v2(img, min_conf)

    def _detect_v3(self, img, min_conf) -> List[Detection]:
        result = self._reader.predict(img)
        if not result:
            return []

        page = result[0]
        texts = page.get("rec_texts") or []
        scores = page.get("rec_scores") or []
        polys = page.get("dt_polys") or []

        out: List[Detection] = []
        for i, text in enumerate(texts):
            conf = float(scores[i]) if i < len(scores) else 0.0
            if conf < min_conf or not str(text).strip():
                continue
            box = polys[i] if i < len(polys) else [[0, 0], [0, 0], [0, 0], [0, 0]]
            out.append(([[float(x), float(y)] for x, y in np.asarray(box)],
                        str(text), conf))
        return out

    def _detect_v2(self, img, min_conf) -> List[Detection]:
        result = self._reader.ocr(img, cls=True)
        if not result or result[0] is None:
            return []
        out: List[Detection] = []
        for box, (text, conf) in result[0]:
            if conf >= min_conf and str(text).strip():
                out.append((box, str(text), float(conf)))
        return out


_ENGINES = {"easyocr": EasyOCREngine, "paddleocr": PaddleOCREngine}


def available() -> List[str]:
    return sorted(_ENGINES)


def get(name: str, languages: List[str], gpu: bool) -> Engine:
    """Build the named engine once, then hand back the same instance."""
    name = (name or "easyocr").strip().lower()
    if name not in _ENGINES:
        raise ValueError(
            f"Unknown OCR engine '{name}'. Available: {', '.join(available())}"
        )

    key = (name, tuple(languages), bool(gpu))
    if key not in _cache:
        try:
            _cache[key] = _ENGINES[name](languages, gpu)
        except ImportError as exc:
            # Only the `both` target ships paddleocr; the default image is
            # easyocr-only to keep it ~1.5GB smaller.
            target = "slim" if name == "easyocr" else "full"
            raise ImportError(
                f"OCR engine '{name}' is not installed in this image: {exc}. "
                f"Rebuild with:  docker build --target {target} -t meter-ocr:{target} ."
            ) from exc
    return _cache[key]
