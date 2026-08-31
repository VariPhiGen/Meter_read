"""Score OCR engines against known-correct readings, on the same images.

Answers one question with evidence instead of opinion: for *your* meter, does
easyocr or paddleocr actually read the digits? Both engines see byte-identical
crops through the identical preprocessing chain, so the only variable is the
recogniser.

Ground truth is a CSV with columns `image,truth` - the filename as it appears
in the watch folder, and the reading a human can see on the display:

    image,truth
    meter_kwh.jpg,13200
    meter_kw.jpg,02.32
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

from . import autocrop, images, ocr, preprocess
from .config import Config

log = logging.getLogger(__name__)


def load_truth(path: Path) -> Dict[str, str]:
    with open(path, "r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        raise ValueError(f"{path} has no rows")
    missing = {"image", "truth"} - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing column(s): {', '.join(sorted(missing))}")

    return {r["image"].strip(): (r["truth"] or "").strip() for r in rows if r.get("image")}


def run(
    cfg: Config,
    image_dir: Path,
    truth_path: Path,
    engine_names: List[str],
    out_path: Optional[Path] = None,
    roi: Optional[List[int]] = None,
    decimals: Optional[int] = None,
) -> int:
    truth = load_truth(truth_path)
    roi = roi if roi is not None else cfg.watch.get("roi")
    decimals = decimals if decimals is not None else cfg.watch.get("decimals")
    rotate = int(cfg.watch.get("rotate", 0) or 0)
    upscale = float(cfg.ocr.get("upscale", 3.0))
    target_w = cfg.ocr.get("target_width")
    allowlist = str(cfg.ocr.get("allowlist", "0123456789."))
    min_conf = float(cfg.ocr.get("min_confidence", 0.35))
    min_hf = float(cfg.ocr.get("min_height_frac", 0.0) or 0.0)
    langs = list(cfg.ocr.get("languages", ["en"]))
    gpu = bool(cfg.ocr.get("gpu", True))

    # Prepare every crop once; each engine then sees exactly the same input.
    prepared: Dict[str, list] = {}
    for name in truth:
        path = image_dir / name
        if not path.exists():
            log.warning("%s listed in truth file but not found in %s", name, image_dir)
            continue
        try:
            img = images.load(path, rotate)
        except ValueError as exc:
            log.warning("%s: %s", name, exc)
            continue
        this_roi = autocrop.resolve(img, roi, name)
        prepared[name] = preprocess.variants(
            img, this_roi, cfg.preprocess, upscale, target_w)

    if not prepared:
        log.error("No usable images - nothing to compare")
        return 1

    results: Dict[str, Dict[str, tuple]] = {}
    for engine_name in engine_names:
        try:
            reader = ocr.get_reader(langs, gpu, engine_name)
        except (ImportError, ValueError) as exc:
            log.error("Skipping %s: %s", engine_name, exc)
            continue

        results[engine_name] = {}
        for name, variants in prepared.items():
            result = ocr.read_meter(
                reader, variants, allowlist, min_conf, decimals, min_hf)
            expected = truth[name]
            got = result.text
            # Compare digits only: a missing decimal point is a formatting
            # difference, a wrong digit is a wrong reading.
            ok = got.replace(".", "") == expected.replace(".", "") and bool(got)
            results[engine_name][name] = (got, result.confidence, ok)

    if not results:
        log.error("No engine could be loaded")
        return 1

    _report(truth, results, out_path)
    return 0


def _report(truth, results, out_path: Optional[Path]) -> None:
    engines_used = list(results)
    names = sorted(truth)

    width = max((len(n) for n in names), default=10) + 2
    header = f"{'IMAGE':<{width}}{'TRUTH':<12}" + "".join(
        f"{e.upper():<22}" for e in engines_used
    )
    print()
    print(header)
    print("-" * len(header))

    for name in names:
        line = f"{name:<{width}}{truth[name]:<12}"
        for engine in engines_used:
            got, conf, ok = results[engine].get(name, ("-", 0.0, False))
            mark = "OK " if ok else "XX "
            line += f"{mark}{got or '(none)':<12}{conf:.2f}  "
        print(line)

    print()
    print("SCORE")
    for engine in engines_used:
        rows = results[engine]
        hits = sum(1 for v in rows.values() if v[2])
        pct = 100.0 * hits / len(rows) if rows else 0.0
        print(f"  {engine:<12} {hits}/{len(rows)} correct  ({pct:.0f}%)")
    print()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["image", "truth"]
        for engine in engines_used:
            cols += [f"{engine}_read", f"{engine}_conf", f"{engine}_ok"]
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(cols)
            for name in names:
                row = [name, truth[name]]
                for engine in engines_used:
                    got, conf, ok = results[engine].get(name, ("", 0.0, False))
                    row += [got, f"{conf:.3f}", "yes" if ok else "no"]
                writer.writerow(row)
        print(f"Written to {out_path}")
