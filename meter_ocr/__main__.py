"""Single entry point for every mode.

    python -m meter_ocr watch        # 24/7: OCR every image dropped in (default)
    python -m meter_ocr image FILE   # read one image and print the result
    python -m meter_ocr compare      # score easyocr vs paddleocr on your images
    python -m meter_ocr publish      # print the Kafka payloads (sends nothing)
    python -m meter_ocr publish --send   # actually publish them
    python -m meter_ocr once         # one RTSP reading from every camera
    python -m meter_ocr schedule     # hourly RTSP loop
    python -m meter_ocr selftest     # synthetic end-to-end check, no camera
    python -m meter_ocr cameras      # camera reachability + what OCR reads now
"""

import argparse
import logging
import sys
from pathlib import Path

from . import config, logsetup

log = logging.getLogger(__name__)


def _roi(text):
    """--roi x,y,w,h  |  --roi auto"""
    if text is None:
        return None
    if str(text).strip().lower() == "auto":
        return "auto"
    parts = [p.strip() for p in str(text).replace(" ", "").split(",")]
    if len(parts) != 4 or not all(p.lstrip("-").isdigit() for p in parts):
        raise argparse.ArgumentTypeError(
            "ROI must be four integers (x,y,w,h) or the word 'auto'")
    return [int(p) for p in parts]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="meter_ocr", description="Meter OCR pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    ap.add_argument("--config", default=None, help="Path to config.yaml")
    ap.add_argument("--engine", default=None, help="Override ocr.engine (easyocr|paddleocr)")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is present")
    sub = ap.add_subparsers(dest="command")

    w = sub.add_parser("watch", help="OCR every image dropped into the watch folder")
    w.add_argument("--dir", default=None, help="Folder to watch (default: watch.dir)")
    w.add_argument("--roi", type=_roi, default=None, help="x,y,w,h, or 'auto' to find the lit panel")
    w.add_argument("--decimals", type=int, default=None)
    w.add_argument("--once", action="store_true", help="Process what is there, then exit")

    i = sub.add_parser("image", help="Read a single image file")
    i.add_argument("path")
    i.add_argument("--roi", type=_roi, default=None)
    i.add_argument("--decimals", type=int, default=None)
    i.add_argument("--no-annotate", action="store_true")

    c = sub.add_parser("compare", help="Score OCR engines against known readings")
    c.add_argument("--dir", default=None, help="Folder of images (default: watch.dir)")
    c.add_argument("--truth", default="truth.csv", help="CSV with image,truth columns")
    c.add_argument("--engines", default="easyocr,paddleocr")
    c.add_argument("--out", default="data/engine_comparison.csv")
    c.add_argument("--roi", type=_roi, default=None)
    c.add_argument("--decimals", type=int, default=None)

    pb = sub.add_parser("publish", help="OCR every image and publish to Kafka")
    pb.add_argument("--dir", default=None, help="Folder of images (default: watch.dir)")
    # Sending is opt-in. Publishing to a shared broker is not undoable - a
    # consumer may already have read the message - so the default prints the
    # payloads and sends nothing.
    pb.add_argument("--send", action="store_true",
                    help="Actually publish. Without this, payloads are printed only.")
    pb.add_argument("--dry-run", action="store_true",
                    help="Deprecated: this is now the default. Use --send to publish.")
    pb.add_argument("--limit", type=int, default=None, help="Only the first N images")
    pb.add_argument("--topic", default=None, help="Override kafka.topic")
    pb.add_argument("--broker", default=None, help="Override kafka.bootstrap_servers")

    o = sub.add_parser("once", help="One RTSP reading from every enabled camera")
    o.add_argument("--camera", action="append", help="Only this camera id (repeatable)")

    s = sub.add_parser("schedule", help="Hourly RTSP loop, forever")
    s.add_argument("--no-initial-run", action="store_true")

    sub.add_parser("selftest", help="Synthetic end-to-end check, no camera needed")

    t = sub.add_parser("cameras", help="Camera reachability and what OCR reads now")
    t.add_argument("--camera", action="append")
    t.add_argument("--save", action="store_true", help="Write a frame per camera")

    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "watch"

    logsetup.setup(command)
    cfg = config.load(args.config)

    if args.engine:
        cfg.ocr["engine"] = args.engine
    if args.cpu:
        cfg.ocr["gpu"] = False

    if command == "watch":
        from . import watch
        if args.dir:
            cfg.watch["dir"] = args.dir
        watch.run(cfg, once=args.once, roi=args.roi, decimals=args.decimals)
        return 0

    if command == "image":
        return _cmd_image(cfg, args)

    if command == "compare":
        from . import compare
        image_dir = Path(args.dir) if args.dir else cfg.watch_dir
        truth = config.resolve(args.truth)
        if not truth.exists():
            log.error("Truth file not found: %s", truth)
            log.error("Create it with two columns:  image,truth")
            return 1
        return compare.run(
            cfg, image_dir, truth,
            [e.strip() for e in args.engines.split(",") if e.strip()],
            config.resolve(args.out), args.roi, args.decimals,
        )

    if command == "publish":
        from . import publish
        if args.topic:
            cfg.kafka["topic"] = args.topic
        if args.broker:
            cfg.kafka["bootstrap_servers"] = args.broker
        return publish.run(
            cfg,
            Path(args.dir) if args.dir else None,
            dry_run=not args.send,
            limit=args.limit,
        )

    if command == "once":
        return _cmd_once(cfg, args)

    if command == "schedule":
        from . import schedule
        return schedule.run(cfg, initial_run=not args.no_initial_run)

    if command == "selftest":
        from . import selftest
        return selftest.run(cfg)

    if command == "cameras":
        from . import cameras
        return cameras.run(cfg, only=args.camera, save=args.save)

    return 1


def _cmd_image(cfg, args) -> int:
    from . import images, ocr, storage

    path = Path(args.path)
    if not path.is_absolute():
        path = config.resolve(args.path)
    if not path.exists():
        log.error("No such image: %s", path)
        return 1

    reader = ocr.get_reader(
        list(cfg.ocr.get("languages", ["en"])),
        bool(cfg.ocr.get("gpu", True)),
        str(cfg.ocr.get("engine", "easyocr")),
    )
    row = images.read_image(
        cfg, path, reader,
        roi=args.roi if args.roi is not None else cfg.watch.get("roi"),
        decimals=args.decimals if args.decimals is not None else cfg.watch.get("decimals"),
        rotate=int(cfg.watch.get("rotate", 0) or 0),
        source_id=str(cfg.watch.get("source_id", "image")),
        annotate_dir=None if args.no_annotate else cfg.annotated_dir,
    )
    storage.append_reading(cfg.csv_path, row)

    print()
    print(f"  image      {row['source']}")
    print(f"  reading    {row['reading'] or '(nothing readable)'}")
    print(f"  raw text   {row['raw_text'] or '-'}")
    print(f"  confidence {row['confidence'] or '-'}")
    print(f"  votes      {row['votes'] or '-'}")
    print(f"  status     {row['status']}" + (f"  ({row['note']})" if row["note"] else ""))
    if row["snapshot"]:
        print(f"  annotated  {row['snapshot']}")
    print()
    return 0 if row["status"] != "FAILED" else 2


def _cmd_once(cfg, args) -> int:
    from . import pipeline

    rows = pipeline.run_cycle(cfg, only=args.camera)
    if not rows:
        return 1

    print()
    print(f"{'CAMERA':<14}{'READING':<16}{'CONF':<8}{'VOTES':<8}STATUS")
    print("-" * 60)
    for r in rows:
        print(
            f"{r['camera_id']:<14}{r['reading'] or '-':<16}"
            f"{r['confidence'] or '-':<8}{r['votes'] or '-':<8}{r['status']}"
            + (f"  ({r['note']})" if r["note"] else "")
        )
    print()
    print(f"Saved to {cfg.csv_path}")
    return 0 if all(r["status"] != "FAILED" for r in rows) else 2


if __name__ == "__main__":
    sys.exit(main())
