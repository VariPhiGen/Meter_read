"""Publish meter readings to Kafka.

One message per reading, keyed by meter_id so every reading from one meter
lands on the same partition and therefore stays in order - which matters here,
because a consumer computing consumption from successive readings gets nonsense
if two readings arrive out of sequence.

Payload (one JSON object per message):

    {
      "meter_id": "meter_5917909",
      "captured_at": "2026-08-29T10:00:00+05:30",
      "crop_image_link": "https://storage/.../display.jpg",
      "reading": {"value": 13288.67, "unit": "kWh"},
      "ocr": {"text": "13288.67 kWh", "confidence": 0.94},
      "status": "valid"
    }
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from .config import Config

log = logging.getLogger(__name__)

# The pipeline's internal statuses mapped onto the wire contract.
STATUS_MAP = {"OK": "valid", "SUSPECT": "suspect", "FAILED": "failed"}


def _tz(offset: str):
    """Turn "+05:30" into a fixed-offset tzinfo.

    Kept explicit rather than relying on the container's TZ, so the timestamp
    on the wire is unambiguous wherever the container happens to run.
    """
    try:
        sign = -1 if offset.strip().startswith("-") else 1
        hh, mm = offset.strip().lstrip("+-").split(":")
        return timezone(sign * timedelta(hours=int(hh), minutes=int(mm)))
    except (ValueError, AttributeError):
        log.warning("Bad utc_offset %r - falling back to +05:30", offset)
        return timezone(timedelta(hours=5, minutes=30))


def captured_at(source: Path, tz) -> str:
    """When the photo was taken, from the file mtime.

    For a live camera the capture time is now; for a folder of stills the
    file timestamp is the honest answer, and it keeps a replayed batch in the
    order the photos were actually taken.
    """
    try:
        ts = datetime.fromtimestamp(source.stat().st_mtime, tz)
    except OSError:
        ts = datetime.now(tz)
    return ts.isoformat(timespec="seconds")


def image_link(cfg: Config, row: dict, source: Path) -> str:
    """Where the cropped display image will be fetchable.

    No object storage is wired up yet, so this is composed from a configured
    base URL and the crop filename. Point `image_base_url` at real storage
    once it exists; nothing else in the payload changes.
    """
    base = str(cfg.kafka.get("image_base_url", "")).rstrip("/")
    snapshot = row.get("snapshot") or ""
    name = Path(snapshot).name if snapshot else f"{source.stem}_crop.jpg"
    return f"{base}/{name}" if base else name


def build(cfg: Config, row: dict, source: Path) -> dict:
    """Turn one pipeline row into one Kafka message."""
    tz = _tz(str(cfg.kafka.get("utc_offset", "+05:30")))
    unit = str(cfg.kafka.get("unit", "kWh"))

    value = None
    if row.get("reading") not in (None, ""):
        try:
            value = float(row["reading"])
        except (TypeError, ValueError):
            value = None

    try:
        confidence = round(float(row.get("confidence") or 0.0), 3)
    except (TypeError, ValueError):
        confidence = 0.0

    status = STATUS_MAP.get(str(row.get("status", "")).upper(), "failed")
    if value is None:
        status = "failed"

    text = f"{value} {unit}" if value is not None else (row.get("raw_text") or "")

    return {
        "meter_id": str(cfg.kafka.get("meter_id", "meter_unknown")),
        "captured_at": captured_at(source, tz),
        "crop_image_link": image_link(cfg, row, source),
        "reading": {"value": value, "unit": unit},
        "ocr": {"text": text, "confidence": confidence},
        "status": status,
    }


def make_producer(cfg: Config):
    """Connect to the broker. Raises if it cannot, rather than dropping data."""
    from kafka import KafkaProducer

    servers = str(cfg.kafka.get("bootstrap_servers", "")).split(",")
    servers = [s.strip() for s in servers if s.strip()]
    if not servers:
        raise ValueError("kafka.bootstrap_servers is not set in config.yaml")

    log.info("Connecting to Kafka at %s", ", ".join(servers))
    return KafkaProducer(
        bootstrap_servers=servers,
        client_id=str(cfg.kafka.get("client_id", "meter-ocr")),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        # acks="all": a reading is not considered sent until every in-sync
        # replica has it. A meter reading that silently vanishes is worse than
        # a slow producer.
        acks=cfg.kafka.get("acks", "all"),
        retries=int(cfg.kafka.get("retries", 5)),
        request_timeout_ms=int(float(cfg.kafka.get("timeout_sec", 30)) * 1000),
        max_block_ms=int(float(cfg.kafka.get("timeout_sec", 30)) * 1000),
    )


def send(cfg: Config, messages: List[dict], dry_run: bool = False) -> int:
    """Publish every message. Returns how many the broker acknowledged."""
    topic = str(cfg.kafka.get("topic", "meter-readings"))
    key_by_meter = bool(cfg.kafka.get("key_by_meter", True))

    if dry_run:
        print(json.dumps(messages, indent=2))
        log.info("Dry run - %d message(s) NOT sent to %s", len(messages), topic)
        return 0

    producer = make_producer(cfg)
    sent = 0
    try:
        futures = []
        for msg in messages:
            key = msg["meter_id"] if key_by_meter else None
            futures.append((msg, producer.send(topic, value=msg, key=key)))

        producer.flush(timeout=float(cfg.kafka.get("timeout_sec", 30)))

        for msg, fut in futures:
            try:
                meta = fut.get(timeout=float(cfg.kafka.get("timeout_sec", 30)))
                sent += 1
                log.info("-> %s[%d]@%d  %s  %s %s",
                         meta.topic, meta.partition, meta.offset,
                         msg["captured_at"], msg["reading"]["value"], msg["status"])
            except Exception as exc:  # noqa: BLE001 - report per message, keep going
                log.error("Failed to publish %s: %s", msg.get("captured_at"), exc)
    finally:
        producer.close(timeout=10)

    log.info("Published %d/%d message(s) to topic %s", sent, len(messages), topic)
    return sent


def run(
    cfg: Config,
    image_dir: Optional[Path] = None,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> int:
    """Read every image in the folder, then publish one message per reading."""
    from . import images, ocr, storage

    image_dir = image_dir or cfg.watch_dir
    files = sorted(p for p in image_dir.iterdir() if p.is_file() and images.is_image(p))
    if limit:
        files = files[:limit]
    if not files:
        log.error("No images found in %s", image_dir)
        return 1

    reader = ocr.get_reader(
        list(cfg.ocr.get("languages", ["en"])),
        bool(cfg.ocr.get("gpu", True)),
        str(cfg.ocr.get("engine", "paddleocr")),
    )

    messages = []
    for path in files:
        row = images.read_image(
            cfg, path, reader,
            roi=cfg.watch.get("roi"),
            decimals=cfg.watch.get("decimals"),
            rotate=int(cfg.watch.get("rotate", 0) or 0),
            source_id=str(cfg.watch.get("source_id", "image")),
            annotate_dir=cfg.annotated_dir,
        )
        if row["status"] == "SKIPPED":
            log.info("[%s] %s - not published", path.name, row["note"])
            continue
        storage.append_reading(cfg.csv_path, row)
        messages.append(build(cfg, row, path))

    if not messages:
        log.error("Nothing to publish")
        return 1

    sent = send(cfg, messages, dry_run)
    return 0 if (dry_run or sent == len(messages)) else 2
