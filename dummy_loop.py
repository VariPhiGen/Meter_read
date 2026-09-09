"""Publish one synthetic meter reading per hour, on the hour.

Goes through the project's own publish path (publish.build_live ->
publish.send), so what lands on the topic is byte-for-byte the shape the RTSP
pipeline produces. Only the reading itself is invented.

The value climbs rather than repeating: a consumer deriving consumption from
successive readings would read a flat line as a stalled meter, which is not
what a test feed should look like. It is anchored to the wall clock rather
than an in-memory counter, so a restart continues the series instead of
resetting it.
"""
import json
import logging
import os
import time
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dummy")

from meter_ocr import config, publish

EVERY_HOURS = int(os.environ.get("EVERY_HOURS", 1))
AT_MINUTE = int(os.environ.get("AT_MINUTE", 0))
BASE = float(os.environ.get("BASE_READING", 13288.87))
RATE = float(os.environ.get("KWH_PER_HOUR", 0.35))
# Must be in the PAST: reading_now() clamps elapsed hours at zero, so an anchor
# in the future pins every reading to BASE and the series goes flat - which
# reads downstream as a stalled meter, not as a test feed.
ANCHOR = float(os.environ.get("ANCHOR_EPOCH", 1788134400))  # 2026-08-31 00:00 UTC


def reading_now() -> float:
    hours = max(0.0, (time.time() - ANCHOR) / 3600.0)
    return round(BASE + hours * RATE, 2)


def next_slot(now: datetime) -> datetime:
    """First :MM slot strictly after `now` that lands on the hour grid.

    Same rule as schedule.next_run, reimplemented here rather than imported:
    meter_ocr.schedule pulls in pipeline -> capture -> cv2, and this image
    deliberately carries no OCR stack.
    """
    slot = now.replace(minute=AT_MINUTE, second=0, microsecond=0)
    if slot <= now:
        slot += timedelta(hours=1)
    while slot.hour % max(1, EVERY_HOURS) != 0:
        slot += timedelta(hours=1)
    return slot


def main() -> None:
    cfg = config.load("config.yaml")
    # Align against the configured offset, not the container clock: the image
    # runs on UTC, where the top of the hour is :30 in IST - so aligning on
    # local time would fire half an hour off.
    tz = publish._tz(str(cfg.kafka.get("utc_offset", "+05:30")))

    log.info("Dummy publisher: every %dh at :%02d (%s) -> %s topic=%s",
             EVERY_HOURS, AT_MINUTE, cfg.kafka.get("utc_offset", "+05:30"),
             cfg.kafka.get("bootstrap_servers"), cfg.kafka.get("topic"))

    n = 0
    while True:
        target = next_slot(datetime.now(tz))
        log.info("Next send at %s", target.isoformat(timespec="seconds"))
        while True:
            left = (target - datetime.now(tz)).total_seconds()
            if left <= 0:
                break
            time.sleep(min(20, left))  # short slices keep Ctrl+C responsive

        value = reading_now()
        row = {
            "reading": f"{value}",
            "raw_text": f"{value} kWh",
            "confidence": "0.94",
            "status": "OK",
            "snapshot": "data/annotated/dummy_annotated.jpg",
        }
        try:
            msg = publish.build_live(cfg, row)
            sent = publish.send(cfg, [msg])
            n += 1
            log.info("cycle %d: %s -> acked %d/1", n, json.dumps(msg["reading"]), sent)
        except Exception:
            # A broker outage must not kill a loop meant to run for days.
            log.exception("publish failed - retrying at the next slot")
        time.sleep(1)  # never fire twice inside the same minute


if __name__ == "__main__":
    main()
