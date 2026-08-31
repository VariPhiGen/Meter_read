"""Runs the RTSP capture cycle on the hour, forever.

Sleeps in short slices so Ctrl+C stays responsive and a host resuming from
suspend re-syncs to the wall clock instead of drifting by however long it slept.
"""

import logging
import time
from datetime import datetime, timedelta

from . import pipeline
from .config import Config

log = logging.getLogger(__name__)


def next_run(now: datetime, every_hours: int, at_minute: int) -> datetime:
    """First slot strictly after `now` that lands on the configured schedule."""
    every_hours = max(1, int(every_hours))
    at_minute = max(0, min(59, int(at_minute)))

    candidate = now.replace(minute=at_minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(hours=1)

    # Align to the every_hours grid measured from midnight.
    while candidate.hour % every_hours != 0:
        candidate += timedelta(hours=1)
    return candidate


def run(cfg: Config, initial_run: bool = True) -> int:
    every = int(cfg.schedule.get("every_hours", 1) or 1)
    minute = int(cfg.schedule.get("at_minute", 0) or 0)
    on_start = bool(cfg.schedule.get("run_on_start", True)) and initial_run

    log.info("Scheduler started - every %dh at :%02d", every, minute)
    log.info("Cameras: %s", ", ".join(c.id for c in cfg.enabled_cameras()) or "none")

    if on_start:
        log.info("Initial run...")
        _safe_cycle(cfg)

    try:
        while True:
            target = next_run(datetime.now(), every, minute)
            log.info("Next reading at %s", target.strftime("%Y-%m-%d %H:%M:%S"))

            while datetime.now() < target:
                time.sleep(min(30, max(1, (target - datetime.now()).total_seconds())))

            _safe_cycle(cfg)
            time.sleep(1)  # never fire twice inside the same minute
    except KeyboardInterrupt:
        log.info("Scheduler stopped")
        return 0


def _safe_cycle(cfg: Config) -> None:
    """A crash in one cycle must not kill a scheduler meant to run for months."""
    try:
        pipeline.run_cycle(cfg)
    except Exception:  # noqa: BLE001
        log.exception("Cycle failed - continuing to next scheduled run")
