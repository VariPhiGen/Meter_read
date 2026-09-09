"""Loads config.yaml and exposes it as plain dataclasses."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import yaml

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Camera:
    id: str
    name: str
    url: str
    enabled: bool = True
    roi: Optional[List[int]] = None
    rotate: int = 0
    decimals: Optional[int] = None


@dataclass
class Config:
    capture: dict = field(default_factory=dict)
    ocr: dict = field(default_factory=dict)
    preprocess: dict = field(default_factory=dict)
    storage: dict = field(default_factory=dict)
    schedule: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    watch: dict = field(default_factory=dict)
    kafka: dict = field(default_factory=dict)
    cameras: List[Camera] = field(default_factory=list)

    @property
    def csv_path(self) -> Path:
        return resolve(self.storage.get("csv_path", "data/readings.csv"))

    @property
    def snapshot_dir(self) -> Path:
        return resolve(self.storage.get("snapshot_dir", "data/snapshots"))

    @property
    def watch_dir(self) -> Path:
        """The folder images are dropped into. Bind-mounted from the host."""
        return resolve(self.watch.get("dir", "images"))

    @property
    def annotated_dir(self) -> Path:
        return resolve(self.watch.get("annotated_dir", "data/annotated"))

    @property
    def state_path(self) -> Path:
        """Which images have already been read. Lives with the output, not the input."""
        return resolve(self.watch.get("state_file", "data/.watch_state.json"))

    def enabled_cameras(self) -> List[Camera]:
        return [c for c in self.cameras if c.enabled]

    def camera(self, cam_id: str) -> Optional[Camera]:
        return next((c for c in self.cameras if c.id == cam_id), None)


def expand_env(value: Any) -> Any:
    """Replace ${VAR} in a string with the environment's value.

    Exists so a camera URL carrying credentials can live in an untracked .env
    while config.yaml - which is committed, and to a public repo - holds only
    the reference. An unset variable is left as the literal ${VAR} rather than
    silently becoming an empty string, so the failure says what is missing
    instead of surfacing as an unopenable stream.
    """
    if not isinstance(value, str):
        return value
    return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)


def resolve(p: Any) -> Path:
    """Project-relative paths stay relative to the project, not the cwd."""
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def load(path: Optional[Path] = None) -> Config:
    path = Path(path) if path else ROOT / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    cams = []
    for entry in raw.get("cameras") or []:
        cams.append(
            Camera(
                id=str(entry["id"]),
                name=str(entry.get("name", entry["id"])),
                url=expand_env(str(entry["url"])),
                enabled=bool(entry.get("enabled", True)),
                roi=entry.get("roi"),
                rotate=int(entry.get("rotate", 0) or 0),
                decimals=entry.get("decimals"),
            )
        )

    return Config(
        capture=raw.get("capture") or {},
        ocr=raw.get("ocr") or {},
        preprocess=raw.get("preprocess") or {},
        storage=raw.get("storage") or {},
        schedule=raw.get("schedule") or {},
        validation=raw.get("validation") or {},
        watch=raw.get("watch") or {},
        kafka=raw.get("kafka") or {},
        cameras=cams,
    )
