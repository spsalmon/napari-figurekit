"""Settings dataclasses shared by all tabs, with dict/JSON (de)serialisation.

Unknown keys are ignored when loading, missing keys keep their defaults, and values of
the wrong type fall back to the default, so old or hand-edited presets still load.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

STATE_VERSION = 1


@dataclass
class CalibrationSettings:
    """Pixel size (physical length per pixel) and where to apply it."""

    pixel_size_y: float = 1.0
    pixel_size_x: float = 1.0
    square: bool = True
    use_z_step: bool = False  # Z is optional: axis -3 may also be time
    z_step: float = 1.0
    unit: str = "um"  # "nm" | "um" | "mm"
    apply_to: str = "all"  # "selected" | "all" | "same_shape"


@dataclass
class CropSettings:
    size: float = 50.0
    size_mode: str = "physical"  # "physical" | "pixels"
    edge_mode: str = "shift"  # "shift" | "clip" | "pad"
    hide_sources: bool = True
    add_outlines: bool = True
    zoom_to_first: bool = True


@dataclass
class ScaleBarSettings:
    visible: bool = True
    length: float = 10.0
    unit: str = "um"
    show_label: bool = True
    custom_label: str = ""
    font_size: int = 24  # px in the exported image
    thickness: int = 8  # px in the exported image
    color: str = "#ffffff"
    position: str = "bottom_right"
    margin: int = 20  # px
    box: bool = False
    box_color: str = "#000000"
    apply_to: str = "all"  # "merge" | "all" | "none"
    font_path: str = ""


@dataclass
class ArrowSettings:
    color: str = "#ffffff"
    width: int = 4  # px in the exported image
    head_length: float = 3.0  # × width
    head_width: float = 2.5  # × width
    apply_to: str = "all"  # "merge" | "all" | "none"


@dataclass
class ExportSettings:
    merge: bool = True
    colored: bool = False
    gray: bool = False
    montage: bool = False
    montage_gap: int = 10
    gap_color: str = "white"  # "white" | "black"
    upscale: int = 1
    dpi: int = 300
    fmt: str = "png"  # "png" | "tiff"
    folder: str = ""
    pattern: str = "{source}_{crop}_{panel}"
    max_projection: bool = False


@dataclass
class FigureKitState:
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    crop: CropSettings = field(default_factory=CropSettings)
    scale_bar: ScaleBarSettings = field(default_factory=ScaleBarSettings)
    arrows: ArrowSettings = field(default_factory=ArrowSettings)
    export: ExportSettings = field(default_factory=ExportSettings)
    version: int = STATE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> FigureKitState:
        data = data if isinstance(data, dict) else {}
        return cls(
            calibration=_section(CalibrationSettings, data.get("calibration")),
            crop=_section(CropSettings, data.get("crop")),
            scale_bar=_section(ScaleBarSettings, data.get("scale_bar")),
            arrows=_section(ArrowSettings, data.get("arrows")),
            export=_section(ExportSettings, data.get("export")),
        )

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> FigureKitState:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def update_from(self, other: FigureKitState) -> None:
        """Copy all settings from ``other`` in place (keeps this object's identity)."""
        for f in fields(self):
            setattr(self, f.name, getattr(other, f.name))


T = TypeVar("T")


def _coerce(value: Any, default: Any) -> Any:
    """Return ``value`` converted to the type of ``default``, or ``default`` if impossible."""
    if isinstance(default, bool):
        return value if isinstance(value, bool) else default
    if isinstance(default, int):
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return int(value)
        return default
    if isinstance(default, float):
        if isinstance(value, bool):
            return default
        return float(value) if isinstance(value, (int, float)) else default
    if isinstance(default, str):
        return value if isinstance(value, str) else default
    return value


def _section(cls: type[T], data: Any) -> T:
    obj = cls()
    if not isinstance(data, dict):
        return obj
    for f in fields(cls):  # type: ignore[arg-type]
        if f.name in data:
            setattr(obj, f.name, _coerce(data[f.name], getattr(obj, f.name)))
    return obj


def state_dir() -> Path:
    """Folder for the auto-saved state (``$NAPARI_FIGUREKIT_HOME`` or ``~/.napari-figurekit``)."""
    env = os.environ.get("NAPARI_FIGUREKIT_HOME")
    return Path(env) if env else Path.home() / ".napari-figurekit"


def last_state_path() -> Path:
    return state_dir() / "last_state.json"
