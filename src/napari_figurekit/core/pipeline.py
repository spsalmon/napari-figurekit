"""Export pipeline: render panels, burn in arrows and scale bar, write files + sidecars.

An :class:`ExportJob` is pure data (2D arrays, colormaps, settings), built in the GUI
thread from the layers. Rendering and writing need no viewer, so they can run in a
worker thread and be unit-tested headless.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .annotations import draw_arrows
from .export import FORMATS, format_filename, unique_path, write_image, write_sidecar
from .render import ChannelSpec, montage, render_panels
from .scalebar import BarGeometry, draw_scale_bar

RESERVED_PANELS = ("merge", "montage")


@dataclass
class ExportJob:
    """Everything needed to render and write a set of panels.

    Attributes
    ----------
    channels : list of ChannelSpec
        One per layer, 2D source-resolution data.
    arrows_px : list of (tail_yx, tip_yx)
        Arrows in *output* pixel coordinates.
    bar : BarGeometry, optional
        Exact scale bar (``None`` = no scale bar).
    bar_label : str, optional
        Label text (``None`` = no label).
    scale_bar, arrow_style : dict
        Settings dicts (``ScaleBarSettings`` / ``ArrowSettings`` fields).
    name_fields : dict
        Values for the file-name pattern (``source``, ``crop``, ``date``).
    sidecar : dict
        Common sidecar content (layers, calibration, versions, ...).
    """

    channels: list[ChannelSpec]
    merge: bool = True
    colored: bool = False
    gray: bool = False
    montage: bool = False
    montage_gap: int = 10
    gap_color: str = "white"
    upscale: int = 1
    dpi: int = 300
    fmt: str = "png"
    folder: str = ""
    pattern: str = "{source}_{crop}_{panel}"
    bar: BarGeometry | None = None
    bar_label: str | None = None
    scale_bar: dict = field(default_factory=dict)
    arrows_px: list = field(default_factory=list)
    arrow_style: dict = field(default_factory=dict)
    name_fields: dict = field(default_factory=dict)
    sidecar: dict = field(default_factory=dict)


def _wants(apply_to: str, is_merge: bool) -> bool:
    return apply_to == "all" or (apply_to == "merge" and is_merge)


def annotate(job: ExportJob, img: np.ndarray, is_merge: bool) -> np.ndarray:
    """Draw arrows, then the scale bar, onto one panel according to the job settings."""
    if job.arrows_px and _wants(job.arrow_style.get("apply_to", "all"), is_merge):
        img = draw_arrows(
            img,
            job.arrows_px,
            color=job.arrow_style.get("color", "#ffffff"),
            width=job.arrow_style.get("width", 4),
            head_length=job.arrow_style.get("head_length", 3.0),
            head_width=job.arrow_style.get("head_width", 2.5),
        )
    if job.bar is not None and _wants(job.scale_bar.get("apply_to", "all"), is_merge):
        sb = job.scale_bar
        img, _ = draw_scale_bar(
            img,
            job.bar.bar_px,
            thickness=sb.get("thickness", 8),
            color=sb.get("color", "#ffffff"),
            position=sb.get("position", "bottom_right"),
            margin=sb.get("margin", 20),
            label=job.bar_label,
            font_size=sb.get("font_size", 24),
            box=sb.get("box", False),
            box_color=sb.get("box_color", "#000000"),
            font_path=sb.get("font_path", ""),
        )
    return img


def render_job(job: ExportJob, panels: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    """Render the requested panels with annotations.

    Parameters
    ----------
    job : ExportJob
    panels : sequence of str, optional
        Restrict to these panel names (e.g. ``["merge"]`` for the preview). Default: the
        job's output checkboxes.

    Returns
    -------
    dict
        ``panel name -> (H, W, 3) uint8``, in the order merge, channels, grays, montage.
    """
    want_merge = job.merge if panels is None else "merge" in panels
    want_montage = job.montage if panels is None else "montage" in panels
    want_colored = job.colored if panels is None else False
    want_gray = job.gray if panels is None else False
    raw = render_panels(
        job.channels,
        merge=want_merge or want_montage,
        colored=want_colored or want_montage,
        gray=want_gray,
        upscale_factor=job.upscale,
    )
    annotated = {name: annotate(job, img, name == "merge") for name, img in raw.items()}
    out: dict[str, np.ndarray] = {}
    if want_merge:
        out["merge"] = annotated["merge"]
    for ch in job.channels:
        if want_colored:
            out[ch.name] = annotated[ch.name]
        if want_gray:
            out[f"{ch.name}_gray"] = annotated[f"{ch.name}_gray"]
    if want_montage:
        gap_rgb = (0, 0, 0) if job.gap_color == "black" else (255, 255, 255)
        parts = [annotated[ch.name] for ch in job.channels] + [annotated["merge"]]
        out["montage"] = montage(parts, job.montage_gap, gap_rgb)
    return out


def write_job(job: ExportJob, images: dict[str, np.ndarray]) -> list[tuple[Path, Path]]:
    """Write every image and its JSON sidecar. Never overwrites (``_1``, ``_2``, ...).

    Returns
    -------
    list of (image_path, sidecar_path)
    """
    if not job.folder:
        raise ValueError("choose an output folder")
    folder = Path(job.folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    suffix = FORMATS[job.fmt]
    reserved: set[Path] = set()
    written: list[tuple[Path, Path]] = []
    fields = dict(job.name_fields)
    fields.setdefault("date", datetime.now().strftime("%Y%m%d"))
    for panel, img in images.items():
        stem = format_filename(job.pattern, **{**fields, "panel": panel})
        path = unique_path(folder, stem, suffix, reserved)
        reserved.add(path)
        write_image(path, img, job.fmt, job.dpi)
        info: dict[str, Any] = dict(job.sidecar)
        info.update(
            panel=panel,
            format=job.fmt,
            dpi=job.dpi,
            upscale=job.upscale,
            output_size_px={"height": int(img.shape[0]), "width": int(img.shape[1])},
        )
        if panel == "montage":
            info["montage"] = {
                "order": [ch.name for ch in job.channels] + ["merge"],
                "gap_px": job.montage_gap,
                "gap_color": job.gap_color,
            }
        info["scale_bar_drawn"] = bool(
            job.bar is not None
            and (
                _wants(job.scale_bar.get("apply_to", "all"), panel == "merge")
                or (panel == "montage" and job.scale_bar.get("apply_to") == "merge")
            )
        )
        written.append((path, write_sidecar(path, info)))
    return written
