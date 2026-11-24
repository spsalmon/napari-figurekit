"""Pixel-size helpers and calibration metadata reading (ImageJ / OME TIFF).

Axis order is always napari's: ``(..., [Z,] Y, X)``. Pixel sizes are physical lengths per
data pixel, in the given unit.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .units import convert, try_parse_unit

logger = logging.getLogger(__name__)

TIFF_SUFFIXES = (".tif", ".tiff", ".btf", ".tf8")


@dataclass
class FileCalibration:
    """Calibration read from a file. Sizes are in ``unit`` per pixel."""

    pixel_size_y: float
    pixel_size_x: float
    unit: str
    z_step: float | None = None
    source: str = ""  # "imagej" or "ome"


def default_channel_axis(shape: Sequence[int], max_channels: int = 6) -> int:
    """Guess the channel axis of an array.

    The smallest axis with size ``<= max_channels`` that is not one of the last two axes
    (Y, X). Ties go to the lowest axis index.

    Parameters
    ----------
    shape : sequence of int
        Data shape, napari axis order.

    Returns
    -------
    int
        Axis index, or ``-1`` if no axis qualifies.
    """
    candidates = [(size, i) for i, size in enumerate(shape[:-2]) if size <= max_channels]
    if not candidates:
        return -1
    return min(candidates)[1]


def drop_axis(values: Sequence[float], axis: int) -> tuple[float, ...]:
    """Return ``values`` without entry ``axis`` (used for scale/translate after a split)."""
    vals = list(values)
    if axis < 0:
        axis += len(vals)
    del vals[axis]
    return tuple(vals)


def updated_scale(
    scale: Sequence[float],
    pixel_size_y: float,
    pixel_size_x: float,
    z_step: float | None = None,
) -> tuple[float, ...]:
    """Return a new scale with the last 2 (or 3 when ``z_step`` is given) axes replaced.

    Other axes (e.g. time) keep their scale. ``z_step`` is ignored for 2D data.
    """
    new = [float(s) for s in scale]
    if len(new) < 2:
        raise ValueError("calibration needs at least 2 spatial axes (Y, X)")
    new[-2] = float(pixel_size_y)
    new[-1] = float(pixel_size_x)
    if z_step is not None and len(new) >= 3:
        new[-3] = float(z_step)
    return tuple(new)


def rescaled_translate(
    translate: Sequence[float], old_scale: Sequence[float], new_scale: Sequence[float]
) -> tuple[float, ...]:
    """Scale ``translate`` per axis by ``new_scale / old_scale``.

    Keeps pixel offsets (e.g. of crops relative to their source) unchanged when the pixel
    size changes.
    """
    out = []
    for t, s0, s1 in zip(translate, old_scale, new_scale):
        out.append(float(t) * float(s1) / float(s0) if s0 else float(t))
    return tuple(out)


def spatial_axes(ndim: int, with_z: bool) -> tuple[int, ...]:
    """Indices of the spatial axes: the last 2, or last 3 if ``with_z`` and ``ndim >= 3``."""
    n = 3 if (with_z and ndim >= 3) else 2
    return tuple(range(ndim - n, ndim))


def _rational(value) -> float | None:
    """Convert a TIFF rational tag value (tuple or number) to a float."""
    try:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            num, den = value
            return float(num) / float(den) if den else None
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_ome(ome_xml: str) -> FileCalibration | None:
    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError:
        return None
    pixels = None
    for elem in root.iter():
        if elem.tag.split("}")[-1] == "Pixels":
            pixels = elem
            break
    if pixels is None:
        return None
    sx = pixels.get("PhysicalSizeX")
    sy = pixels.get("PhysicalSizeY")
    if sx is None and sy is None:
        return None
    sx = float(sx if sx is not None else sy)
    sy = float(sy if sy is not None else sx)
    ux = try_parse_unit(pixels.get("PhysicalSizeXUnit", "µm"))
    uy = try_parse_unit(pixels.get("PhysicalSizeYUnit", pixels.get("PhysicalSizeXUnit", "µm")))
    if ux is None or uy is None:
        return None
    sy = convert(sy, uy, ux)
    z = pixels.get("PhysicalSizeZ")
    z_step = None
    if z is not None:
        uz = try_parse_unit(pixels.get("PhysicalSizeZUnit", "µm"))
        if uz is not None:
            z_step = convert(float(z), uz, ux)
    return FileCalibration(sy, sx, ux, z_step, "ome")


def _read_imagej(tif) -> FileCalibration | None:
    meta = tif.imagej_metadata or {}
    page = tif.pages.first if hasattr(tif.pages, "first") else tif.pages[0]
    tags = page.tags
    xres = _rational(tags["XResolution"].value) if "XResolution" in tags else None
    yres = _rational(tags["YResolution"].value) if "YResolution" in tags else None
    unit = try_parse_unit(meta.get("unit"))
    if unit is None or not xres:
        return None
    yres = yres or xres
    z = meta.get("spacing")
    return FileCalibration(
        pixel_size_y=1.0 / yres,
        pixel_size_x=1.0 / xres,
        unit=unit,
        z_step=float(z) if z is not None else None,
        source="imagej",
    )


def read_tiff_calibration(path: str | Path) -> FileCalibration | None:
    """Read the pixel size from an ImageJ TIFF or an OME-TIFF.

    * ImageJ: ``XResolution``/``YResolution`` are pixels per unit, so the pixel size is
      ``1 / value``; the unit comes from ``imagej_metadata["unit"]`` and the Z step from
      ``imagej_metadata["spacing"]``.
    * OME-TIFF: ``PhysicalSizeX/Y/Z`` with their ``...Unit`` attributes (default µm).

    Returns
    -------
    FileCalibration or None
        ``None`` if the file is not a TIFF or carries no usable calibration.
    """
    import tifffile

    path = Path(path)
    if path.suffix.lower() not in TIFF_SUFFIXES or not path.is_file():
        return None
    with tifffile.TiffFile(path) as tif:
        if tif.is_ome and tif.ome_metadata:
            cal = _read_ome(tif.ome_metadata)
            if cal is not None:
                return cal
        if tif.is_imagej:
            return _read_imagej(tif)
    return None
