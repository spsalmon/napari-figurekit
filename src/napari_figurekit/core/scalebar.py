"""Scale-bar geometry and drawing into RGB output pixels.

The exported scale bar is exact in output pixels::

    bar_px = round(length_in_pixel_unit / pixel_size) * upscale

The label always shows the length the user typed (e.g. "10 µm"). If
``length / pixel_size`` is not an integer, the bar is rounded to whole source pixels and
the represented length is reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from .units import convert, display_unit, format_length, round_half_up

logger = logging.getLogger(__name__)

POSITIONS: tuple[str, ...] = ("bottom_right", "bottom_left", "top_right", "top_left")

#: Relative rounding error above which the UI shows the bar report in red.
WARN_REL_ERROR = 0.005


@dataclass(frozen=True)
class BarGeometry:
    """Result of :func:`compute_bar`.

    Attributes
    ----------
    length, unit : float, str
        Length typed by the user and its unit (``"nm" | "um" | "mm"``).
    exact_px : float
        ``length / pixel_size`` in source pixels (not rounded).
    source_px : int
        Bar length in source pixels (``round(exact_px)``).
    bar_px : int
        Bar length in output pixels (``source_px * upscale``).
    represented : float
        Length actually represented by ``source_px`` pixels, in ``unit``.
    upscale : int
        Integer upscale factor.
    """

    length: float
    unit: str
    exact_px: float
    source_px: int
    bar_px: int
    represented: float
    upscale: int

    @property
    def rel_error(self) -> float:
        return abs(self.represented - self.length) / self.length

    @property
    def is_exact(self) -> bool:
        return abs(self.exact_px - self.source_px) < 1e-6

    @property
    def label(self) -> str:
        """Automatic label: the typed length, e.g. ``"10 µm"``."""
        return format_length(self.length, self.unit)

    def report(self) -> str:
        """E.g. ``"10 µm = 38.46 px → 38 px (9.88 µm)"``."""
        text = (
            f"{self.label} = {self.exact_px:.2f} px → {self.source_px} px "
            f"({self.represented:.4g} {display_unit(self.unit)})"
        )
        if self.upscale != 1:
            text += f"; ×{self.upscale} = {self.bar_px} output px"
        return text

    def warning(self) -> str | None:
        """Non-blocking warning when the bar had to be rounded, else ``None``."""
        if self.is_exact:
            return None
        return (
            f"Scale bar rounded to whole pixels: bar = {self.source_px} px = "
            f"{self.represented:.4g} {display_unit(self.unit)} (label shows {self.label})"
        )


def compute_bar(
    length: float, length_unit: str, pixel_size: float, pixel_unit: str, upscale: int = 1
) -> BarGeometry:
    """Compute the exact scale-bar length in output pixels.

    Parameters
    ----------
    length : float
        Bar length typed by the user, in ``length_unit``.
    length_unit : str
        Unit of the bar (may differ from the pixel unit, e.g. bar in nm, pixels in µm).
    pixel_size : float
        Physical size of one source pixel along X, in ``pixel_unit``.
    pixel_unit : str
        Unit of ``pixel_size``.
    upscale : int
        Integer nearest-neighbour upscale factor of the export.

    Raises
    ------
    ValueError
        If the bar would be shorter than one source pixel.
    """
    if length <= 0 or pixel_size <= 0:
        raise ValueError("scale bar length and pixel size must be > 0")
    exact = convert(length, length_unit, pixel_unit) / pixel_size
    src = round_half_up(exact)
    if src < 1:
        raise ValueError(
            f"scale bar of {format_length(length, length_unit)} is shorter than one pixel"
        )
    represented = convert(src * pixel_size, pixel_unit, length_unit)
    return BarGeometry(
        length=float(length),
        unit=length_unit,
        exact_px=exact,
        source_px=src,
        bar_px=src * int(upscale),
        represented=represented,
        upscale=int(upscale),
    )


def parse_color(color) -> tuple[int, int, int]:
    """Parse ``"#rrggbb"``, a color name, or an RGB(A) tuple (0–255 ints or 0–1 floats)."""
    if isinstance(color, str):
        rgb = ImageColor.getrgb(color)
        return (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    vals = list(color)[:3]
    if all(isinstance(v, float) for v in vals) and all(0.0 <= v <= 1.0 for v in vals):
        return tuple(round_half_up(v * 255) for v in vals)  # type: ignore[return-value]
    return (int(vals[0]), int(vals[1]), int(vals[2]))


def _font_candidates() -> list[str]:
    cands = ["DejaVuSans.ttf", "Arial.ttf", "arial.ttf"]
    try:
        import matplotlib

        cands.append(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"))
    except Exception:  # matplotlib is optional
        pass
    return cands


@lru_cache(maxsize=64)
def load_font(size: int, path: str = "") -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TrueType font at ``size`` px.

    Order: custom ``path`` (if given), ``DejaVuSans.ttf``, ``Arial.ttf``/``arial.ttf``,
    matplotlib's bundled DejaVuSans, then ``ImageFont.load_default(size=size)``.
    """
    size = max(int(size), 1)
    cands = ([path] if path else []) + _font_candidates()
    for cand in cands:
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    if path:
        logger.warning("Could not load font %s; using the default font", path)
    return ImageFont.load_default(size=size)


def _check_fits(w: int, h: int, bar_px: int, block_w: int, block_h: int, margin: int) -> None:
    if block_w + 2 * margin > w or block_h + 2 * margin > h:
        raise ValueError(
            f"scale bar ({bar_px} px wide, block {block_w}×{block_h} px) does not fit in a "
            f"{w}×{h} px image with a {margin} px margin; use a shorter bar, a smaller font "
            "or margin, or a larger upscale"
        )


def check_scale_bar_fits(
    width: int,
    height: int,
    bar_px: int,
    *,
    thickness: int = 8,
    margin: int = 20,
    label: str | None = None,
    font_size: int = 24,
    box: bool = False,
    font_path: str = "",
    **_ignored,
) -> None:
    """Raise ``ValueError`` if :func:`draw_scale_bar` would not fit (without drawing)."""
    thickness, margin = max(int(thickness), 1), max(int(margin), 0)
    tw = th = gap = 0
    if label:
        font = load_font(int(font_size), font_path)
        bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        gap = round_half_up(0.3 * font_size)
    pad = max(round_half_up(0.3 * font_size), 2) if box else 0
    block_w = max(int(bar_px), tw) + 2 * pad
    block_h = thickness + (th + gap if label else 0) + 2 * pad
    _check_fits(int(width), int(height), int(bar_px), block_w, block_h, margin)


@dataclass(frozen=True)
class DrawnBar:
    """Where the bar was drawn (output pixel indices, inclusive start, exclusive end)."""

    x0: int
    y0: int
    x1: int
    y1: int


def draw_scale_bar(
    img: np.ndarray,
    bar_px: int,
    *,
    thickness: int = 8,
    color="#ffffff",
    position: str = "bottom_right",
    margin: int = 20,
    label: str | None = None,
    font_size: int = 24,
    box: bool = False,
    box_color="#000000",
    font_path: str = "",
) -> tuple[np.ndarray, DrawnBar]:
    """Draw a scale bar (and optional label above it) into a copy of ``img``.

    Parameters
    ----------
    img : ndarray
        uint8 RGB image (H, W, 3) in output pixels.
    bar_px : int
        Exact bar length in output pixels (from :func:`compute_bar`).
    thickness, margin, font_size : int
        In output pixels. The margin separates the image border from the bar (or box).
    color, box_color : str or tuple
        Bar/text colour and background box colour.
    position : str
        One of ``bottom_right`` (default), ``bottom_left``, ``top_right``, ``top_left``.
    label : str, optional
        Text centred above the bar, separated by ``0.3 × font_size`` px. None = no label.
    box : bool
        Draw a filled background box behind bar and label.

    Returns
    -------
    (ndarray, DrawnBar)
        The annotated image and the drawn bar rectangle. The bar is a solid, non
        anti-aliased rectangle of exactly ``bar_px × thickness`` pixels.

    Raises
    ------
    ValueError
        If the bar (with label, box and margin) does not fit in the image.
    """
    if position not in POSITIONS:
        raise ValueError(f"unknown scale bar position {position!r}")
    bar_px, thickness, margin = int(bar_px), max(int(thickness), 1), max(int(margin), 0)
    h, w = img.shape[:2]
    pil = Image.fromarray(np.ascontiguousarray(img[..., :3], dtype=np.uint8))
    draw = ImageDraw.Draw(pil)
    rgb = parse_color(color)

    font = load_font(int(font_size), font_path) if label else None
    bbox = draw.textbbox((0, 0), label, font=font) if label else (0, 0, 0, 0)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    gap = round_half_up(0.3 * font_size) if label else 0
    content_w = max(bar_px, tw)
    content_h = thickness + (th + gap if label else 0)
    pad = max(round_half_up(0.3 * font_size), 2) if box else 0
    block_w, block_h = content_w + 2 * pad, content_h + 2 * pad
    _check_fits(w, h, bar_px, block_w, block_h, margin)
    left = position.endswith("left")
    top = position.startswith("top")
    bx = margin if left else w - margin - block_w
    by = margin if top else h - margin - block_h
    cx, cy = bx + pad, by + pad
    bar_x = cx if left else cx + content_w - bar_px
    bar_y = cy + content_h - thickness

    if box:
        draw.rectangle([bx, by, bx + block_w - 1, by + block_h - 1], fill=parse_color(box_color))
    draw.rectangle([bar_x, bar_y, bar_x + bar_px - 1, bar_y + thickness - 1], fill=rgb)
    if label and font is not None:
        tx = bar_x + (bar_px - tw) / 2.0
        tx = min(max(tx, cx), cx + content_w - tw)
        draw.text((round(tx) - bbox[0], cy - bbox[1]), label, font=font, fill=rgb)
    return np.asarray(pil, dtype=np.uint8).copy(), DrawnBar(
        bar_x, bar_y, bar_x + bar_px, bar_y + thickness
    )
