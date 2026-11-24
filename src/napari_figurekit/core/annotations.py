"""Arrows: Shapes lines → Vectors conversion, and drawing arrows into RGB output pixels.

Coordinate conventions
----------------------
* Lines/vectors use napari's (…, Y, X) order.
* Output pixel coordinates: integer ``(row, col)`` is the centre of output pixel
  ``[row, col]``. A source data coordinate ``d`` maps to ``(d + 0.5) * upscale - 0.5``
  (source pixel centres land on the centre of their upscaled block).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw

from .scalebar import parse_color


def lines_to_vectors(
    lines: Sequence[np.ndarray],
    scale: Sequence[float] | None = None,
    translate: Sequence[float] | None = None,
) -> np.ndarray:
    """Convert drawn lines (tail → tip) into napari vectors ``[start, end - start]``.

    Parameters
    ----------
    lines : sequence of (k, D) arrays
        Vertices in the Shapes layer's data coordinates; the first vertex is the tail and
        the last the tip (so paths work too).
    scale, translate : sequence of float, optional
        The Shapes layer's scale/translate, to express vectors in world coordinates.

    Returns
    -------
    ndarray
        ``(N, 2, D)`` array of ``[start, direction]`` in world coordinates.
    """
    out = []
    ndim = None
    for line in lines:
        pts = np.asarray(line, dtype=float)
        if pts.ndim != 2 or len(pts) < 2:
            continue
        ndim = pts.shape[1]
        s = np.ones(ndim) if scale is None else np.asarray(scale, float)[-ndim:]
        t = np.zeros(ndim) if translate is None else np.asarray(translate, float)[-ndim:]
        start = pts[0] * s + t
        end = pts[-1] * s + t
        out.append(np.stack([start, end - start]))
    if not out:
        return np.zeros((0, 2, ndim or 2), dtype=float)
    return np.asarray(out, dtype=float)


def data_to_output_px(data_yx: Sequence[float], upscale: int = 1) -> tuple[float, float]:
    """Map source data coordinates (y, x) to output pixel coordinates (row, col)."""
    k = float(upscale)
    return ((float(data_yx[0]) + 0.5) * k - 0.5, (float(data_yx[1]) + 0.5) * k - 0.5)


def arrow_polygons(
    tail_yx: Sequence[float],
    tip_yx: Sequence[float],
    width: float,
    head_length: float = 3.0,
    head_width: float = 2.5,
) -> tuple[tuple[tuple[float, float], tuple[float, float]] | None, list[tuple[float, float]]]:
    """Geometry of one arrow in output pixel coordinates.

    Parameters
    ----------
    tail_yx, tip_yx : (float, float)
        Output pixel coordinates (row, col).
    width : float
        Shaft width in output pixels.
    head_length, head_width : float
        Multiples of ``width``.

    Returns
    -------
    (shaft, head)
        ``shaft`` is ``((x0, y0), (x1, y1))`` in PIL (x, y) order or ``None`` if the arrow
        is shorter than its head; ``head`` is the triangle ``[tip, left, right]`` in (x, y)
        order. The tip vertex is exactly at ``tip_yx``.
    """
    ty, tx = float(tail_yx[0]), float(tail_yx[1])
    py, px = float(tip_yx[0]), float(tip_yx[1])
    dy, dx = py - ty, px - tx
    length = math.hypot(dy, dx)
    if length == 0:
        raise ValueError("arrow has zero length")
    uy, ux = dy / length, dx / length
    hl = head_length * width
    hw = head_width * width
    by, bx = py - uy * hl, px - ux * hl  # centre of the head base
    ny, nx = -ux, uy  # unit normal
    head = [
        (px, py),
        (bx + nx * hw / 2.0, by + ny * hw / 2.0),
        (bx - nx * hw / 2.0, by - ny * hw / 2.0),
    ]
    shaft = None
    if length > hl:
        # Stop the shaft slightly inside the head so the joint has no gap.
        overlap = min(hl * 0.5, 1.0)
        shaft = ((tx, ty), (bx + ux * overlap, by + uy * overlap))
    return shaft, head


def draw_arrows(
    img: np.ndarray,
    arrows: Sequence[tuple[Sequence[float], Sequence[float]]],
    *,
    color="#ffffff",
    width: float = 4,
    head_length: float = 3.0,
    head_width: float = 2.5,
) -> np.ndarray:
    """Draw arrows (line + filled triangular head) into a copy of a uint8 RGB image.

    Parameters
    ----------
    img : ndarray
        (H, W, 3) uint8.
    arrows : sequence of (tail_yx, tip_yx)
        In output pixel coordinates (see :func:`data_to_output_px`).
    color : str or tuple
        Arrow colour.
    width : float
        Shaft width in output pixels; head size is ``head_length × width`` long and
        ``head_width × width`` wide.
    """
    pil = Image.fromarray(np.ascontiguousarray(img[..., :3], dtype=np.uint8))
    draw = ImageDraw.Draw(pil)
    rgb = parse_color(color)
    for tail, tip in arrows:
        try:
            shaft, head = arrow_polygons(tail, tip, width, head_length, head_width)
        except ValueError:
            continue
        if shaft is not None:
            draw.line([shaft[0], shaft[1]], fill=rgb, width=max(int(round(width)), 1))
        draw.polygon(head, fill=rgb)
    return np.asarray(pil, dtype=np.uint8).copy()
