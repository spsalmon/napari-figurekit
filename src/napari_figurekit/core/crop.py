"""Crop-window maths and array cropping.

Coordinate conventions
----------------------
* Data pixel coordinates follow napari: the centre of pixel ``i`` is at coordinate ``i``,
  so pixel ``i`` covers ``[i - 0.5, i + 0.5)``.
* Axis order is napari's ``(..., Y, X)``. Only the last two axes are ever cropped.
* World coordinates are ``data * scale + translate`` (axis-aligned layers only).

Crops are pure slices of the source array: same dtype, no resampling, no normalisation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .units import round_half_up

EDGE_MODES: tuple[str, ...] = ("shift", "clip", "pad")
SIZE_MODES: tuple[str, ...] = ("physical", "pixels")


@dataclass(frozen=True)
class AxisWindow:
    """Crop window along one axis, in source data pixels.

    Attributes
    ----------
    start, stop : int
        Requested window ``[start, stop)`` in source pixels; may extend beyond the image
        only in ``"pad"`` mode.
    src_start, src_stop : int
        Part of the window that lies inside the image (what is actually sliced).
    """

    start: int
    stop: int
    src_start: int
    src_stop: int

    @property
    def size(self) -> int:
        """Size of the cropped output along this axis (including padding)."""
        return self.stop - self.start

    @property
    def pad_before(self) -> int:
        return self.src_start - self.start

    @property
    def pad_after(self) -> int:
        return self.stop - self.src_stop


@dataclass(frozen=True)
class CropWindow:
    """A 2D (Y, X) crop window.

    Attributes
    ----------
    y, x : AxisWindow
        Per-axis windows.
    mode : str
        Edge mode actually used (``"shift" | "clip" | "pad"``).
    fell_back : bool
        True if the requested mode was ``"shift"`` but the window is larger than the image
        on some axis, so ``"clip"`` was used instead.
    """

    y: AxisWindow
    x: AxisWindow
    mode: str
    fell_back: bool = False

    @property
    def origin(self) -> tuple[int, int]:
        """Source pixel (y, x) that becomes pixel (0, 0) of the crop (negative if padded)."""
        return (self.y.start, self.x.start)

    @property
    def shape(self) -> tuple[int, int]:
        """Output (ny, nx), including padding."""
        return (self.y.size, self.x.size)

    @property
    def padding(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """((before_y, after_y), (before_x, after_x)) zero padding in pixels."""
        return ((self.y.pad_before, self.y.pad_after), (self.x.pad_before, self.x.pad_after))

    @property
    def is_padded(self) -> bool:
        return any(v for pair in self.padding for v in pair)


def window_size_px(
    size: float, mode: str, pixel_size_y: float = 1.0, pixel_size_x: float = 1.0
) -> tuple[int, int]:
    """Number of pixels (ny, nx) of a square crop.

    Parameters
    ----------
    size : float
        Side length: physical length (same unit as the pixel sizes) in ``"physical"`` mode,
        pixels in ``"pixels"`` mode.
    mode : str
        ``"physical"``: ``n_y = round(size / pixel_size_y)``, ``n_x = round(size /
        pixel_size_x)`` (physically square, even with anisotropic pixels).
        ``"pixels"``: ``n_y = n_x = round(size)``.
    pixel_size_y, pixel_size_x : float
        Physical size of a pixel along Y and X.

    Returns
    -------
    tuple of int
        ``(n_y, n_x)``, each >= 1.
    """
    if size <= 0:
        raise ValueError("crop size must be > 0")
    if mode == "pixels":
        n = round_half_up(size)
        ny = nx = n
    elif mode == "physical":
        if pixel_size_y <= 0 or pixel_size_x <= 0:
            raise ValueError("pixel sizes must be > 0")
        ny = round_half_up(size / pixel_size_y)
        nx = round_half_up(size / pixel_size_x)
    else:
        raise ValueError(f"unknown size mode {mode!r}")
    if ny < 1 or nx < 1:
        raise ValueError("crop size is smaller than one pixel")
    return ny, nx


def centered_start(center: float, n: int) -> int:
    """First pixel of an ``n``-pixel window centred on data coordinate ``center``.

    Rule: ``start = round_half_up(center - (n - 1) / 2)``, then ``stop = start + n``.

    In napari the centre of pixel ``i`` is at coordinate ``i``, so the window
    ``[start, start + n)`` is centred at ``start + (n - 1) / 2``. The rule picks the window
    whose centre is nearest to ``center``; for odd ``n`` the middle pixel is the pixel that
    contains ``center``. Exact ties (even ``n`` and an integer centre) go to the larger
    start. (The textbook ``round(center - n / 2)`` assumes pixel *corners* at integers and
    would be biased by half a pixel towards the top-left in napari; Python's ``round`` is
    avoided because banker's rounding depends on parity.)

    Examples
    --------
    center = 10,  n = 5 → start 8  (pixels 8..12, centred on 10)
    center = 10.6, n = 5 → start 9  (pixels 9..13, centred on 11, which contains 10.6)
    center = 10,  n = 4 → start 9  (pixels 9..12, centre 10.5)
    center = 9.9, n = 4 → start 8  (pixels 8..11, centre 9.5)
    """
    return round_half_up(center - (n - 1) / 2.0)


def axis_window(center: float, n: int, size: int, mode: str) -> tuple[AxisWindow, bool]:
    """Compute a 1D crop window. Returns ``(window, fell_back)``; see :func:`crop_window`."""
    if mode not in EDGE_MODES:
        raise ValueError(f"unknown edge mode {mode!r}")
    if n < 1:
        raise ValueError("window size must be >= 1")
    if not (-0.5 <= center <= size - 0.5):
        raise ValueError(f"center {center:.2f} is outside the image (size {size})")
    start = centered_start(center, n)
    fell_back = False
    if mode == "shift":
        if n > size:
            mode = "clip"
            fell_back = True
        else:
            start = min(max(start, 0), size - n)
            return AxisWindow(start, start + n, start, start + n), False
    stop = start + n
    src_start, src_stop = max(start, 0), min(stop, size)
    if mode == "clip":
        return AxisWindow(src_start, src_stop, src_start, src_stop), fell_back
    return AxisWindow(start, stop, src_start, src_stop), fell_back  # pad


def crop_window(
    center_yx: Sequence[float],
    n_yx: Sequence[int],
    shape_yx: Sequence[int],
    mode: str = "shift",
) -> CropWindow:
    """Compute the 2D crop window around a centre.

    Parameters
    ----------
    center_yx : (float, float)
        Centre in source data pixel coordinates (pixel centres are integers).
    n_yx : (int, int)
        Window size in pixels (from :func:`window_size_px`).
    shape_yx : (int, int)
        Size of the source image along Y and X.
    mode : str
        Edge handling:

        * ``"shift"`` (default): keep the size, move the window inside the image. If the
          window is larger than the image, fall back to ``"clip"`` (``fell_back=True``).
        * ``"clip"``: keep the centre, cut the window at the image border (smaller crop).
        * ``"pad"``: keep size and centre; the part outside the image is zero-padded.

    Raises
    ------
    ValueError
        If the centre is outside the image or the mode is unknown.
    """
    wy, fy = axis_window(float(center_yx[0]), int(n_yx[0]), int(shape_yx[0]), mode)
    wx, fx = axis_window(float(center_yx[1]), int(n_yx[1]), int(shape_yx[1]), mode)
    fell_back = fy or fx
    used = "clip" if (fell_back and mode == "shift") else mode
    return CropWindow(wy, wx, used, fell_back)


def _is_dask(arr) -> bool:
    return type(arr).__module__.split(".")[0] == "dask"


def crop_array(arr, window: CropWindow):
    """Crop the last two axes of ``arr`` with ``window``; all other axes stay whole.

    Without padding the result is exactly ``arr[..., y0:y1, x0:x1]`` (a view for NumPy,
    still lazy for dask). With padding, the missing area is filled with zeros of the same
    dtype (``dask.array.pad`` for dask arrays).
    """
    sub = arr[..., window.y.src_start : window.y.src_stop, window.x.src_start : window.x.src_stop]
    if not window.is_padded:
        return sub
    pad_width = [(0, 0)] * (sub.ndim - 2) + [tuple(p) for p in window.padding]
    if _is_dask(sub):
        import dask.array as da

        return da.pad(sub, pad_width, mode="constant", constant_values=0)
    return np.pad(np.asarray(sub), pad_width, mode="constant", constant_values=0)


def crop_translate(
    translate: Sequence[float], scale: Sequence[float], origin_yx: Sequence[int]
) -> tuple[float, ...]:
    """Translate of a crop layer so it overlays its source exactly.

    ``translate_new = translate + origin * scale`` on the last two axes; other axes are
    unchanged.
    """
    out = [float(t) for t in translate]
    out[-2] = float(translate[-2]) + int(origin_yx[0]) * float(scale[-2])
    out[-1] = float(translate[-1]) + int(origin_yx[1]) * float(scale[-1])
    return tuple(out)


def window_rectangle_world(
    window: CropWindow, scale_yx: Sequence[float], translate_yx: Sequence[float]
) -> np.ndarray:
    """Corners (4, 2) of the crop window in world (y, x) coordinates.

    The rectangle follows pixel edges: from ``start - 0.5`` to ``stop - 0.5`` in data
    coordinates, so it outlines exactly the cropped pixels.
    """
    sy, sx = float(scale_yx[0]), float(scale_yx[1])
    ty, tx = float(translate_yx[0]), float(translate_yx[1])
    y0 = (window.y.start - 0.5) * sy + ty
    y1 = (window.y.stop - 0.5) * sy + ty
    x0 = (window.x.start - 0.5) * sx + tx
    x1 = (window.x.stop - 0.5) * sx + tx
    return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)


def world_to_data_yx(
    world_yx: Sequence[float], scale_yx: Sequence[float], translate_yx: Sequence[float]
) -> tuple[float, float]:
    """Axis-aligned world → data conversion for (y, x)."""
    return (
        (float(world_yx[0]) - float(translate_yx[0])) / float(scale_yx[0]),
        (float(world_yx[1]) - float(translate_yx[1])) / float(scale_yx[1]),
    )
