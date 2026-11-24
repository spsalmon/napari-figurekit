"""Deterministic RGB compositing (additive), grayscale panels, upscaling and montages.

All functions are pure NumPy. Images are ``(H, W)`` or ``(H, W, 3)`` arrays in output
pixel coordinates (row = Y, column = X). Float RGB images are in ``[0, 1]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Colormap = Any  # napari Colormap (has .map) or a callable v -> (N, 4) RGBA


@dataclass
class ChannelSpec:
    """Everything needed to render one channel, without any viewer.

    Attributes
    ----------
    data : ndarray
        2D (Y, X) slice of the channel.
    contrast_limits : (float, float)
        Values mapped to 0 and 1.
    colormap : object
        napari ``Colormap`` (anything with ``.map(values) -> (N, 4)``) or a callable.
    gamma : float
        Gamma exponent applied after the linear contrast mapping (1 = linear).
    opacity : float
        Multiplies the RGB of the coloured channel (as napari's additive blending).
    name : str
        Channel name used in file names.
    """

    data: np.ndarray
    contrast_limits: tuple[float, float]
    colormap: Colormap
    gamma: float = 1.0
    opacity: float = 1.0
    name: str = "channel"
    info: dict = field(default_factory=dict)


def normalize(x: np.ndarray, contrast_limits: Sequence[float], gamma: float = 1.0) -> np.ndarray:
    """Linear contrast mapping ``clip((x - lo) / (hi - lo), 0, 1) ** gamma`` in float64.

    If ``hi == lo`` the result is 1 where ``x >= hi`` and 0 elsewhere.
    """
    lo, hi = float(contrast_limits[0]), float(contrast_limits[1])
    xf = np.asarray(x, dtype=np.float64)
    if hi == lo:
        v = (xf >= hi).astype(np.float64)
    else:
        v = np.clip((xf - lo) / (hi - lo), 0.0, 1.0)
    if gamma != 1.0:
        v = v**gamma
    return v


def apply_colormap(v: np.ndarray, colormap: Colormap) -> np.ndarray:
    """Map normalised values ``v`` (H, W) to float RGBA (H, W, 4) with a colormap."""
    flat = np.ascontiguousarray(v, dtype=np.float64).ravel()
    if hasattr(colormap, "map"):
        rgba = np.asarray(colormap.map(flat), dtype=np.float64)
    elif callable(colormap):
        rgba = np.asarray(colormap(flat), dtype=np.float64)
    else:
        raise TypeError("colormap must have a .map method or be callable")
    if rgba.shape[-1] == 3:
        rgba = np.concatenate([rgba, np.ones(rgba.shape[:-1] + (1,))], axis=-1)
    return rgba.reshape(v.shape + (4,))


def render_channel(spec: ChannelSpec) -> np.ndarray:
    """Coloured float RGB (H, W, 3) of one channel: colormap(normalize(x)) × opacity."""
    v = normalize(spec.data, spec.contrast_limits, spec.gamma)
    rgba = apply_colormap(v, spec.colormap)
    return rgba[..., :3] * float(spec.opacity)


def render_gray(spec: ChannelSpec) -> np.ndarray:
    """Grayscale float RGB (H, W, 3) of one channel (colormap and opacity are ignored)."""
    v = normalize(spec.data, spec.contrast_limits, spec.gamma)
    return np.repeat(v[..., None], 3, axis=-1)


def composite_additive(rgbs: Sequence[np.ndarray]) -> np.ndarray:
    """Additive blend ``clip(sum(rgb_i), 0, 1)``, as napari's additive blending."""
    if not rgbs:
        raise ValueError("nothing to composite")
    total = np.zeros_like(np.asarray(rgbs[0], dtype=np.float64))
    for rgb in rgbs:
        total += rgb
    return np.clip(total, 0.0, 1.0)


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    """Convert float [0, 1] to uint8 with ``np.round(x * 255)``."""
    return np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def upscale(img: np.ndarray, factor: int) -> np.ndarray:
    """Integer nearest-neighbour upscale of the first two axes with ``np.repeat``.

    Each source pixel becomes a ``factor × factor`` block of identical values; nothing is
    interpolated.
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError("upscale factor must be an integer >= 1")
    if factor == 1:
        return img
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)


def montage(panels: Sequence[np.ndarray], gap: int = 10, gap_color=(255, 255, 255)) -> np.ndarray:
    """Place uint8 RGB panels of equal height side by side with a ``gap`` px separator."""
    if not panels:
        raise ValueError("no panels for the montage")
    h = panels[0].shape[0]
    if any(p.shape[0] != h for p in panels):
        raise ValueError("montage panels must have the same height")
    gap = max(int(gap), 0)
    sep = np.empty((h, gap, 3), dtype=np.uint8)
    sep[...] = np.asarray(gap_color, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for i, p in enumerate(panels):
        if i and gap:
            parts.append(sep)
        parts.append(p)
    return np.concatenate(parts, axis=1)


def slice_indices(
    data_point: Sequence[float], shape: Sequence[int], n_displayed: int = 2
) -> tuple[int, ...]:
    """Indices of the non-displayed (leading) axes for a data-coordinate point.

    ``index = clip(round(coord), 0, size - 1)`` for each of the first ``ndim -
    n_displayed`` axes.
    """
    lead = len(shape) - n_displayed
    out = []
    for axis in range(lead):
        coord = float(data_point[axis]) if axis < len(data_point) else 0.0
        idx = int(np.floor(coord + 0.5))
        out.append(int(min(max(idx, 0), int(shape[axis]) - 1)))
    return tuple(out)


def extract_2d(data, indices: Sequence[int], max_project_axis: int | None = None) -> np.ndarray:
    """Extract a 2D (Y, X) plane from nD data as a NumPy array.

    Parameters
    ----------
    data : array-like (NumPy, dask, ...)
        nD data, last two axes are Y, X.
    indices : sequence of int
        One index per leading axis (from :func:`slice_indices`).
    max_project_axis : int, optional
        Leading axis to max-project instead of indexing (e.g. ``ndim - 3`` for Z).
    """
    index: list = list(indices)
    if max_project_axis is not None:
        index[max_project_axis] = slice(None)
    sub = data[tuple(index) + (slice(None), slice(None))]
    arr = np.asarray(sub)
    if max_project_axis is not None:
        arr = arr.max(axis=0)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D plane, got shape {arr.shape}")
    return arr


def render_panels(
    channels: Sequence[ChannelSpec],
    merge: bool = True,
    colored: bool = False,
    gray: bool = False,
    upscale_factor: int = 1,
) -> dict[str, np.ndarray]:
    """Render the requested panels as upscaled uint8 RGB images (no annotations).

    Returns
    -------
    dict
        Keys: ``"merge"``, ``"<channel name>"``, ``"<channel name>_gray"``, in that order,
        values ``(H * k, W * k, 3)`` uint8 arrays.
    """
    out: dict[str, np.ndarray] = {}
    colored_rgb = [render_channel(c) for c in channels] if (merge or colored) else []
    if merge:
        out["merge"] = upscale(to_uint8(composite_additive(colored_rgb)), upscale_factor)
    for i, c in enumerate(channels):
        if colored:
            out[c.name] = upscale(to_uint8(colored_rgb[i]), upscale_factor)
        if gray:
            out[f"{c.name}_gray"] = upscale(to_uint8(render_gray(c)), upscale_factor)
    return out
