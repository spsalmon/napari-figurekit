"""Channel helpers: default names and colormaps, splitting along a channel axis."""

from __future__ import annotations

from collections.abc import Sequence

#: Colormaps offered in the channel table (all are napari built-ins).
COLORMAP_CHOICES: tuple[str, ...] = (
    "gray",
    "green",
    "magenta",
    "cyan",
    "yellow",
    "red",
    "blue",
    "bop orange",
    "bop blue",
    "bop purple",
    "inferno",
    "viridis",
)

_CYCLE: tuple[str, ...] = ("green", "magenta", "cyan", "yellow")


def default_colormaps(n_channels: int) -> list[str]:
    """Default colormaps by channel count.

    1 → gray; 2 → green, magenta; 3 → + cyan; 4 → + yellow; > 4 → cycle these four.
    """
    if n_channels <= 0:
        return []
    if n_channels == 1:
        return ["gray"]
    return [_CYCLE[i % len(_CYCLE)] for i in range(n_channels)]


def default_channel_names(layer_name: str, n_channels: int) -> list[str]:
    """Default channel names: ``"<layer> ch<i>"`` with ``i`` starting at 0."""
    return [f"{layer_name} ch{i}" for i in range(n_channels)]


def split_array(data, axis: int) -> list:
    """Split ``data`` along ``axis`` into views (lazy arrays stay lazy).

    Uses basic slicing, so NumPy results are views and dask results stay dask arrays.
    """
    ndim = data.ndim
    if axis < 0:
        axis += ndim
    if not 0 <= axis < ndim:
        raise ValueError(f"channel axis {axis} out of range for {ndim}D data")
    out = []
    for i in range(data.shape[axis]):
        index: list = [slice(None)] * ndim
        index[axis] = i
        out.append(data[tuple(index)])
    return out


def unique_names(names: Sequence[str]) -> list[str]:
    """Make names unique by appending `` [2]``, `` [3]``, ... to duplicates."""
    seen: dict[str, int] = {}
    out = []
    for name in names:
        name = name.strip() or "channel"
        if name in seen:
            seen[name] += 1
            out.append(f"{name} [{seen[name]}]")
        else:
            seen[name] = 1
            out.append(name)
    return out
