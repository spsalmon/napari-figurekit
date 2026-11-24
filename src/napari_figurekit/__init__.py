"""napari-figurekit: publication-ready figure panels from multichannel microscopy images."""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("napari-figurekit")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.1.0"

__all__ = ["__version__"]
