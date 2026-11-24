"""Calibrated cells3d sample (skimage), as two additive channel layers."""

from __future__ import annotations

from typing import Any


def cells3d_sample() -> list[tuple[Any, dict, str]]:
    """``skimage.data.cells3d()`` (Z, C, Y, X) split into membrane/nuclei layers.

    Scale (Z, Y, X) = (0.29, 0.26, 0.26) µm, green/magenta, additive blending.
    """
    try:
        from skimage import data
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The cells3d sample needs scikit-image: pip install scikit-image"
        ) from err

    cells = data.cells3d()  # (60, 2, 256, 256), downloaded and cached on first use
    scale = (0.29, 0.26, 0.26)
    layers = []
    for index, (name, cmap) in enumerate((("membrane", "green"), ("nuclei", "magenta"))):
        meta = {
            "figurekit": {
                "channel_name": name,
                "channel_index": index,
                "original_layer": "cells3d",
                "source_layer": "cells3d",
                "pixel_size": [scale[1], scale[2]],
                "unit": "um",
                "z_step": scale[0],
            }
        }
        kwargs = {
            "name": name,
            "colormap": cmap,
            "blending": "additive",
            "scale": scale,
            "metadata": meta,
            "units": ("um", "um", "um"),
        }
        layers.append((cells[:, index], kwargs, "image"))
    return layers
