"""Writing panels to disk: file names, PNG/TIFF with DPI, JSON sidecars.

Pure Python + NumPy/Pillow/tifffile; no Qt and no viewer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

FORMATS = {"png": ".png", "tiff": ".tif"}

_UNSAFE = re.compile(r"[^A-Za-z0-9._\-]+")


def sanitize_filename(name: str) -> str:
    """Make a safe file-name stem: unsafe characters → ``_``, ``µ`` → ``u``."""
    name = str(name).replace("µ", "u").replace("μ", "u")
    name = _UNSAFE.sub("_", name).strip("._")
    name = re.sub(r"_+", "_", name)
    return name or "panel"


def format_filename(pattern: str, **fields: str) -> str:
    """Fill a pattern such as ``"{source}_{crop}_{panel}"`` and sanitise the result.

    Unknown placeholders raise ``ValueError`` with a clear message.
    """
    try:
        raw = pattern.format(**fields)
    except (KeyError, IndexError, ValueError) as err:
        allowed = ", ".join("{" + k + "}" for k in fields)
        raise ValueError(f"Bad filename pattern {pattern!r}: {err}. Use {allowed}.") from err
    return sanitize_filename(raw)


def sidecar_path(image_path: str | Path) -> Path:
    """JSON sidecar next to an image: ``panel.png`` → ``panel.png.json``."""
    p = Path(image_path)
    return p.with_name(p.name + ".json")


def unique_path(folder: str | Path, stem: str, suffix: str, reserved: set[Path] | None = None):
    """``folder/stem.suffix``, or ``stem_1``, ``stem_2``, … if it (or its sidecar) exists.

    ``reserved`` holds paths already claimed in the same export run.
    """
    folder = Path(folder)
    reserved = reserved or set()
    i = 0
    while True:
        name = stem if i == 0 else f"{stem}_{i}"
        path = folder / f"{name}{suffix}"
        if not path.exists() and not sidecar_path(path).exists() and path not in reserved:
            return path
        i += 1


def write_image(path: str | Path, rgb: np.ndarray, fmt: str = "png", dpi: float = 300) -> Path:
    """Write an 8-bit RGB image with its DPI in the file metadata.

    PNG via Pillow (``pHYs`` chunk); TIFF via tifffile (``XResolution``/``YResolution``
    tags, inch unit).
    """
    path = Path(path)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got {rgb.shape}")
    if fmt == "png":
        from PIL import Image

        Image.fromarray(rgb).save(path, format="PNG", dpi=(float(dpi), float(dpi)))
    elif fmt == "tiff":
        import tifffile

        tifffile.imwrite(
            path,
            rgb,
            photometric="rgb",
            resolution=(float(dpi), float(dpi)),
            resolutionunit="INCH",
            metadata=None,
        )
    else:
        raise ValueError(f"unknown format {fmt!r}")
    return path


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def write_sidecar(image_path: str | Path, info: Mapping[str, Any]) -> Path:
    """Write the JSON sidecar of an exported image and return its path."""
    path = sidecar_path(image_path)
    data = dict(info)
    data.setdefault("timestamp", datetime.now().astimezone().isoformat(timespec="seconds"))
    data["file"] = Path(image_path).name
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(data), fh, indent=2, ensure_ascii=False)
    return path


def printed_size_mm(width_px: int, height_px: int, dpi: float) -> tuple[float, float]:
    """Printed size in millimetres at ``dpi``."""
    return (width_px / dpi * 25.4, height_px / dpi * 25.4)
