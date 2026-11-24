"""Version adapter for napari APIs that changed between 0.5 and 0.7.

Everything version-sensitive (scale-bar overlay fields, ``layer.units``) goes through
here. Every attribute is set only if it exists, inside try/except, so a newer or older
napari degrades gracefully. The layer ``scale`` is always the source of truth.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from .scalebar import parse_color
from .units import convert, pint_unit, try_parse_unit

logger = logging.getLogger(__name__)


def napari_version() -> str:
    """Installed napari version string (``"unknown"`` if napari is not importable)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("napari")
    except PackageNotFoundError:  # pragma: no cover
        return "unknown"


def _field_names(obj: Any) -> set[str] | None:
    """Declared model fields (pydantic v2/v1), or None for non-model objects."""
    cls = type(obj)
    names = getattr(cls, "model_fields", None) or getattr(cls, "__fields__", None)
    return set(names) if names else None


def _set(obj: Any, name: str, value: Any) -> bool:
    """Set ``obj.name`` only if it is a real field of the installed version.

    Deprecated properties (e.g. ``ScaleBar.unit`` in napari 0.7, where units come from
    ``layer.units``) are skipped so no deprecation warnings are triggered.
    """
    fields = _field_names(obj)
    if (fields is not None and name not in fields) or not hasattr(obj, name):
        return False
    try:
        setattr(obj, name, value)
        return True
    except Exception as err:  # noqa: BLE001 - version differences, log and continue
        logger.debug("Could not set %s.%s=%r: %s", type(obj).__name__, name, value, err)
        return False


def _position_value(overlay: Any, name: str) -> Any:
    """Convert e.g. ``"bottom_right"`` to the installed enum member, if there is one."""
    current = getattr(overlay, "position", None)
    enum_cls = type(current)
    members = getattr(enum_cls, "__members__", None)
    if members:
        for member in members.values():
            if str(getattr(member, "value", "")).lower() == name.lower():
                return member
    return name


def apply_scale_bar(viewer: Any, settings: Any, world_unit: str | None) -> list[str]:
    """Mirror :class:`~napari_figurekit.state.ScaleBarSettings` on ``viewer.scale_bar``.

    The canvas overlay is only a preview; the exported bar is drawn by
    :mod:`napari_figurekit.core.scalebar`.

    Parameters
    ----------
    viewer : napari.Viewer or ViewerModel
    settings : ScaleBarSettings
    world_unit : str, optional
        Unit of the layers' world coordinates; the fixed length is converted to it.

    Returns
    -------
    list of str
        Names of the fields that were set.
    """
    sb = getattr(viewer, "scale_bar", None)
    if sb is None:
        return []
    done = []
    length = float(settings.length)
    if world_unit and try_parse_unit(world_unit) and try_parse_unit(settings.unit):
        length = convert(length, settings.unit, world_unit)
    rgb = tuple(c / 255.0 for c in parse_color(settings.color)) + (1.0,)
    box_rgb = tuple(c / 255.0 for c in parse_color(settings.box_color)) + (1.0,)
    fields: list[tuple[str, Any]] = [
        ("visible", bool(settings.visible)),
        ("length", length),
        ("font_size", float(settings.font_size)),
        ("colored", True),
        ("color", rgb),
        ("box", bool(settings.box)),
        ("box_color", box_rgb),
        ("position", _position_value(sb, settings.position)),
        ("ticks", False),
    ]
    if world_unit and try_parse_unit(world_unit):
        fields.insert(1, ("unit", pint_unit(world_unit)))
    for name, value in fields:
        if _set(sb, name, value):
            done.append(name)
    return done


def get_layer_units(layer: Any) -> tuple[str, ...] | None:
    """Return ``layer.units`` as strings, or ``None`` if unsupported."""
    units = getattr(layer, "units", None)
    if units is None:
        return None
    try:
        return tuple(str(u) for u in units)
    except TypeError:
        return None


def set_layer_units(layer: Any, unit: str, spatial_axes: Sequence[int]) -> bool:
    """Set ``layer.units`` on the spatial axes, keeping the other axes' units.

    Only if the installed napari has ``layer.units``. Failures are logged, not raised.
    """
    current = get_layer_units(layer)
    if current is None:
        return False
    try:
        new = list(current)
        for ax in spatial_axes:
            new[ax] = pint_unit(unit)
        layer.units = tuple(new)
        return True
    except Exception as err:  # noqa: BLE001
        logger.info("Could not set units on layer %r: %s", getattr(layer, "name", "?"), err)
        return False


def copy_layer_units(src_units: Sequence[str] | None, layer: Any) -> bool:
    """Copy a full units tuple onto ``layer`` if supported and of matching length."""
    if src_units is None or get_layer_units(layer) is None:
        return False
    try:
        if len(src_units) != layer.ndim:
            return False
        layer.units = tuple(src_units)
        return True
    except Exception as err:  # noqa: BLE001
        logger.info("Could not copy units: %s", err)
        return False


def layer_unit(layer: Any) -> str | None:
    """Physical unit of a layer: ``metadata["figurekit"]["unit"]``, else ``layer.units``."""
    meta = getattr(layer, "metadata", {}) or {}
    fk = meta.get("figurekit", {}) if isinstance(meta, dict) else {}
    unit = try_parse_unit(fk.get("unit")) if isinstance(fk, dict) else None
    if unit:
        return unit
    units = get_layer_units(layer)
    if units:
        return try_parse_unit(units[-1])
    return None
