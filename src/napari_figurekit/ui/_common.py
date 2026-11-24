"""Shared Qt helpers: layer watching, layer combo/checklist, error-safe slots."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
from napari.layers import Image
from napari.utils.notifications import show_error
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QListWidget,
    QListWidgetItem,
    QSpinBox,
    QWidget,
)

if TYPE_CHECKING:
    import napari

logger = logging.getLogger(__name__)

PREVIEW_LAYER = "figurekit preview"

#: Minimum width of spin boxes/combos, so the dock fits in ~340 px.
MIN_FIELD_WIDTH = 64


class FigureKitError(Exception):
    """An expected, user-facing error (shown as a notification, no traceback)."""


def fk_meta(layer: Any) -> dict:
    """``layer.metadata["figurekit"]`` (read-only view; empty dict if missing)."""
    meta = getattr(layer, "metadata", None)
    if not isinstance(meta, dict):
        return {}
    fk = meta.get("figurekit")
    return fk if isinstance(fk, dict) else {}


def set_fk_meta(layer: Any, **values: Any) -> None:
    """Update ``layer.metadata["figurekit"]`` in place with ``values``."""
    meta = dict(layer.metadata) if isinstance(layer.metadata, dict) else {}
    fk = dict(meta.get("figurekit") or {})
    fk.update(values)
    meta["figurekit"] = fk
    layer.metadata = meta


def is_image(layer: Any) -> bool:
    """True for image layers that the user works with (not the export preview)."""
    return isinstance(layer, Image) and fk_meta(layer).get("role") != "preview"


def image_layers(viewer: napari.Viewer) -> list[Image]:
    return [layer for layer in viewer.layers if is_image(layer)]


def get_layer(viewer: napari.Viewer, name: str | None):
    if not name:
        return None
    try:
        return viewer.layers[name]
    except (KeyError, ValueError):
        return None


def guarded(action: str) -> Callable:
    """Decorate a Qt slot: errors become notifications instead of crashing the GUI."""

    def deco(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except FigureKitError as err:
                show_error(f"{action}: {err}")
            except Exception as err:  # noqa: BLE001 - never crash the GUI
                logger.exception("%s failed", action)
                show_error(f"{action} failed: {type(err).__name__}: {err}")
            return None

        return wrapper

    return deco


class LayerWatcher(QObject):
    """Emits :attr:`changed` when layers are inserted, removed, moved or renamed."""

    changed = Signal()

    def __init__(self, viewer: napari.Viewer, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._named: dict[int, Any] = {}
        ev = viewer.layers.events
        ev.inserted.connect(self._on_inserted)
        ev.removed.connect(self._on_removed)
        ev.moved.connect(self._emit)
        for layer in viewer.layers:
            self._watch(layer)

    def _watch(self, layer: Any) -> None:
        if id(layer) not in self._named:
            layer.events.name.connect(self._emit)
            self._named[id(layer)] = layer

    def _on_inserted(self, event=None) -> None:
        if event is not None and getattr(event, "value", None) is not None:
            self._watch(event.value)
        self._emit()

    def _on_removed(self, event=None) -> None:
        layer = getattr(event, "value", None)
        if layer is not None and id(layer) in self._named:
            try:
                layer.events.name.disconnect(self._emit)
            except Exception:  # noqa: BLE001
                pass
            self._named.pop(id(layer), None)
        self._emit()

    def _emit(self, *args) -> None:
        try:
            self.changed.emit()
        except RuntimeError:  # Qt object already deleted during teardown
            pass

    def disconnect_all(self) -> None:
        ev = self.viewer.layers.events
        for emitter, cb in (
            (ev.inserted, self._on_inserted),
            (ev.removed, self._on_removed),
            (ev.moved, self._emit),
        ):
            try:
                emitter.disconnect(cb)
            except Exception:  # noqa: BLE001
                pass
        for layer in list(self._named.values()):
            try:
                layer.events.name.disconnect(self._emit)
            except Exception:  # noqa: BLE001
                pass
        self._named.clear()


class LayerCombo(QComboBox):
    """Combo box of layer names kept in sync with the viewer.

    Emits :attr:`layer_changed` (with the layer or ``None``) when the selection changes.
    """

    layer_changed = Signal(object)

    def __init__(
        self,
        viewer: napari.Viewer,
        watcher: LayerWatcher,
        predicate: Callable[[Any], bool] = is_image,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumWidth(MIN_FIELD_WIDTH)
        self.viewer = viewer
        self.predicate = predicate
        self._last: str | None = None
        self.currentIndexChanged.connect(self._on_index)
        watcher.changed.connect(self.refresh)
        self.refresh()

    def layers(self) -> list:
        return [layer for layer in self.viewer.layers if self.predicate(layer)]

    def refresh(self) -> None:
        names = [layer.name for layer in self.layers()]
        current = self.currentText() or self._last
        self.blockSignals(True)
        self.clear()
        self.addItems(names)
        if current in names:
            self.setCurrentIndex(names.index(current))
        elif names:
            # Prefer the active layer if it qualifies.
            active = self.viewer.layers.selection.active
            if active is not None and active.name in names:
                self.setCurrentIndex(names.index(active.name))
            else:
                self.setCurrentIndex(0)
        self.blockSignals(False)
        if self.currentText() != self._last:
            self._on_index()

    def _on_index(self, *args) -> None:
        self._last = self.currentText() or None
        self.layer_changed.emit(self.layer())

    def layer(self):
        return get_layer(self.viewer, self.currentText())

    def set_layer(self, name: str) -> None:
        idx = self.findText(name)
        if idx >= 0:
            self.setCurrentIndex(idx)


class LayerChecklist(QListWidget):
    """Checkable list of layers. Items carry the layer name; disabled items get a tooltip."""

    toggled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.itemChanged.connect(lambda *_: self.toggled.emit())

    def set_items(self, entries: Iterable[tuple[str, bool, bool, str]]) -> None:
        """Replace the items with ``(name, checked, enabled, tooltip)`` entries."""
        self.blockSignals(True)
        self.clear()
        for name, checked, enabled, tip in entries:
            item = QListWidgetItem(name)
            flags = Qt.ItemFlag.ItemIsUserCheckable
            if enabled:
                flags |= Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            item.setFlags(flags)
            item.setCheckState(
                Qt.CheckState.Checked if checked and enabled else Qt.CheckState.Unchecked
            )
            if tip:
                item.setToolTip(tip)
            self.addItem(item)
        self.blockSignals(False)

    def entries(self) -> list[tuple[str, bool]]:
        """``(name, checked)`` for every item, in list order."""
        out = []
        for i in range(self.count()):
            item = self.item(i)
            out.append((item.text(), item.checkState() == Qt.CheckState.Checked))
        return out

    def checked_names(self) -> list[str]:
        return [name for name, checked in self.entries() if checked]

    def set_checked(self, names: Iterable[str]) -> None:
        wanted = set(names)
        self.blockSignals(True)
        for i in range(self.count()):
            item = self.item(i)
            if item.flags() & Qt.ItemFlag.ItemIsEnabled:
                item.setCheckState(
                    Qt.CheckState.Checked if item.text() in wanted else Qt.CheckState.Unchecked
                )
        self.blockSignals(False)
        self.toggled.emit()


def dspin(
    value: float,
    lo: float,
    hi: float,
    decimals: int = 3,
    step: float = 1.0,
    tip: str = "",
    suffix: str = "",
) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(lo, hi)
    box.setSingleStep(step)
    box.setValue(value)
    box.setKeyboardTracking(False)
    box.setMinimumWidth(MIN_FIELD_WIDTH)  # override the range-based size hint
    if suffix:
        box.setSuffix(suffix)
    if tip:
        box.setToolTip(tip)
    return box


def ispin(value: int, lo: int, hi: int, tip: str = "", suffix: str = "") -> QSpinBox:
    box = QSpinBox()
    box.setRange(lo, hi)
    box.setValue(value)
    box.setKeyboardTracking(False)
    box.setMinimumWidth(MIN_FIELD_WIDTH)  # override the range-based size hint
    if suffix:
        box.setSuffix(suffix)
    if tip:
        box.setToolTip(tip)
    return box


def combo(items: Iterable[tuple[str, str]], current: str | None = None, tip: str = "") -> QComboBox:
    """Combo with ``(label, data)`` items; :func:`combo_value` reads the data back."""
    box = QComboBox()
    box.setMinimumWidth(MIN_FIELD_WIDTH)
    for label, data in items:
        box.addItem(label, data)
    if current is not None:
        set_combo_value(box, current)
    if tip:
        box.setToolTip(tip)
    return box


def combo_value(box: QComboBox) -> str:
    data = box.currentData()
    return data if data is not None else box.currentText()


def set_combo_value(box: QComboBox, value: str) -> None:
    idx = box.findData(value)
    if idx < 0:
        idx = box.findText(value)
    if idx >= 0:
        box.setCurrentIndex(idx)


def base_shape(layer: Any) -> tuple[int, ...]:
    """Shape of the full-resolution data (level 0 for multiscale layers)."""
    data = layer.data[0] if getattr(layer, "multiscale", False) else layer.data
    return tuple(int(n) for n in data.shape)


def base_data(layer: Any):
    """Full-resolution data array (level 0 for multiscale layers)."""
    return layer.data[0] if getattr(layer, "multiscale", False) else layer.data


def world_point_for(viewer: Any, world_yx) -> np.ndarray:
    """Full world coordinate: the viewer's current point with the last axes replaced.

    Used to convert 2D annotations (points, lines) into nD layers' data coordinates.
    """
    full = np.asarray(viewer.dims.point, dtype=float).copy()
    w = np.asarray(world_yx, dtype=float)
    n = min(len(w), len(full))
    full[len(full) - n :] = w[len(w) - n :]
    return full


def units_kwargs(add_method: Callable, ref: Any, ndim: int) -> dict:
    """``{"units": ...}`` for a new helper layer, matching ``ref`` (if supported).

    Passing units at creation avoids napari's "inconsistent units" warning that would
    fire if they were only set after insertion.
    """
    import inspect

    from ..core import napari_compat

    if ref is None:
        return {}
    units = napari_compat.get_layer_units(ref)
    if units is None or len(units) < ndim:
        return {}
    try:
        if "units" not in inspect.signature(add_method).parameters:
            return {}
    except (TypeError, ValueError):
        return {}
    return {"units": tuple(units[len(units) - ndim :])}


def match_units(layer: Any, ref: Any) -> None:
    """Give a helper layer (points, shapes, vectors, preview) the units of ``ref``.

    napari ignores units for the scale bar when layers disagree, so helper layers must
    carry the same physical units as the images.
    """
    from ..core import napari_compat

    if ref is None:
        return
    units = napari_compat.get_layer_units(ref)
    if units is None or len(units) < layer.ndim:
        return
    napari_compat.copy_layer_units(tuple(units[len(units) - layer.ndim :]), layer)


def layer_pixel_size(layer: Any) -> tuple[float, float]:
    """(pixel_size_y, pixel_size_x) of a layer = its scale on the last two axes."""
    return float(layer.scale[-2]), float(layer.scale[-1])


def is_axis_aligned(layer: Any) -> bool:
    """True if the layer has no rotation/shear/extra affine (required for exact crops)."""
    try:
        return bool(
            np.allclose(layer.rotate, np.eye(layer.ndim))
            and np.allclose(layer.shear, 0)
            and np.allclose(layer.affine.affine_matrix, np.eye(layer.ndim + 1))
        )
    except Exception:  # noqa: BLE001
        return True
