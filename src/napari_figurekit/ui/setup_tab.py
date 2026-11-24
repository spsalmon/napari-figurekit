"""Tab 1 — Setup: split channels, calibrate pixel size, display helpers."""

from __future__ import annotations

import copy
import inspect
import math
from typing import TYPE_CHECKING, Any

from napari.layers import Image
from napari.utils.notifications import show_info, show_warning
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import napari_compat
from ..core.calibration import (
    default_channel_axis,
    drop_axis,
    read_tiff_calibration,
    rescaled_translate,
    spatial_axes,
    updated_scale,
)
from ..core.channels import (
    COLORMAP_CHOICES,
    default_channel_names,
    default_colormaps,
    unique_names,
)
from ..core.units import UNITS, display_unit
from ._common import (
    FigureKitError,
    LayerCombo,
    base_shape,
    combo,
    combo_value,
    dspin,
    fk_meta,
    guarded,
    image_layers,
    ispin,
    set_combo_value,
    set_fk_meta,
)

if TYPE_CHECKING:
    from .._widget import FigureKitWidget

MAX_CHANNELS = 64
UNIT_ITEMS = [(display_unit(u), u) for u in UNITS]
APPLY_ITEMS = [
    ("selected layer", "selected"),
    ("all image layers", "all"),
    ("layers with the same shape", "same_shape"),
]


class SetupTab(QWidget):
    """Channel splitting, calibration and display helpers."""

    def __init__(self, kit: FigureKitWidget) -> None:
        super().__init__()
        self.kit = kit
        self.viewer = kit.viewer
        self.state = kit.state
        self._contrast_layer: Any = None

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_channels())
        layout.addWidget(self._build_calibration())
        layout.addWidget(self._build_display())
        layout.addStretch(1)

        self.source.layer_changed.connect(self._on_source_changed)
        self._on_source_changed(self.source.layer())

    # ------------------------------------------------------------------ build
    def _build_channels(self) -> QGroupBox:
        box = QGroupBox("Channels")
        form = QFormLayout(box)
        self.source = LayerCombo(self.viewer, self.kit.watcher)
        self.source.setToolTip("Image layer to split into channels and to calibrate.")
        form.addRow("Source layer", self.source)
        self.shape_label = QLabel("–")
        self.shape_label.setWordWrap(True)
        self.shape_label.setToolTip("Size of each axis of the selected layer (napari order).")
        form.addRow("Shape", self.shape_label)
        self.axis_spin = ispin(-1, -1, 0, tip="Axis holding the channels (-1 = no channel axis).")
        self.axis_spin.valueChanged.connect(self._rebuild_table)
        form.addRow("Channel axis", self.axis_spin)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Name", "Colormap"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(90)
        self.table.setToolTip("Name and colormap of each channel created by the split.")
        form.addRow(self.table)
        self.remove_original = QCheckBox("Remove original")
        self.remove_original.setChecked(True)
        self.remove_original.setToolTip(
            "Remove the multichannel layer after splitting (unchecked: keep it hidden)."
        )
        self.split_btn = QPushButton("Split channels")
        self.split_btn.setToolTip("Create one image layer per channel with additive blending.")
        self.split_btn.clicked.connect(self._on_split_clicked)
        row = QHBoxLayout()
        row.addWidget(self.remove_original)
        row.addWidget(self.split_btn)
        form.addRow(row)
        return box

    def _build_calibration(self) -> QGroupBox:
        cal = self.state.calibration
        box = QGroupBox("Calibration")
        form = QFormLayout(box)
        self.py_spin = dspin(cal.pixel_size_y, 1e-6, 1e9, 6, 0.01, "Pixel size along Y.")
        self.px_spin = dspin(cal.pixel_size_x, 1e-6, 1e9, 6, 0.01, "Pixel size along X.")
        self.square = QCheckBox("square pixels")
        self.square.setChecked(cal.square)
        self.square.setToolTip("When checked, the X pixel size mirrors Y.")
        self.py_spin.valueChanged.connect(self._on_py_changed)
        self.px_spin.valueChanged.connect(self._store)
        self.square.toggled.connect(self._on_square_toggled)
        form.addRow("Pixel size Y", self.py_spin)
        row = QHBoxLayout()
        row.addWidget(self.px_spin, 1)
        row.addWidget(self.square)
        form.addRow("Pixel size X", row)
        self.z_check = QCheckBox("Z step")
        self.z_check.setChecked(cal.use_z_step)
        self.z_check.setToolTip("Also set the step of the third-to-last axis (Z).")
        self.z_spin = dspin(cal.z_step, 1e-6, 1e9, 6, 0.1, "Distance between Z planes.")
        self.z_check.toggled.connect(self._store)
        self.z_spin.valueChanged.connect(self._store)
        self.z_row = QWidget()
        zl = QHBoxLayout(self.z_row)
        zl.setContentsMargins(0, 0, 0, 0)
        zl.addWidget(self.z_check)
        zl.addWidget(self.z_spin, 1)
        form.addRow(self.z_row)
        self.unit_combo = combo(UNIT_ITEMS, cal.unit, "Physical unit of the pixel size.")
        self.unit_combo.currentIndexChanged.connect(self._store)
        form.addRow("Unit", self.unit_combo)
        self.apply_to = combo(APPLY_ITEMS, cal.apply_to, "Which layers get the calibration.")
        self.apply_to.currentIndexChanged.connect(self._store)
        form.addRow("Apply to", self.apply_to)
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setToolTip("Set the layers' scale (and units) to this pixel size.")
        self.apply_btn.clicked.connect(self._on_apply_clicked)
        self.read_btn = QPushButton("Read from file")
        self.read_btn.setToolTip("Fill in the pixel size from ImageJ/OME TIFF metadata.")
        self.read_btn.clicked.connect(self._on_read_clicked)
        row = QHBoxLayout()
        row.addWidget(self.read_btn)
        row.addWidget(self.apply_btn)
        form.addRow(row)
        return box

    def _build_display(self) -> QGroupBox:
        box = QGroupBox("Display helpers")
        form = QFormLayout(box)
        self.additive_btn = QPushButton("Additive blending for all layers")
        self.additive_btn.setToolTip("Additive blending is how the export merges channels.")
        self.additive_btn.clicked.connect(self._on_additive_clicked)
        form.addRow(self.additive_btn)
        self.lo_spin = dspin(0.0, -1e12, 1e12, 4, 1.0, "Lower contrast limit (maps to black).")
        self.hi_spin = dspin(1.0, -1e12, 1e12, 4, 1.0, "Upper contrast limit (maps to full color).")
        self.lo_spin.editingFinished.connect(self._on_contrast_edited)
        self.hi_spin.editingFinished.connect(self._on_contrast_edited)
        row = QHBoxLayout()
        row.addWidget(self.lo_spin)
        row.addWidget(self.hi_spin)
        self.contrast_label = QLabel("Contrast")
        self.contrast_label.setToolTip("Exact contrast limits of the source layer above.")
        form.addRow(self.contrast_label, row)
        self.ref_combo = LayerCombo(self.viewer, self.kit.watcher)
        self.ref_combo.setToolTip("Layer whose contrast limits and gamma are copied.")
        form.addRow("Sync from", self.ref_combo)
        self.match_edit = QLineEdit()
        self.match_edit.setPlaceholderText("empty = same channel name")
        self.match_edit.setToolTip(
            "Copy to layers whose name contains this text; empty = same channel name."
        )
        form.addRow("Name contains", self.match_edit)
        self.sync_btn = QPushButton("Sync contrast")
        self.sync_btn.setToolTip("Give matching layers identical display settings.")
        self.sync_btn.clicked.connect(self._on_sync_clicked)
        form.addRow(self.sync_btn)
        return box

    # -------------------------------------------------------------- state I/O
    def _store(self, *args) -> None:
        cal = self.state.calibration
        cal.pixel_size_y = self.py_spin.value()
        cal.pixel_size_x = self.px_spin.value()
        cal.square = self.square.isChecked()
        cal.use_z_step = self.z_check.isChecked()
        cal.z_step = self.z_spin.value()
        cal.unit = combo_value(self.unit_combo)
        cal.apply_to = combo_value(self.apply_to)

    def load_state(self) -> None:
        cal = self.state.calibration
        for w in (self.py_spin, self.px_spin, self.z_spin, self.square, self.z_check):
            w.blockSignals(True)
        self.py_spin.setValue(cal.pixel_size_y)
        self.px_spin.setValue(cal.pixel_size_x)
        self.square.setChecked(cal.square)
        self.px_spin.setEnabled(not cal.square)
        self.z_check.setChecked(cal.use_z_step)
        self.z_spin.setValue(cal.z_step)
        for w in (self.py_spin, self.px_spin, self.z_spin, self.square, self.z_check):
            w.blockSignals(False)
        set_combo_value(self.unit_combo, cal.unit)
        set_combo_value(self.apply_to, cal.apply_to)

    # ---------------------------------------------------------------- events
    def _on_py_changed(self, value: float) -> None:
        if self.square.isChecked():
            self.px_spin.setValue(value)
        self._store()

    def _on_square_toggled(self, checked: bool) -> None:
        self.px_spin.setEnabled(not checked)
        if checked:
            self.px_spin.setValue(self.py_spin.value())
        self._store()

    def _on_source_changed(self, layer: Any) -> None:
        self._bind_contrast(layer)
        if layer is None:
            self.shape_label.setText("–")
            self.axis_spin.setRange(-1, -1)
            self.split_btn.setEnabled(False)
            self.split_btn.setToolTip("Select an image layer first.")
            self.z_row.setVisible(False)
            self._rebuild_table()
            return
        shape = base_shape(layer)
        self.shape_label.setText(" | ".join(f"axis {i}: {n}" for i, n in enumerate(shape)))
        self.axis_spin.blockSignals(True)
        self.axis_spin.setRange(-1, max(len(shape) - 3, -1))
        self.axis_spin.setValue(default_channel_axis(shape))
        self.axis_spin.blockSignals(False)
        if layer.rgb:
            self.split_btn.setEnabled(False)
            self.split_btn.setToolTip(
                "RGB layers store color, not channels, so they cannot be split here."
            )
        elif len(shape) < 3:
            self.split_btn.setEnabled(False)
            self.split_btn.setToolTip("This layer is 2D: there is no channel axis to split.")
        else:
            self.split_btn.setEnabled(True)
            self.split_btn.setToolTip("Create one image layer per channel with additive blending.")
        self._rebuild_table()
        self._prefill_calibration(layer)

    def _prefill_calibration(self, layer: Any) -> None:
        """Pre-fill the spin boxes from the layer's current scale and unit."""
        scale = [float(s) for s in layer.scale]
        self.z_row.setVisible(layer.ndim >= 3)
        for w in (self.py_spin, self.px_spin, self.z_spin, self.square):
            w.blockSignals(True)
        self.py_spin.setValue(scale[-2])
        self.px_spin.setValue(scale[-1])
        square = math.isclose(scale[-2], scale[-1], rel_tol=1e-9)
        self.square.setChecked(square)
        self.px_spin.setEnabled(not square)
        if layer.ndim >= 3:
            self.z_spin.setValue(scale[-3])
        for w in (self.py_spin, self.px_spin, self.z_spin, self.square):
            w.blockSignals(False)
        unit = napari_compat.layer_unit(layer)
        if unit:
            set_combo_value(self.unit_combo, unit)
        self._store()

    def _rebuild_table(self, *args) -> None:
        layer = self.source.layer()
        axis = self.axis_spin.value()
        n = 0
        if layer is not None and axis >= 0 and not layer.rgb:
            shape = base_shape(layer)
            if axis < len(shape) - 2:
                n = min(shape[axis], MAX_CHANNELS)
        names = default_channel_names(layer.name, n) if layer is not None else []
        cmaps = default_colormaps(n)
        self.table.setRowCount(n)
        for i in range(n):
            self.table.setItem(i, 0, QTableWidgetItem(names[i]))
            box = QComboBox()
            box.addItems(COLORMAP_CHOICES)
            box.setCurrentText(cmaps[i])
            box.setToolTip("Colormap of this channel.")
            self.table.setCellWidget(i, 1, box)

    def set_channel_rows(self, names: list[str] | None = None, cmaps: list[str] | None = None):
        """Programmatic access to the channel table (used by scripts/tests)."""
        for i in range(self.table.rowCount()):
            if names and i < len(names):
                self.table.item(i, 0).setText(names[i])
            if cmaps and i < len(cmaps):
                self.table.cellWidget(i, 1).setCurrentText(cmaps[i])

    # ------------------------------------------------------------ contrast UI
    def _bind_contrast(self, layer: Any) -> None:
        old = self._contrast_layer
        if old is not None:
            try:
                old.events.contrast_limits.disconnect(self._update_contrast_spins)
            except Exception:  # noqa: BLE001
                pass
        self._contrast_layer = layer
        if layer is not None:
            layer.events.contrast_limits.connect(self._update_contrast_spins)
        self._update_contrast_spins()

    def _update_contrast_spins(self, *args) -> None:
        layer = self._contrast_layer
        try:
            enabled = layer is not None and not layer.rgb
            self.lo_spin.setEnabled(enabled)
            self.hi_spin.setEnabled(enabled)
            if not enabled:
                return
            lo, hi = layer.contrast_limits
            for w, v in ((self.lo_spin, lo), (self.hi_spin, hi)):
                w.blockSignals(True)
                w.setValue(float(v))
                w.blockSignals(False)
            self.contrast_label.setText("Contrast")
        except RuntimeError:  # widget deleted
            pass

    def disconnect_all(self) -> None:
        self._bind_contrast(None)

    @guarded("Set contrast")
    def _on_contrast_edited(self, *args) -> None:
        layer = self._contrast_layer
        if layer is None:
            return
        lo, hi = self.lo_spin.value(), self.hi_spin.value()
        if lo >= hi:
            raise FigureKitError("the lower limit must be smaller than the upper limit")
        if (lo, hi) != tuple(float(v) for v in layer.contrast_limits):
            layer.contrast_limits = (lo, hi)

    # ---------------------------------------------------------------- actions
    @guarded("Split channels")
    def _on_split_clicked(self, *args) -> None:
        self.split_channels()

    def split_channels(self) -> list:
        """Split the source layer along the channel axis. Returns the new layers."""
        layer = self.source.layer()
        if layer is None:
            raise FigureKitError("select an image layer first")
        if layer.rgb:
            raise FigureKitError("RGB layers cannot be split")
        axis = self.axis_spin.value()
        shape = base_shape(layer)
        if axis < 0 or axis >= len(shape) - 2:
            raise FigureKitError("choose a channel axis (not one of the last two axes)")
        n = shape[axis]
        if n > MAX_CHANNELS:
            raise FigureKitError(f"axis {axis} has {n} entries; at most {MAX_CHANNELS} channels")
        names = unique_names(
            [
                (self.table.item(i, 0).text() if self.table.item(i, 0) else "")
                for i in range(self.table.rowCount())
            ]
        )
        cmaps = [self.table.cellWidget(i, 1).currentText() for i in range(self.table.rowCount())]
        if len(names) != n:  # table out of date
            names = default_channel_names(layer.name, n)
            cmaps = default_colormaps(n)

        scale = drop_axis(layer.scale, axis)
        translate = drop_axis(layer.translate, axis)
        units = napari_compat.get_layer_units(layer)
        units = list(drop_axis(units, axis)) if units else None
        src_fk = fk_meta(layer)
        unit = napari_compat.layer_unit(layer)
        source_path = getattr(layer.source, "path", None) or src_fk.get("source_path")
        try:
            base_meta = copy.deepcopy(dict(layer.metadata))
        except Exception:  # noqa: BLE001 - un-copyable metadata: shallow copy
            base_meta = dict(layer.metadata)
        metas = []
        for i in range(n):
            meta = dict(base_meta)
            fk = dict(src_fk)
            fk.update(
                channel_name=names[i],
                channel_index=i,
                source_layer=layer.name,
                original_layer=src_fk.get("original_layer", layer.name),
                source_path=str(source_path) if source_path else None,
                pixel_size=[float(scale[-2]), float(scale[-1])],
                unit=unit,
            )
            meta["figurekit"] = fk
            metas.append(meta)
        data = list(layer.data) if layer.multiscale else layer.data
        has_units_kw = "units" in inspect.signature(self.viewer.add_image).parameters
        new_layers = self.viewer.add_image(
            data,
            channel_axis=axis,
            name=names,
            colormap=cmaps,
            blending="additive",
            scale=[scale] * n,
            translate=[translate] * n,
            metadata=metas,
            multiscale=layer.multiscale,
            **({"units": [tuple(units)] * n} if units and has_units_kw else {}),
        )
        if units:
            for new in new_layers:
                napari_compat.copy_layer_units(units, new)
        if self.remove_original.isChecked():
            self.viewer.layers.remove(layer)
        else:
            layer.visible = False
        show_info(f"Split '{layer.name}' into {n} channels: {', '.join(names)}")
        return list(new_layers)

    def _targets(self) -> list:
        mode = combo_value(self.apply_to)
        layer = self.source.layer()
        layers = image_layers(self.viewer)
        if mode == "selected":
            return [layer] if layer is not None else []
        if mode == "same_shape":
            if layer is None:
                return []
            shape = base_shape(layer)
            return [lay for lay in layers if base_shape(lay) == shape]
        return layers

    @guarded("Apply calibration")
    def _on_apply_clicked(self, *args) -> None:
        self.apply_calibration()

    def apply_calibration(self) -> list:
        """Apply the pixel size to the target layers. Returns the calibrated layers."""
        self._store()
        cal = self.state.calibration
        targets = self._targets()
        if not targets:
            raise FigureKitError("no image layer to calibrate")
        py, px = cal.pixel_size_y, (cal.pixel_size_y if cal.square else cal.pixel_size_x)
        use_z = cal.use_z_step  # applied only to layers with >= 3 dims
        for layer in targets:
            with_z = use_z and layer.ndim >= 3
            old = tuple(float(s) for s in layer.scale)
            new = updated_scale(old, py, px, cal.z_step if with_z else None)
            new_t = rescaled_translate(layer.translate, old, new)
            layer.scale = new
            layer.translate = new_t
            values: dict[str, Any] = {"pixel_size": [py, px], "unit": cal.unit}
            if with_z:
                values["z_step"] = cal.z_step
            set_fk_meta(layer, **values)
            napari_compat.set_layer_units(layer, cal.unit, spatial_axes(layer.ndim, with_z))
        # Helper layers (points, shapes, vectors) live in the same world: same units.
        for layer in self.viewer.layers:
            if not isinstance(layer, Image) and layer.ndim >= 2:
                napari_compat.set_layer_units(layer, cal.unit, spatial_axes(layer.ndim, False))
        self.viewer.reset_view()
        self.kit.on_calibration_changed()
        show_info(
            f"Calibrated {len(targets)} layer(s): {py:g} × {px:g} {display_unit(cal.unit)}/px"
        )
        return targets

    @guarded("Read calibration")
    def _on_read_clicked(self, *args) -> None:
        self.read_from_file()

    def read_from_file(self) -> bool:
        layer = self.source.layer()
        if layer is None:
            raise FigureKitError("select an image layer first")
        path = getattr(layer.source, "path", None) or fk_meta(layer).get("source_path")
        if not path:
            show_warning("No file is known for this layer; values left unchanged.")
            return False
        cal = read_tiff_calibration(path)
        if cal is None:
            show_warning(f"No pixel size found in {path}; values left unchanged.")
            return False
        self.square.setChecked(math.isclose(cal.pixel_size_y, cal.pixel_size_x, rel_tol=1e-6))
        self.py_spin.setValue(cal.pixel_size_y)
        self.px_spin.setValue(cal.pixel_size_x)
        set_combo_value(self.unit_combo, cal.unit)
        if cal.z_step is not None and layer.ndim >= 3:
            self.z_spin.setValue(cal.z_step)
            self.z_check.setChecked(True)
        self._store()
        z = f", Z {cal.z_step:g}" if cal.z_step else ""
        show_info(
            f"Read {cal.source} calibration: {cal.pixel_size_y:g} × {cal.pixel_size_x:g}"
            f"{z} {display_unit(cal.unit)}. Press Apply to use it."
        )
        return True

    @guarded("Additive blending")
    def _on_additive_clicked(self, *args) -> None:
        layers = image_layers(self.viewer)
        for layer in layers:
            layer.blending = "additive"
        show_info(f"{len(layers)} image layer(s) set to additive blending")

    @guarded("Sync contrast")
    def _on_sync_clicked(self, *args) -> None:
        self.sync_contrast()

    def sync_contrast(self) -> list:
        ref = self.ref_combo.layer()
        if ref is None:
            raise FigureKitError("choose a reference layer")
        text = self.match_edit.text().strip()
        layers = [lay for lay in image_layers(self.viewer) if lay is not ref and not lay.rgb]
        if text:
            targets = [lay for lay in layers if text in lay.name]
        else:
            channel = fk_meta(ref).get("channel_name")
            if not channel:
                raise FigureKitError(
                    f"'{ref.name}' has no channel name; type a name substring to match instead"
                )
            targets = [lay for lay in layers if fk_meta(lay).get("channel_name") == channel]
        for lay in targets:
            lay.contrast_limits = tuple(float(v) for v in ref.contrast_limits)
            lay.gamma = float(ref.gamma)
        if targets:
            show_info(f"Copied contrast of '{ref.name}' to {len(targets)} layer(s)")
        else:
            show_warning("No matching layers found")
        return targets
