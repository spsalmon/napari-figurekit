"""Tab 2 — Crop around points: live preview of exact windows, pure-slice crops."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
from napari.layers import Points, Shapes
from napari.utils.notifications import show_info, show_warning
from qtpy.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core import napari_compat
from ..core.crop import (
    CropWindow,
    crop_array,
    crop_translate,
    crop_window,
    window_rectangle_world,
    window_size_px,
)
from ..core.units import convert, display_unit
from ._common import (
    FigureKitError,
    LayerChecklist,
    base_data,
    base_shape,
    combo,
    combo_value,
    dspin,
    fk_meta,
    get_layer,
    guarded,
    image_layers,
    is_axis_aligned,
    match_units,
    set_combo_value,
    units_kwargs,
    world_point_for,
)

if TYPE_CHECKING:
    from .._widget import FigureKitWidget

POINTS_LAYER = "crop centers"
PREVIEW_LAYER = "crop preview"
OUTLINES_LAYER = "crop outlines"

EDGE_ITEMS = [
    ("shift inside image", "shift"),
    ("clip (smaller crop)", "clip"),
    ("pad with zeros", "pad"),
]


def _compatible(layer: Any, ref: Any) -> bool:
    return base_shape(layer)[-2:] == base_shape(ref)[-2:] and np.allclose(
        np.asarray(layer.scale[-2:], float), np.asarray(ref.scale[-2:], float)
    )


class CropTab(QWidget):
    """Crop squares around the points of the ``crop centers`` layer."""

    def __init__(self, kit: FigureKitWidget) -> None:
        super().__init__()
        self.kit = kit
        self.viewer = kit.viewer
        self.state = kit.state
        self._points: Any = None
        self._updating = False
        cs = self.state.crop

        layout = QVBoxLayout(self)
        box = QGroupBox("Crop window")
        form = QFormLayout(box)
        self.size_spin = dspin(cs.size, 0.001, 1e9, 3, 1.0, "Side length of the square crop.")
        self.mode_combo = combo(
            [("physical units", "physical"), ("pixels", "pixels")],
            cs.size_mode,
            "Crop size in physical units (calibration unit) or in pixels.",
        )
        row = QHBoxLayout()
        row.addWidget(self.size_spin, 1)
        row.addWidget(self.mode_combo)
        form.addRow("Crop size", row)
        self.edge_combo = combo(EDGE_ITEMS, cs.edge_mode, "What to do when a crop hits the edge.")
        form.addRow("At image edge", self.edge_combo)
        self.window_label = QLabel("–")
        self.window_label.setWordWrap(True)
        self.window_label.setToolTip("Exact crop window in pixels of the reference layer.")
        form.addRow("Window", self.window_label)
        layout.addWidget(box)

        self.points_btn = QPushButton("Create / select center points layer")
        self.points_btn.setToolTip("Click in the image to add one point per crop.")
        self.points_btn.clicked.connect(self._on_points_clicked)
        layout.addWidget(self.points_btn)

        lbox = QGroupBox("Layers to crop")
        lv = QVBoxLayout(lbox)
        self.layer_list = LayerChecklist()
        self.layer_list.setToolTip("Image layers to crop; all must share Y/X shape and pixel size.")
        self.layer_list.setMinimumHeight(90)
        lv.addWidget(self.layer_list)
        layout.addWidget(lbox)

        obox = QGroupBox("Options")
        ov = QVBoxLayout(obox)
        self.hide_check = QCheckBox("Hide source layers after crop")
        self.hide_check.setChecked(cs.hide_sources)
        self.hide_check.setToolTip("Hide the full images so the crops are visible.")
        self.outline_check = QCheckBox("Add outline of crop to source")
        self.outline_check.setChecked(cs.add_outlines)
        self.outline_check.setToolTip(
            "Keep the crop rectangles in a 'crop outlines' layer for inset boxes."
        )
        self.zoom_check = QCheckBox("Zoom to first crop")
        self.zoom_check.setChecked(cs.zoom_to_first)
        self.zoom_check.setToolTip("Center and zoom the view on the first crop.")
        for w in (self.hide_check, self.outline_check, self.zoom_check):
            ov.addWidget(w)
            w.toggled.connect(self._store)
        layout.addWidget(obox)

        self.crop_btn = QPushButton("Crop")
        self.crop_btn.setToolTip("Create one cropped layer per point and per checked layer.")
        self.crop_btn.clicked.connect(self._on_crop_clicked)
        layout.addWidget(self.crop_btn)
        layout.addStretch(1)

        self.size_spin.valueChanged.connect(self._on_settings_changed)
        self.mode_combo.currentIndexChanged.connect(self._on_settings_changed)
        self.edge_combo.currentIndexChanged.connect(self._on_settings_changed)
        self.layer_list.toggled.connect(self._on_list_toggled)
        kit.watcher.changed.connect(self._on_layers_changed)
        self._update_unit_label()
        self._on_layers_changed()

    # -------------------------------------------------------------- state I/O
    def _store(self, *args) -> None:
        cs = self.state.crop
        cs.size = self.size_spin.value()
        cs.size_mode = combo_value(self.mode_combo)
        cs.edge_mode = combo_value(self.edge_combo)
        cs.hide_sources = self.hide_check.isChecked()
        cs.add_outlines = self.outline_check.isChecked()
        cs.zoom_to_first = self.zoom_check.isChecked()

    def load_state(self) -> None:
        cs = self.state.crop
        self.size_spin.setValue(cs.size)
        set_combo_value(self.mode_combo, cs.size_mode)
        set_combo_value(self.edge_combo, cs.edge_mode)
        self.hide_check.setChecked(cs.hide_sources)
        self.outline_check.setChecked(cs.add_outlines)
        self.zoom_check.setChecked(cs.zoom_to_first)
        self._update_unit_label()

    def _update_unit_label(self) -> None:
        unit = display_unit(self.state.calibration.unit)
        self.mode_combo.setItemText(0, f"physical ({unit})")

    def on_calibration_changed(self) -> None:
        self._update_unit_label()
        self._refresh_list()
        self.update_preview()

    def _on_settings_changed(self, *args) -> None:
        self._store()
        self.update_preview()

    # ---------------------------------------------------------- layer list
    def _on_layers_changed(self) -> None:
        self._bind_points()
        self._refresh_list()
        self.update_preview()

    def _on_list_toggled(self) -> None:
        if self._updating:
            return
        self._refresh_list()
        self.update_preview()

    def _refresh_list(self) -> None:
        """Rebuild the checklist, keeping checks; incompatible layers are disabled."""
        self._updating = True
        try:
            prev = dict(self.layer_list.entries())
            layers = image_layers(self.viewer)
            croppable = {lay.name: (not lay.rgb and is_axis_aligned(lay)) for lay in layers}
            ref = None
            for lay in layers:
                if prev.get(lay.name) and croppable[lay.name]:
                    ref = lay
                    break
            if ref is None:
                for lay in layers:
                    if lay.name not in prev and lay.visible and croppable[lay.name]:
                        ref = lay
                        break
            entries = []
            for lay in layers:
                if lay.rgb:
                    entries.append((lay.name, False, False, "RGB layers cannot be cropped here."))
                    continue
                if not croppable[lay.name]:
                    entries.append(
                        (lay.name, False, False, "Rotated/sheared layers cannot be cropped.")
                    )
                    continue
                ok = ref is None or _compatible(lay, ref)
                tip = (
                    ""
                    if ok
                    else (f"Y/X shape or pixel size differs from '{ref.name}'; uncheck it first.")
                )
                checked = prev[lay.name] if lay.name in prev else (lay.visible and ok)
                entries.append((lay.name, checked and ok, ok, tip))
            self.layer_list.set_items(entries)
        finally:
            self._updating = False

    def checked_layers(self) -> list:
        out = []
        for name in self.layer_list.checked_names():
            layer = get_layer(self.viewer, name)
            if layer is not None:
                out.append(layer)
        return out

    def set_checked(self, names: list[str]) -> None:
        self._updating = True
        try:
            self.layer_list.set_checked(names)
        finally:
            self._updating = False
        self._refresh_list()
        self.update_preview()

    # ---------------------------------------------------------------- points
    def points_layer(self):
        layer = get_layer(self.viewer, POINTS_LAYER)
        return layer if isinstance(layer, Points) else None

    def _bind_points(self) -> None:
        layer = self.points_layer()
        if layer is self._points:
            return
        if self._points is not None:
            try:
                self._points.events.data.disconnect(self._on_points_data)
            except Exception:  # noqa: BLE001
                pass
        self._points = layer
        if layer is not None:
            layer.events.data.connect(self._on_points_data)

    def _on_points_data(self, *args) -> None:
        try:
            self.update_preview()
        except RuntimeError:  # widget deleted
            pass

    def disconnect_all(self) -> None:
        if self._points is not None:
            try:
                self._points.events.data.disconnect(self._on_points_data)
            except Exception:  # noqa: BLE001
                pass
        self._points = None

    @guarded("Center points")
    def _on_points_clicked(self, *args) -> None:
        self.ensure_points_layer()

    def ensure_points_layer(self):
        """Create (or reuse) the ``crop centers`` points layer and make it active."""
        layer = self.points_layer()
        if layer is None:
            self._ensure_shapes(PREVIEW_LAYER, "yellow")
            ref = self.reference_layer()
            size = 10.0
            if ref is not None:
                shape = base_shape(ref)
                size = max(shape[-1] * float(ref.scale[-1]) * 0.02, 1e-9)
            uref = ref or self.kit.reference_layer()
            layer = self.viewer.add_points(
                ndim=2,
                name=POINTS_LAYER,
                size=size,
                face_color="red",
                border_color="white",
                **units_kwargs(self.viewer.add_points, uref, 2),
            )
            match_units(layer, uref)
            self._bind_points()
        self.viewer.layers.selection.active = layer
        layer.mode = "add"
        self.update_preview()
        return layer

    # ------------------------------------------------------------ geometry
    def reference_layer(self):
        """First checked layer: defines the preview windows."""
        layers = self.checked_layers()
        return layers[0] if layers else None

    def _crop_size_px(self, layer: Any) -> tuple[int, int]:
        cs = self.state.crop
        if cs.size_mode == "pixels":
            return window_size_px(cs.size, "pixels")
        layer_unit = napari_compat.layer_unit(layer) or self.state.calibration.unit
        size = convert(cs.size, self.state.calibration.unit, layer_unit)
        return window_size_px(size, "physical", float(layer.scale[-2]), float(layer.scale[-1]))

    def window_for(self, layer: Any, world_yx) -> CropWindow:
        """Crop window of ``layer`` around a world point (only the last 2 axes are used)."""
        full = world_point_for(self.viewer, world_yx)
        center = np.asarray(layer.world_to_data(full), float)[-2:]
        n = self._crop_size_px(layer)
        return crop_window(center, n, base_shape(layer)[-2:], self.state.crop.edge_mode)

    def _points_world(self) -> list[np.ndarray]:
        pts = self.points_layer()
        if pts is None or len(pts.data) == 0:
            return []
        return [np.asarray(pts.data_to_world(p), float) for p in pts.data]

    def _ensure_shapes(self, name: str, color: str) -> Shapes:
        layer = get_layer(self.viewer, name)
        if isinstance(layer, Shapes):
            return layer
        active = self.viewer.layers.selection.active
        uref = self.reference_layer() or self.kit.reference_layer()
        layer = self.viewer.add_shapes(
            ndim=2,
            name=name,
            edge_color=color,
            face_color="transparent",
            **units_kwargs(self.viewer.add_shapes, uref, 2),
        )
        match_units(layer, uref)
        if active is not None and active in self.viewer.layers:
            self.viewer.layers.selection.active = active
        return layer

    def _edge_width(self, ref: Any) -> float:
        # napari Shapes edge widths are in world units: ~1.5 source pixels.
        return 1.5 * float(max(ref.scale[-2], ref.scale[-1]))

    def update_preview(self) -> None:
        """Redraw the ``crop preview`` rectangles (exact, pixel-snapped windows)."""
        ref = self.reference_layer()
        points = self._points_world()
        if ref is not None:
            try:
                ny, nx = self._crop_size_px(ref)
                unit = napari_compat.layer_unit(ref) or self.state.calibration.unit
                py, px = float(ref.scale[-2]), float(ref.scale[-1])
                self.window_label.setText(
                    f"{ny} × {nx} px = {ny * py:g} × {nx * px:g} {display_unit(unit)} "
                    f"(on '{ref.name}')"
                )
            except ValueError as err:
                self.window_label.setText(str(err))
        else:
            self.window_label.setText("check a layer to crop")
        preview = get_layer(self.viewer, PREVIEW_LAYER)
        if not points and not isinstance(preview, Shapes):
            return
        rects = []
        if ref is not None:
            for p in points:
                try:
                    w = self.window_for(ref, p)
                except ValueError:
                    continue
                rects.append(window_rectangle_world(w, ref.scale[-2:], ref.translate[-2:]))
        if not isinstance(preview, Shapes):
            preview = self._ensure_shapes(PREVIEW_LAYER, "yellow")
        preview.data = []
        if rects:
            preview.add_rectangles(
                rects,
                edge_color="yellow",
                face_color="transparent",
                edge_width=self._edge_width(ref),
            )

    # ----------------------------------------------------------------- crop
    @guarded("Crop")
    def _on_crop_clicked(self, *args) -> None:
        self.crop()

    def crop(self) -> list:
        """Crop every checked layer around every point. Returns the new layers."""
        self._store()
        points = self._points_world()
        if not points:
            raise FigureKitError(
                f"add at least one point to the '{POINTS_LAYER}' layer (button above)"
            )
        layers = self.checked_layers()
        if not layers:
            raise FigureKitError("check at least one image layer to crop")
        ref = layers[0]
        for lay in layers[1:]:
            if not _compatible(lay, ref):
                raise FigureKitError(f"'{lay.name}' does not match '{ref.name}' (shape/scale)")
        if any(lay.multiscale for lay in layers):
            show_warning("Multiscale layer: cropping the full-resolution level (data[0]).")

        created: list = []
        outline_rects: list[np.ndarray] = []
        skipped: list[int] = []
        fell_back = padded = False
        for i, p in enumerate(points):
            try:
                windows = [self.window_for(lay, p) for lay in layers]
            except ValueError:
                skipped.append(i + 1)
                continue
            for lay, win in zip(layers, windows):
                created.append(self._make_crop(lay, win, i, p))
                fell_back |= win.fell_back
                padded |= win.is_padded
            outline_rects.append(
                window_rectangle_world(windows[0], ref.scale[-2:], ref.translate[-2:])
            )
        if skipped:
            show_warning(f"Point(s) {skipped} are outside the image and were skipped.")
        if not created:
            raise FigureKitError("no crop was created")
        if fell_back:
            show_warning(
                "The crop is larger than the image: 'shift' was replaced by 'clip' (smaller crop)."
            )
        if padded:
            show_info("Some crops were zero-padded; the padding is recorded in the metadata.")

        if self.state.crop.hide_sources:
            for lay in layers:
                lay.visible = False
        if self.state.crop.add_outlines and outline_rects:
            outlines = self._ensure_shapes(OUTLINES_LAYER, "white")
            outlines.add_rectangles(
                outline_rects,
                edge_color="white",
                face_color="transparent",
                edge_width=self._edge_width(ref),
            )
            outlines.visible = True
        if self.state.crop.zoom_to_first:
            self._zoom_to(outline_rects[0])
        first = created[0].metadata["figurekit"]["crop_index"]
        if self.kit.export_tab is not None:  # export the first crop group by default
            self.kit.export_tab.select_crop_group(first)
        n_pts = len(points) - len(skipped)
        show_info(f"Created {len(created)} crop layer(s) from {n_pts} point(s).")
        return created

    def _make_crop(self, layer: Any, win: CropWindow, index: int, point_world) -> Any:
        data = crop_array(base_data(layer), win)
        try:
            meta = copy.deepcopy(dict(layer.metadata))
        except Exception:  # noqa: BLE001
            meta = dict(layer.metadata)
        src_fk = fk_meta(layer)
        fk = dict(src_fk)
        fk.update(
            source_layer=layer.name,
            original_layer=src_fk.get("original_layer", src_fk.get("source_layer", layer.name)),
            channel_name=src_fk.get("channel_name", layer.name),
            pixel_size=[float(layer.scale[-2]), float(layer.scale[-1])],
            unit=napari_compat.layer_unit(layer) or self.state.calibration.unit,
            crop_index=index + 1,
            crop_origin_px=[int(win.origin[0]), int(win.origin[1])],
            crop_size_px=[int(win.shape[0]), int(win.shape[1])],
            center_world=[float(v) for v in point_world],
            edge_mode=win.mode,
            padding_px=[list(p) for p in win.padding] if win.is_padded else None,
            source_shape=list(base_shape(layer)),
        )
        meta["figurekit"] = fk
        new = self.viewer.add_image(
            data,
            name=f"{layer.name} crop{index + 1}",
            colormap=layer.colormap,
            contrast_limits=tuple(float(v) for v in layer.contrast_limits),
            gamma=float(layer.gamma),
            blending=layer.blending,
            opacity=float(layer.opacity),
            scale=tuple(float(s) for s in layer.scale),
            translate=crop_translate(layer.translate, layer.scale, win.origin),
            metadata=meta,
            rgb=False,
            **units_kwargs(self.viewer.add_image, layer, layer.ndim),
        )
        napari_compat.copy_layer_units(napari_compat.get_layer_units(layer), new)
        return new

    def _zoom_to(self, rect: np.ndarray) -> None:
        """Centre the camera on a world rectangle and zoom so it fills the view."""
        try:
            self.viewer.reset_view()
            fit_zoom = float(self.viewer.camera.zoom)
            ext = np.asarray(self.viewer.layers.extent.world, float)[:, -2:]
            full = np.maximum(ext[1] - ext[0], 1e-12)
            size = np.maximum(rect.max(axis=0) - rect.min(axis=0), 1e-12)
            center = (rect.max(axis=0) + rect.min(axis=0)) / 2.0
            cam_center = list(self.viewer.camera.center)
            cam_center[-2:] = [float(center[0]), float(center[1])]
            self.viewer.camera.center = tuple(cam_center)
            self.viewer.camera.zoom = fit_zoom * float(np.max(full) / np.max(size))
        except Exception as err:  # noqa: BLE001 - zoom is a convenience
            show_warning(f"Could not zoom to the crop: {err}")
