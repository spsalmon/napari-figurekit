"""Tab 5 — Export: deterministic composites with exact scale bar + JSON sidecars."""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from napari.utils.notifications import show_error, show_info, show_warning
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..core import napari_compat
from ..core.annotations import data_to_output_px
from ..core.export import printed_size_mm
from ..core.pipeline import RESERVED_PANELS, ExportJob, render_job, write_job
from ..core.render import ChannelSpec, extract_2d, slice_indices
from ..core.scalebar import check_scale_bar_fits
from ..core.units import display_unit
from ._common import (
    PREVIEW_LAYER,
    FigureKitError,
    LayerChecklist,
    base_data,
    base_shape,
    combo,
    combo_value,
    fk_meta,
    get_layer,
    guarded,
    image_layers,
    ispin,
    match_units,
    set_combo_value,
    units_kwargs,
    world_point_for,
)

if TYPE_CHECKING:
    from .._widget import FigureKitWidget

logger = logging.getLogger(__name__)


def _colormap_name(layer: Any) -> str:
    return str(getattr(layer.colormap, "name", layer.colormap))


class ExportTab(QWidget):
    """Composite the checked layers and write PNG/TIFF panels with sidecars."""

    def __init__(self, kit: FigureKitWidget) -> None:
        super().__init__()
        self.kit = kit
        self.viewer = kit.viewer
        self.state = kit.state
        self._worker: Any = None
        ex = self.state.export

        layout = QVBoxLayout(self)
        lbox = QGroupBox("Layers to composite")
        lv = QVBoxLayout(lbox)
        self.layer_list = LayerChecklist()
        self.layer_list.setToolTip("Layers merged into the panels (same Y/X shape and position).")
        self.layer_list.setMinimumHeight(90)
        lv.addWidget(self.layer_list)
        self.visible_btn = QPushButton("Select visible layers")
        self.visible_btn.setToolTip("Check exactly the image layers that are currently visible.")
        self.visible_btn.clicked.connect(self.select_visible)
        self.group_combo = QComboBox()
        self.group_combo.setToolTip("Check all layers of one crop (one point) at once.")
        self.group_combo.activated.connect(self._on_group_activated)
        row = QHBoxLayout()
        row.addWidget(self.visible_btn, 1)
        row.addWidget(self.group_combo, 1)
        lv.addLayout(row)
        self.maxproj_check = QCheckBox("Max projection along Z")
        self.maxproj_check.setChecked(ex.max_projection)
        self.maxproj_check.setToolTip(
            "Project the third-to-last axis instead of using the current slice."
        )
        lv.addWidget(self.maxproj_check)
        layout.addWidget(lbox)

        obox = QGroupBox("Outputs")
        of = QFormLayout(obox)
        self.merge_check = QCheckBox("Merge (additive composite)")
        self.merge_check.setChecked(ex.merge)
        self.merge_check.setToolTip("All channels blended additively.")
        self.colored_check = QCheckBox("Each channel, colored")
        self.colored_check.setChecked(ex.colored)
        self.colored_check.setToolTip("One panel per channel with its colormap.")
        self.gray_check = QCheckBox("Each channel, grayscale")
        self.gray_check.setChecked(ex.gray)
        self.gray_check.setToolTip("One grayscale panel per channel (same contrast limits).")
        self.montage_check = QCheckBox("Montage (channels + merge)")
        self.montage_check.setChecked(ex.montage)
        self.montage_check.setToolTip("Colored channels and the merge side by side in one row.")
        for w in (self.merge_check, self.colored_check, self.gray_check, self.montage_check):
            of.addRow(w)
        self.gap_spin = ispin(ex.montage_gap, 0, 1000, "Gap between montage panels.", " px")
        self.gap_color = combo(
            [("white", "white"), ("black", "black")], ex.gap_color, "Montage gap color."
        )
        row = QHBoxLayout()
        row.addWidget(self.gap_spin, 1)
        row.addWidget(self.gap_color)
        of.addRow("Montage gap", row)
        layout.addWidget(obox)

        fbox = QGroupBox("File")
        ff = QFormLayout(fbox)
        self.upscale_spin = ispin(ex.upscale, 1, 10, "Integer nearest-neighbour upscale.", " ×")
        ff.addRow("Upscale", self.upscale_spin)
        self.dpi_spin = ispin(ex.dpi, 1, 10000, "Resolution written into the file.", " dpi")
        ff.addRow("DPI", self.dpi_spin)
        self.size_label = QLabel("–")
        self.size_label.setWordWrap(True)
        self.size_label.setToolTip("Output size in pixels and printed width at this DPI.")
        ff.addRow("Output size", self.size_label)
        self.fmt_combo = combo(
            [("PNG (8-bit RGB)", "png"), ("TIFF (8-bit RGB)", "tiff")], ex.fmt, "File format."
        )
        ff.addRow("Format", self.fmt_combo)
        self.folder_edit = QLineEdit(ex.folder)
        self.folder_edit.setPlaceholderText("choose a folder…")
        self.folder_edit.setToolTip("Folder where panels and sidecars are written.")
        self.folder_btn = QPushButton("…")
        self.folder_btn.setFixedWidth(28)
        self.folder_btn.setToolTip("Choose the output folder.")
        self.folder_btn.clicked.connect(self._on_folder_clicked)
        row = QHBoxLayout()
        row.addWidget(self.folder_edit, 1)
        row.addWidget(self.folder_btn)
        ff.addRow("Folder", row)
        self.pattern_edit = QLineEdit(ex.pattern)
        self.pattern_edit.setToolTip(
            "File name: {source}, {crop}, {panel}, {date}; existing files are never overwritten."
        )
        ff.addRow("File name", self.pattern_edit)
        layout.addWidget(fbox)

        self.preview_btn = QPushButton("Preview")
        self.preview_btn.setToolTip("Show the merged render (with annotations) as a layer.")
        self.preview_btn.clicked.connect(self._on_preview_clicked)
        self.export_btn = QPushButton("Export")
        self.export_btn.setToolTip("Write all selected panels and their JSON sidecars.")
        self.export_btn.clicked.connect(self._on_export_clicked)
        row = QHBoxLayout()
        row.addWidget(self.preview_btn)
        row.addWidget(self.export_btn)
        layout.addLayout(row)
        self.canvas_btn = QPushButton("Export canvas (napari export_figure)")
        self.canvas_btn.setToolTip("Quick screenshot of all layers via napari (not pixel exact).")
        self.canvas_btn.clicked.connect(self._on_canvas_clicked)
        layout.addWidget(self.canvas_btn)
        layout.addStretch(1)

        for w in (
            self.merge_check,
            self.colored_check,
            self.gray_check,
            self.montage_check,
            self.maxproj_check,
        ):
            w.toggled.connect(self._changed)
        for w in (self.gap_spin, self.upscale_spin, self.dpi_spin):
            w.valueChanged.connect(self._changed)
        for w in (self.gap_color, self.fmt_combo):
            w.currentIndexChanged.connect(self._changed)
        self.folder_edit.editingFinished.connect(self._changed)
        self.pattern_edit.editingFinished.connect(self._changed)
        self.layer_list.toggled.connect(self._on_list_toggled)
        kit.watcher.changed.connect(self._refresh_list)
        self._refresh_list()

    # -------------------------------------------------------------- state I/O
    def _store(self) -> None:
        ex = self.state.export
        ex.merge = self.merge_check.isChecked()
        ex.colored = self.colored_check.isChecked()
        ex.gray = self.gray_check.isChecked()
        ex.montage = self.montage_check.isChecked()
        ex.montage_gap = self.gap_spin.value()
        ex.gap_color = combo_value(self.gap_color)
        ex.upscale = self.upscale_spin.value()
        ex.dpi = self.dpi_spin.value()
        ex.fmt = combo_value(self.fmt_combo)
        ex.folder = self.folder_edit.text().strip()
        ex.pattern = self.pattern_edit.text().strip() or "{source}_{crop}_{panel}"
        ex.max_projection = self.maxproj_check.isChecked()

    def load_state(self) -> None:
        ex = self.state.export
        widgets = [
            self.merge_check,
            self.colored_check,
            self.gray_check,
            self.montage_check,
            self.maxproj_check,
            self.gap_spin,
            self.upscale_spin,
            self.dpi_spin,
            self.gap_color,
            self.fmt_combo,
        ]
        for w in widgets:
            w.blockSignals(True)
        self.merge_check.setChecked(ex.merge)
        self.colored_check.setChecked(ex.colored)
        self.gray_check.setChecked(ex.gray)
        self.montage_check.setChecked(ex.montage)
        self.maxproj_check.setChecked(ex.max_projection)
        self.gap_spin.setValue(ex.montage_gap)
        self.upscale_spin.setValue(ex.upscale)
        self.dpi_spin.setValue(ex.dpi)
        set_combo_value(self.gap_color, ex.gap_color)
        set_combo_value(self.fmt_combo, ex.fmt)
        for w in widgets:
            w.blockSignals(False)
        self.folder_edit.setText(ex.folder)
        self.pattern_edit.setText(ex.pattern)
        self._update_size_label()

    def _changed(self, *args) -> None:
        self._store()
        self._update_size_label()
        self.kit.on_export_changed()

    # ---------------------------------------------------------- layer list
    def _refresh_list(self) -> None:
        prev = dict(self.layer_list.entries())
        entries = []
        for lay in image_layers(self.viewer):
            if lay.rgb:
                entries.append(
                    (lay.name, False, False, "RGB layers have no colormap; not composited.")
                )
                continue
            checked = prev[lay.name] if lay.name in prev else bool(lay.visible)
            entries.append((lay.name, checked, True, ""))
        self.layer_list.set_items(entries)
        self._refresh_groups()
        self._on_list_toggled()

    def _on_list_toggled(self) -> None:
        try:
            self._update_size_label()
            self.kit.on_export_changed()
        except RuntimeError:
            pass

    def _crop_groups(self) -> dict[int, list[str]]:
        groups: dict[int, list[str]] = {}
        for lay in image_layers(self.viewer):
            idx = fk_meta(lay).get("crop_index")
            if isinstance(idx, int) and not lay.rgb:
                groups.setdefault(idx, []).append(lay.name)
        return dict(sorted(groups.items()))

    def _refresh_groups(self) -> None:
        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        self.group_combo.addItem("select crop…", None)
        for idx in self._crop_groups():
            self.group_combo.addItem(f"crop {idx}", idx)
        self.group_combo.blockSignals(False)

    def _on_group_activated(self, index: int) -> None:
        idx = self.group_combo.itemData(index)
        if idx is not None:
            self.select_crop_group(int(idx))

    def select_crop_group(self, index: int) -> None:
        """Check exactly the crop layers made from point ``index`` (1-based)."""
        self.layer_list.set_checked(self._crop_groups().get(index, []))

    def select_visible(self, *args) -> None:
        self.layer_list.set_checked(
            [lay.name for lay in image_layers(self.viewer) if lay.visible and not lay.rgb]
        )

    def set_checked(self, names: list[str]) -> None:
        self.layer_list.set_checked(names)

    def checked_layers(self) -> list:
        out = []
        for name in self.layer_list.checked_names():
            layer = get_layer(self.viewer, name)
            if layer is not None:
                out.append(layer)
        return out

    def _update_size_label(self) -> None:
        layers = self.checked_layers()
        if not layers:
            self.size_label.setText("no layer checked")
            return
        h, w = base_shape(layers[0])[-2:]
        k, dpi = self.state.export.upscale, self.state.export.dpi
        mm_w, mm_h = printed_size_mm(w * k, h * k, dpi)
        self.size_label.setText(
            f"{w * k} × {h * k} px (W × H) → {mm_w:.1f} × {mm_h:.1f} mm at {dpi} dpi"
        )

    # ------------------------------------------------------------ build job
    def _validate(self, layers: list) -> None:
        if not layers:
            raise FigureKitError("check at least one image layer to export")
        ref = layers[0]
        rshape = base_shape(ref)[-2:]
        rscale = np.asarray(ref.scale[-2:], float)
        rtrans = np.asarray(ref.translate[-2:], float)
        for lay in layers[1:]:
            problems = []
            if base_shape(lay)[-2:] != rshape:
                problems.append(f"Y/X shape {base_shape(lay)[-2:]} ≠ {rshape}")
            if not np.allclose(np.asarray(lay.scale[-2:], float), rscale):
                problems.append("pixel size differs")
            if not np.allclose(
                np.asarray(lay.translate[-2:], float), rtrans, atol=1e-9 * float(rscale.max())
            ):
                problems.append("position (translate) differs")
            if problems:
                raise FigureKitError(
                    f"'{lay.name}' does not match '{ref.name}': {'; '.join(problems)}. "
                    "Tip: check only the crops of one point together."
                )
        dims = self.viewer.dims
        nd = dims.ndim
        if dims.ndisplay != 2 or tuple(dims.displayed) != (nd - 2, nd - 1):
            raise FigureKitError(
                "export needs the 2D view of the last two axes (Y, X); switch to 2D / reset "
                "the axis order"
            )

    def build_job(self) -> ExportJob:
        """Collect slices, display settings and annotations (GUI thread)."""
        self._store()
        ex, sbs = self.state.export, self.state.scale_bar
        layers = self.checked_layers()
        self._validate(layers)
        ref = layers[0]
        point = np.asarray(self.viewer.dims.point, float)
        warnings: list[str] = []
        channels: list[ChannelSpec] = []
        layer_info: list[dict] = []
        used: set[str] = set()
        for lay in layers:
            data = base_data(lay)
            if lay.multiscale:
                warnings.append(f"'{lay.name}' is multiscale: exported from level 0")
            data_pt = np.asarray(lay.world_to_data(point), float)
            idx = slice_indices(data_pt, data.shape)
            proj_axis = data.ndim - 3 if (ex.max_projection and data.ndim >= 3) else None
            plane = extract_2d(data, idx, proj_axis)
            fk = fk_meta(lay)
            name = str(fk.get("channel_name") or lay.name)
            if name in RESERVED_PANELS:
                name = f"{name}_ch"
            base, i = name, 2
            while name in used:
                name, i = f"{base}_{i}", i + 1
            used.add(name)
            gamma = float(lay.gamma)
            if gamma != 1.0:
                warnings.append(f"'{lay.name}' uses gamma {gamma:g} (non-linear display)")
            channels.append(
                ChannelSpec(
                    data=plane,
                    contrast_limits=(float(lay.contrast_limits[0]), float(lay.contrast_limits[1])),
                    colormap=lay.colormap,
                    gamma=gamma,
                    opacity=float(lay.opacity),
                    name=name,
                )
            )
            layer_info.append(
                {
                    "name": lay.name,
                    "channel_name": name,
                    "source_layer": fk.get("source_layer"),
                    "original_layer": fk.get("original_layer"),
                    "source_path": str(getattr(lay.source, "path", None) or fk.get("source_path"))
                    if (getattr(lay.source, "path", None) or fk.get("source_path"))
                    else None,
                    "dtype": str(data.dtype),
                    "data_shape": list(data.shape),
                    "slice_indices": list(idx),
                    "max_projection_axis": proj_axis,
                    "colormap": _colormap_name(lay),
                    "contrast_limits": [float(v) for v in lay.contrast_limits],
                    "gamma": gamma,
                    "opacity": float(lay.opacity),
                    "blending": str(lay.blending),
                    "scale": [float(s) for s in lay.scale],
                    "translate": [float(t) for t in lay.translate],
                    "crop": {
                        "origin_px": fk.get("crop_origin_px"),
                        "size_px": fk.get("crop_size_px"),
                        "center_world": fk.get("center_world"),
                        "edge_mode": fk.get("edge_mode"),
                        "padding_px": fk.get("padding_px"),
                        "source_shape": fk.get("source_shape"),
                    }
                    if "crop_origin_px" in fk
                    else None,
                }
            )

        unit = napari_compat.layer_unit(ref) or self.state.calibration.unit
        py, px = float(ref.scale[-2]), float(ref.scale[-1])
        bar = None
        bar_label = None
        if sbs.apply_to != "none":
            bar = self.kit.scalebar_tab.geometry(ex.upscale)
            if bar is not None:
                bar_label = self.kit.scalebar_tab.label_text(bar)
                if bar.warning():
                    warnings.append(bar.warning())

        arrows_px = []
        arrows_world = []
        for tail_w, tip_w in self.kit.annotate_tab.arrows_world():
            tail_d = np.asarray(ref.world_to_data(world_point_for(self.viewer, tail_w)))[-2:]
            tip_d = np.asarray(ref.world_to_data(world_point_for(self.viewer, tip_w)))[-2:]
            arrows_px.append(
                (data_to_output_px(tail_d, ex.upscale), data_to_output_px(tip_d, ex.upscale))
            )
            arrows_world.append({"tail": list(tail_w), "tip": list(tip_w)})

        rfk = fk_meta(ref)
        source = rfk.get("original_layer") or rfk.get("source_layer") or ref.name
        crop_idx = rfk.get("crop_index")
        name_fields = {
            "source": str(source),
            "crop": f"crop{crop_idx}" if crop_idx else "full",
            "date": datetime.now().strftime("%Y%m%d"),
        }
        sidecar = {
            "plugin": {"name": "napari-figurekit", "version": __version__},
            "napari_version": napari_compat.napari_version(),
            "pixel_size": {"y": py, "x": px, "unit": unit, "unit_display": display_unit(unit)},
            "output_pixel_size": {"y": py / ex.upscale, "x": px / ex.upscale, "unit": unit},
            "slice": {
                "viewer_dims_point": [float(v) for v in point],
                "max_projection_z": bool(ex.max_projection),
            },
            "layers": layer_info,
            "scale_bar": None
            if bar is None
            else {
                **asdict(sbs),
                "label_text": bar_label,
                "exact_source_px": bar.exact_px,
                "bar_source_px": bar.source_px,
                "bar_output_px": bar.bar_px,
                "represented_length": bar.represented,
            },
            "arrows": {**asdict(self.state.arrows), "world": arrows_world, "output_px": arrows_px}
            if arrows_px
            else None,
            "warnings": warnings,
        }
        if bar is not None:
            h, w = (n * ex.upscale for n in base_shape(ref)[-2:])
            try:
                check_scale_bar_fits(w, h, bar.bar_px, label=bar_label, **asdict(sbs))
            except ValueError as err:
                raise FigureKitError(str(err)) from err
        return ExportJob(
            channels=channels,
            merge=ex.merge,
            colored=ex.colored,
            gray=ex.gray,
            montage=ex.montage,
            montage_gap=ex.montage_gap,
            gap_color=ex.gap_color,
            upscale=ex.upscale,
            dpi=ex.dpi,
            fmt=ex.fmt,
            folder=ex.folder,
            pattern=ex.pattern,
            bar=bar,
            bar_label=bar_label,
            scale_bar=asdict(sbs),
            arrows_px=arrows_px,
            arrow_style=asdict(self.state.arrows),
            name_fields=name_fields,
            sidecar=sidecar,
        )

    # ---------------------------------------------------------------- preview
    @guarded("Preview")
    def _on_preview_clicked(self, *args) -> None:
        self.preview()

    def preview(self):
        """Add/replace the ``figurekit preview`` RGB layer with the merged render."""
        job = self.build_job()
        for w in job.sidecar["warnings"]:
            show_warning(w)
        try:
            rgb = render_job(job, panels=["merge"])["merge"]
        except ValueError as err:
            raise FigureKitError(str(err)) from err
        ref = self.checked_layers()[0]
        k = job.upscale
        s = np.asarray(ref.scale[-2:], float)
        t = np.asarray(ref.translate[-2:], float)
        scale = s / k
        translate = t + s * (0.5 / k - 0.5)  # output pixel centres onto source pixels
        layer = get_layer(self.viewer, PREVIEW_LAYER)
        if layer is not None:
            self.viewer.layers.remove(layer)
        layer = self.viewer.add_image(
            rgb,
            rgb=True,
            name=PREVIEW_LAYER,
            scale=tuple(scale),
            translate=tuple(translate),
            metadata={"figurekit": {"role": "preview"}},
            blending="translucent",
            **units_kwargs(self.viewer.add_image, ref, 2),
        )
        match_units(layer, ref)
        show_info("Preview added as layer 'figurekit preview' (hide or delete it when done).")
        return layer

    # ----------------------------------------------------------------- export
    @guarded("Export")
    def _on_export_clicked(self, *args) -> None:
        self.export(blocking=False)

    def _check_folder(self) -> None:
        if not self.state.export.folder:
            folder = QFileDialog.getExistingDirectory(self, "Choose output folder")
            if not folder:
                raise FigureKitError("choose an output folder")
            self.folder_edit.setText(folder)
            self._store()

    def export(self, blocking: bool = True):
        """Render and write all panels.

        Parameters
        ----------
        blocking : bool
            True: run in this thread and return the written ``(image, sidecar)`` paths.
            False: run in a napari ``thread_worker`` (button disabled meanwhile).
        """
        self._store()
        if not self.state.export.folder:
            if blocking:
                raise FigureKitError("choose an output folder")
            self._check_folder()  # file picker
        job = self.build_job()
        if not (job.merge or job.colored or job.gray or job.montage):
            raise FigureKitError("select at least one output (merge, channels, montage)")
        for w in job.sidecar["warnings"]:
            show_warning(w)
        if blocking:
            written = _run(job)
            self._report(written)
            return written
        from napari.qt.threading import thread_worker

        # ignore_errors: failures are reported through the errored signal only.
        worker = thread_worker(_run, ignore_errors=True)(job)
        worker.returned.connect(self._report)
        worker.errored.connect(lambda err: show_error(f"Export failed: {err}"))
        worker.finished.connect(self._export_finished)
        self.export_btn.setEnabled(False)
        self.export_btn.setText("Exporting…")
        self._worker = worker
        worker.start()
        return worker

    def _export_finished(self) -> None:
        try:
            self.export_btn.setEnabled(True)
            self.export_btn.setText("Export")
        except RuntimeError:
            pass
        self._worker = None

    def _report(self, written) -> None:
        if not written:
            show_warning("Nothing was written")
            return
        folder = written[0][0].parent
        names = "\n".join(p.name for p, _ in written)
        show_info(f"Exported {len(written)} panel(s) + JSON sidecars to {folder}:\n{names}")

    # ------------------------------------------------------------ misc
    def _on_folder_clicked(self, *args) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Choose output folder", self.folder_edit.text()
        )
        if folder:
            self.folder_edit.setText(folder)
            self._changed()

    @guarded("Export canvas")
    def _on_canvas_clicked(self, *args) -> None:
        if not hasattr(self.viewer, "export_figure"):
            raise FigureKitError("this napari version has no export_figure")
        start = str(Path(self.state.export.folder or Path.home()) / "canvas.png")
        path, _ = QFileDialog.getSaveFileName(self, "Export canvas", start, "PNG (*.png)")
        if not path:
            return
        self.viewer.export_figure(path, scale_factor=self.state.export.upscale)
        show_info(f"Canvas exported to {path} (napari rendering; not pixel exact)")


def _run(job: ExportJob):
    try:
        images = render_job(job)
        return write_job(job, images)
    except ValueError as err:
        raise FigureKitError(str(err)) from err
