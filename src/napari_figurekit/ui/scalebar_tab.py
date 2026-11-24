"""Tab 3 — Scale bar: exact export settings + live napari overlay preview."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from napari.utils.notifications import show_info
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
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

from ..core import napari_compat
from ..core.scalebar import WARN_REL_ERROR, BarGeometry, compute_bar
from ..core.units import UNITS, convert, display_unit, suggest_bar_length
from ._common import (
    FigureKitError,
    base_shape,
    combo,
    combo_value,
    dspin,
    guarded,
    ispin,
    set_combo_value,
)

if TYPE_CHECKING:
    from .._widget import FigureKitWidget

POSITION_ITEMS = [
    ("bottom right", "bottom_right"),
    ("bottom left", "bottom_left"),
    ("top right", "top_right"),
    ("top left", "top_left"),
]
APPLY_ITEMS = [("all panels", "all"), ("merge only", "merge"), ("none", "none")]
UNIT_ITEMS = [(display_unit(u), u) for u in UNITS]


class ColorPicker(QWidget):
    """White / black / custom color selector (custom opens a QColorDialog)."""

    def __init__(self, color: str, tip: str, extra: tuple[tuple[str, str], ...] = ()) -> None:
        super().__init__()
        self._custom = color
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox()
        self.combo.setMinimumWidth(64)
        self._presets = (("white", "#ffffff"), ("black", "#000000")) + extra
        for label, value in self._presets:
            self.combo.addItem(label, value)
        self.combo.addItem("custom…", "custom")
        self.combo.setToolTip(tip)
        self.swatch = QPushButton()
        self.swatch.setFixedWidth(28)
        self.swatch.setToolTip("Pick a custom color.")
        self.swatch.clicked.connect(self._pick)
        lay.addWidget(self.combo, 1)
        lay.addWidget(self.swatch)
        self.callbacks: list = []
        self.combo.activated.connect(self._on_activated)
        self.set_color(color)

    def _on_activated(self, index: int) -> None:
        if self.combo.itemData(index) == "custom":
            self._pick()
        else:
            self._update_swatch()
            self._notify()

    def _pick(self, *args) -> None:
        col = QColorDialog.getColor(QColor(self.color()), self, "Choose color")
        if col.isValid():
            self.set_color(col.name())
            self._notify()
        else:
            self.set_color(self.color())

    def _notify(self) -> None:
        for cb in self.callbacks:
            cb()

    def color(self) -> str:
        data = self.combo.currentData()
        return self._custom if data == "custom" else str(data)

    def set_color(self, color: str) -> None:
        color = str(color).lower()
        values = [v for _, v in self._presets]
        self.combo.blockSignals(True)
        if color in values:
            self.combo.setCurrentIndex(values.index(color))
        else:
            self._custom = color
            self.combo.setCurrentIndex(self.combo.count() - 1)
        self.combo.blockSignals(False)
        self._update_swatch()

    def _update_swatch(self) -> None:
        self.swatch.setStyleSheet(f"background-color: {self.color()}; border: 1px solid gray;")


class ScaleBarTab(QWidget):
    """Scale bar settings (export truth) mirrored onto the napari overlay (preview)."""

    def __init__(self, kit: FigureKitWidget) -> None:
        super().__init__()
        self.kit = kit
        self.viewer = kit.viewer
        self.state = kit.state
        sb = self.state.scale_bar
        self._last_cal_unit = self.state.calibration.unit

        layout = QVBoxLayout(self)
        box = QGroupBox("Scale bar")
        form = QFormLayout(box)
        self.visible_check = QCheckBox("Visible")
        self.visible_check.setChecked(sb.visible)
        self.visible_check.setToolTip("Show the scale bar on the canvas preview.")
        form.addRow(self.visible_check)

        self.length_spin = dspin(sb.length, 1e-6, 1e9, 4, 1.0, "Bar length in the chosen unit.")
        self.unit_combo = combo(UNIT_ITEMS, sb.unit, "Unit of the bar (may differ from pixels).")
        self.suggest_btn = QPushButton("Suggest")
        self.suggest_btn.setToolTip("Pick a nice length (1, 2, 5 × 10^k) ≈ 20 % of the width.")
        self.suggest_btn.clicked.connect(self._on_suggest_clicked)
        row = QHBoxLayout()
        row.addWidget(self.length_spin, 1)
        row.addWidget(self.unit_combo)
        form.addRow("Length", row)
        form.addRow("", self.suggest_btn)

        self.report_label = QLabel("–")
        self.report_label.setWordWrap(True)
        self.report_label.setToolTip("Exact bar length in pixels (red: rounding error > 0.5 %).")
        form.addRow("Exact bar", self.report_label)

        self.label_check = QCheckBox("Show text")
        self.label_check.setChecked(sb.show_label)
        self.label_check.setToolTip("Draw the length label above the bar.")
        self.label_edit = QLineEdit(sb.custom_label)
        self.label_edit.setPlaceholderText("auto, e.g. 10 µm")
        self.label_edit.setToolTip("Custom label text; leave empty for the typed length.")
        row = QHBoxLayout()
        row.addWidget(self.label_check)
        row.addWidget(self.label_edit, 1)
        form.addRow("Label", row)

        self.font_spin = ispin(sb.font_size, 4, 500, "Label font size in exported pixels.", " px")
        self.pt_label = QLabel()
        self.pt_label.setToolTip("Printed font size at the export DPI (pt = px × 72 / dpi).")
        form.addRow("Font size", self.font_spin)
        form.addRow("", self.pt_label)
        self.font_edit = QLineEdit(sb.font_path)
        self.font_edit.setPlaceholderText("default (DejaVu Sans)")
        self.font_edit.setToolTip("Optional path to a .ttf font file for labels.")
        self.font_btn = QPushButton("…")
        self.font_btn.setFixedWidth(28)
        self.font_btn.setToolTip("Choose a TrueType font file.")
        self.font_btn.clicked.connect(self._on_font_clicked)
        row = QHBoxLayout()
        row.addWidget(self.font_edit, 1)
        row.addWidget(self.font_btn)
        form.addRow("Font file", row)

        self.thick_spin = ispin(sb.thickness, 1, 500, "Bar thickness in exported pixels.", " px")
        form.addRow("Thickness", self.thick_spin)
        self.color_pick = ColorPicker(sb.color, "Bar and text color.")
        form.addRow("Color", self.color_pick)
        self.pos_combo = combo(POSITION_ITEMS, sb.position, "Corner of the image for the bar.")
        form.addRow("Position", self.pos_combo)
        self.margin_spin = ispin(sb.margin, 0, 2000, "Distance from the image border.", " px")
        form.addRow("Margin", self.margin_spin)
        self.box_check = QCheckBox("box")
        self.box_check.setChecked(sb.box)
        self.box_check.setToolTip("Draw a filled box behind bar and label.")
        self.box_pick = ColorPicker(sb.box_color, "Background box color.")
        row = QHBoxLayout()
        row.addWidget(self.box_check)
        row.addWidget(self.box_pick, 1)
        form.addRow("Background", row)
        self.apply_combo = combo(APPLY_ITEMS, sb.apply_to, "Which exported panels get a bar.")
        form.addRow("Scale bar on", self.apply_combo)
        layout.addWidget(box)
        note = QLabel("The canvas overlay is a preview; the exported file is the source of truth.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        layout.addWidget(note)
        layout.addStretch(1)

        for w in (self.visible_check, self.label_check, self.box_check):
            w.toggled.connect(self._changed)
        for w in (self.length_spin, self.font_spin, self.thick_spin, self.margin_spin):
            w.valueChanged.connect(self._changed)
        for w in (self.unit_combo, self.pos_combo, self.apply_combo):
            w.currentIndexChanged.connect(self._changed)
        self.label_edit.editingFinished.connect(self._changed)
        self.font_edit.editingFinished.connect(self._changed)
        self.color_pick.callbacks.append(self._changed)
        self.box_pick.callbacks.append(self._changed)
        kit.watcher.changed.connect(self.refresh)
        self.refresh()

    # -------------------------------------------------------------- state I/O
    def _store(self) -> None:
        sb = self.state.scale_bar
        sb.visible = self.visible_check.isChecked()
        sb.length = self.length_spin.value()
        sb.unit = combo_value(self.unit_combo)
        sb.show_label = self.label_check.isChecked()
        sb.custom_label = self.label_edit.text()
        sb.font_size = self.font_spin.value()
        sb.font_path = self.font_edit.text().strip()
        sb.thickness = self.thick_spin.value()
        sb.color = self.color_pick.color()
        sb.position = combo_value(self.pos_combo)
        sb.margin = self.margin_spin.value()
        sb.box = self.box_check.isChecked()
        sb.box_color = self.box_pick.color()
        sb.apply_to = combo_value(self.apply_combo)

    def load_state(self) -> None:
        sb = self.state.scale_bar
        widgets = [
            self.visible_check,
            self.length_spin,
            self.unit_combo,
            self.label_check,
            self.label_edit,
            self.font_spin,
            self.font_edit,
            self.thick_spin,
            self.pos_combo,
            self.margin_spin,
            self.box_check,
            self.apply_combo,
        ]
        for w in widgets:
            w.blockSignals(True)
        self.visible_check.setChecked(sb.visible)
        self.length_spin.setValue(sb.length)
        set_combo_value(self.unit_combo, sb.unit)
        self.label_check.setChecked(sb.show_label)
        self.label_edit.setText(sb.custom_label)
        self.font_spin.setValue(sb.font_size)
        self.font_edit.setText(sb.font_path)
        self.thick_spin.setValue(sb.thickness)
        self.color_pick.set_color(sb.color)
        set_combo_value(self.pos_combo, sb.position)
        self.margin_spin.setValue(sb.margin)
        self.box_check.setChecked(sb.box)
        self.box_pick.set_color(sb.box_color)
        set_combo_value(self.apply_combo, sb.apply_to)
        for w in widgets:
            w.blockSignals(False)
        self._last_cal_unit = self.state.calibration.unit
        self.refresh()

    def _changed(self, *args) -> None:
        self._store()
        self.refresh()

    # --------------------------------------------------------------- helpers
    def pixel_info(self) -> tuple[Any, float, str] | None:
        """(reference layer, X pixel size, pixel unit), or None without image layers."""
        ref = self.kit.reference_layer()
        if ref is None:
            return None
        unit = napari_compat.layer_unit(ref) or self.state.calibration.unit
        return ref, float(ref.scale[-1]), unit

    def geometry(self, upscale: int | None = None) -> BarGeometry | None:
        """Exact bar geometry for the reference layer (None if there is no layer)."""
        info = self.pixel_info()
        if info is None:
            return None
        _, pixel_size, unit = info
        sb = self.state.scale_bar
        k = self.state.export.upscale if upscale is None else upscale
        return compute_bar(sb.length, sb.unit, pixel_size, unit, k)

    def label_text(self, geom: BarGeometry) -> str | None:
        sb = self.state.scale_bar
        if not sb.show_label:
            return None
        return sb.custom_label.strip() or geom.label

    def on_calibration_changed(self) -> None:
        """Bar unit follows the calibration unit (keeping the physical length)."""
        new = self.state.calibration.unit
        if new != self._last_cal_unit and self.state.scale_bar.unit == self._last_cal_unit:
            length = convert(self.state.scale_bar.length, self._last_cal_unit, new)
            self.length_spin.blockSignals(True)
            self.length_spin.setValue(length)
            self.length_spin.blockSignals(False)
            self.unit_combo.blockSignals(True)
            set_combo_value(self.unit_combo, new)
            self.unit_combo.blockSignals(False)
            self._store()
        self._last_cal_unit = new
        self.refresh()

    def refresh(self, *args) -> None:
        """Update the exact-bar report, the pt helper and the napari overlay."""
        try:
            sb = self.state.scale_bar
            dpi = max(int(self.state.export.dpi), 1)
            self.pt_label.setText(f"= {sb.font_size * 72 / dpi:.1f} pt at {dpi} dpi")
            try:
                geom = self.geometry()
            except ValueError as err:
                geom = None
                self.report_label.setText(str(err))
                self.report_label.setStyleSheet("color: red;")
            else:
                if geom is None:
                    self.report_label.setText("no image layer")
                    self.report_label.setStyleSheet("")
                else:
                    self.report_label.setText(geom.report())
                    bad = geom.rel_error > WARN_REL_ERROR
                    self.report_label.setStyleSheet("color: red;" if bad else "")
            info = self.pixel_info()
            world_unit = info[2] if info else self.state.calibration.unit
            napari_compat.apply_scale_bar(self.viewer, sb, world_unit)
        except RuntimeError:  # widget deleted during teardown
            pass

    # ---------------------------------------------------------------- actions
    @guarded("Suggest length")
    def _on_suggest_clicked(self, *args) -> None:
        self.suggest()

    def suggest(self) -> float:
        info = self.pixel_info()
        if info is None:
            raise FigureKitError("no image layer to measure")
        ref, pixel_size, unit = info
        width = base_shape(ref)[-1] * pixel_size
        bar_unit = combo_value(self.unit_combo)
        length = suggest_bar_length(convert(width, unit, bar_unit))
        self.length_spin.setValue(length)
        show_info(f"Suggested scale bar: {length:g} {display_unit(bar_unit)}")
        return length

    def _on_font_clicked(self, *args) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose font", "", "TrueType fonts (*.ttf *.otf);;All files (*)"
        )
        if path:
            self.font_edit.setText(path)
            self._changed()
