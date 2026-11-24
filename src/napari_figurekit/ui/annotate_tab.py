"""Tab 4 — Annotate: draw arrows as lines, mirrored live on a Vectors layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from napari.layers import Shapes, Vectors
from qtpy.QtWidgets import QFormLayout, QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget

from ..core.annotations import lines_to_vectors
from ._common import (
    combo,
    combo_value,
    dspin,
    get_layer,
    guarded,
    ispin,
    match_units,
    set_combo_value,
    units_kwargs,
)
from .scalebar_tab import ColorPicker

if TYPE_CHECKING:
    from .._widget import FigureKitWidget

DRAW_LAYER = "arrows (draw)"
VECTORS_LAYER = "arrows"
APPLY_ITEMS = [("all panels", "all"), ("merge only", "merge"), ("none", "none")]


class AnnotateTab(QWidget):
    """Arrows drawn tail → tip; exported crisp with PIL at output resolution."""

    def __init__(self, kit: FigureKitWidget) -> None:
        super().__init__()
        self.kit = kit
        self.viewer = kit.viewer
        self.state = kit.state
        self._draw: Any = None
        ar = self.state.arrows

        layout = QVBoxLayout(self)
        self.new_btn = QPushButton("New arrow layer")
        self.new_btn.setToolTip("Create (or select) the 'arrows (draw)' layer in line mode.")
        self.new_btn.clicked.connect(self._on_new_clicked)
        layout.addWidget(self.new_btn)
        help_label = QLabel("Draw lines from tail to tip. The arrow head goes at the end point.")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        box = QGroupBox("Arrow style (export)")
        form = QFormLayout(box)
        self.color_pick = ColorPicker(ar.color, "Arrow color.", (("yellow", "#ffff00"),))
        form.addRow("Color", self.color_pick)
        self.width_spin = ispin(ar.width, 1, 200, "Shaft width in exported pixels.", " px")
        form.addRow("Width", self.width_spin)
        self.hl_spin = dspin(ar.head_length, 0.5, 20, 2, 0.5, "Head length as a multiple of width.")
        form.addRow("Head length (× width)", self.hl_spin)
        self.hw_spin = dspin(ar.head_width, 0.5, 20, 2, 0.5, "Head width as a multiple of width.")
        form.addRow("Head width (× width)", self.hw_spin)
        self.apply_combo = combo(APPLY_ITEMS, ar.apply_to, "Which exported panels get arrows.")
        form.addRow("Arrows on", self.apply_combo)
        layout.addWidget(box)
        note = QLabel("The exported arrows may look slightly different from the napari preview.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        layout.addWidget(note)
        layout.addStretch(1)

        self.color_pick.callbacks.append(self._changed)
        for w in (self.width_spin, self.hl_spin, self.hw_spin):
            w.valueChanged.connect(self._changed)
        self.apply_combo.currentIndexChanged.connect(self._changed)
        kit.watcher.changed.connect(self._bind_draw)
        self._bind_draw()

    # -------------------------------------------------------------- state I/O
    def _store(self) -> None:
        ar = self.state.arrows
        ar.color = self.color_pick.color()
        ar.width = self.width_spin.value()
        ar.head_length = self.hl_spin.value()
        ar.head_width = self.hw_spin.value()
        ar.apply_to = combo_value(self.apply_combo)

    def load_state(self) -> None:
        ar = self.state.arrows
        for w in (self.width_spin, self.hl_spin, self.hw_spin, self.apply_combo):
            w.blockSignals(True)
        self.color_pick.set_color(ar.color)
        self.width_spin.setValue(ar.width)
        self.hl_spin.setValue(ar.head_length)
        self.hw_spin.setValue(ar.head_width)
        set_combo_value(self.apply_combo, ar.apply_to)
        for w in (self.width_spin, self.hl_spin, self.hw_spin, self.apply_combo):
            w.blockSignals(False)
        self._restyle()

    def _changed(self, *args) -> None:
        self._store()
        self._restyle()

    # ---------------------------------------------------------------- layers
    def draw_layer(self):
        layer = get_layer(self.viewer, DRAW_LAYER)
        return layer if isinstance(layer, Shapes) else None

    def vectors_layer(self):
        layer = get_layer(self.viewer, VECTORS_LAYER)
        return layer if isinstance(layer, Vectors) else None

    def _bind_draw(self) -> None:
        layer = self.draw_layer()
        if layer is self._draw:
            return
        self.disconnect_all()
        self._draw = layer
        if layer is not None:
            layer.events.data.connect(self._on_draw_data)

    def disconnect_all(self) -> None:
        if self._draw is not None:
            try:
                self._draw.events.data.disconnect(self._on_draw_data)
            except Exception:  # noqa: BLE001
                pass
        self._draw = None

    def _world_width(self) -> float:
        ref = self.kit.reference_layer()
        px = float(ref.scale[-1]) if ref is not None else 1.0
        return max(self.state.arrows.width, 1) * px / max(self.state.export.upscale, 1)

    @guarded("New arrow layer")
    def _on_new_clicked(self, *args) -> None:
        self.ensure_layers()

    def ensure_layers(self) -> Shapes:
        """Create (or reuse) the vectors + draw layers and activate line drawing."""
        if self.vectors_layer() is None:
            vec = self.viewer.add_vectors(
                np.zeros((0, 2, 2)),
                name=VECTORS_LAYER,
                vector_style="arrow",
                edge_color=self.state.arrows.color,
                edge_width=self._world_width(),
                **units_kwargs(self.viewer.add_vectors, self.kit.reference_layer(), 2),
            )
            match_units(vec, self.kit.reference_layer())
        draw = self.draw_layer()
        if draw is None:
            draw = self.viewer.add_shapes(
                ndim=2,
                name=DRAW_LAYER,
                edge_color=self.state.arrows.color,
                face_color="transparent",
                edge_width=self._world_width(),
                opacity=0.3,
                **units_kwargs(self.viewer.add_shapes, self.kit.reference_layer(), 2),
            )
            match_units(draw, self.kit.reference_layer())
            self._bind_draw()
        draw.current_edge_color = self.state.arrows.color
        draw.current_edge_width = self._world_width()
        self.viewer.layers.selection.active = draw
        draw.mode = "add_line"
        return draw

    def _restyle(self) -> None:
        vec = self.vectors_layer()
        if vec is not None:
            try:
                vec.edge_color = self.state.arrows.color
                vec.edge_width = self._world_width()
            except Exception:  # noqa: BLE001 - cosmetic only
                pass

    def _on_draw_data(self, *args) -> None:
        try:
            self.sync_vectors()
        except RuntimeError:  # widget deleted
            pass

    def _lines(self) -> list[np.ndarray]:
        draw = self.draw_layer()
        if draw is None:
            return []
        return [
            np.asarray(d, float)
            for d, kind in zip(draw.data, draw.shape_type)
            if kind in ("line", "path")
        ]

    def sync_vectors(self) -> None:
        """Mirror the drawn lines as arrows (``[start, end - start]``, world coords)."""
        draw = self.draw_layer()
        if draw is None:
            return
        vec_data = lines_to_vectors(self._lines(), draw.scale, draw.translate)
        vec = self.vectors_layer()
        if vec is None:
            active = self.viewer.layers.selection.active
            vec = self.viewer.add_vectors(
                vec_data,
                name=VECTORS_LAYER,
                vector_style="arrow",
                edge_color=self.state.arrows.color,
                edge_width=self._world_width(),
                **units_kwargs(
                    self.viewer.add_vectors, self.kit.reference_layer(), vec_data.shape[-1]
                ),
            )
            match_units(vec, self.kit.reference_layer())
            if active is not None and active in self.viewer.layers:
                self.viewer.layers.selection.active = active
        else:
            if vec_data.shape[-1] != vec.data.shape[-1] and len(vec.data):
                return
            vec.data = vec_data
            self._restyle()

    def arrows_world(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """All arrows as (tail, tip) world coordinates (from the draw layer, else vectors)."""
        draw = self.draw_layer()
        if draw is not None:
            vecs = lines_to_vectors(self._lines(), draw.scale, draw.translate)
        else:
            vec = self.vectors_layer()
            if vec is None or len(vec.data) == 0:
                return []
            raw = np.asarray(vec.data, float)
            s = np.asarray(vec.scale, float)
            t = np.asarray(vec.translate, float)
            vecs = np.stack([raw[:, 0] * s + t, raw[:, 1] * s], axis=1)
        return [(v[0], v[0] + v[1]) for v in vecs]
