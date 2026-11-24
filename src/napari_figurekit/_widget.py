"""FigureKitWidget: the dock widget (tabs + preset bar) sharing one FigureKitState."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from napari.utils.notifications import show_info, show_warning
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .state import FigureKitState, last_state_path
from .ui._common import FigureKitError, LayerWatcher, guarded, image_layers
from .ui.annotate_tab import AnnotateTab
from .ui.crop_tab import CropTab
from .ui.export_tab import ExportTab
from .ui.scalebar_tab import ScaleBarTab
from .ui.setup_tab import SetupTab

if TYPE_CHECKING:
    import napari

logger = logging.getLogger(__name__)


def _save_state_quietly(state: FigureKitState) -> None:
    """Auto-save the state; never raises (called from close/teardown hooks)."""
    try:
        state.save_json(last_state_path())
    except Exception as err:  # noqa: BLE001
        logger.warning("Could not save Figure Kit state: %s", err)


def _scrollable(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(widget)
    return area


class FigureKitWidget(QWidget):
    """Dock widget with the Setup · Crop · Scale bar · Annotate · Export tabs."""

    def __init__(self, napari_viewer: napari.viewer.Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.state = FigureKitState()
        self._closed = False
        self._restore_last_state()
        self.watcher = LayerWatcher(napari_viewer, self)

        self.setMinimumWidth(340)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        # Construction order matters: Scale bar and Annotate are queried by Export.
        self.export_tab = None
        self.setup_tab = SetupTab(self)
        self.crop_tab = CropTab(self)
        self.scalebar_tab = ScaleBarTab(self)
        self.annotate_tab = AnnotateTab(self)
        self.export_tab = ExportTab(self)
        for tab, title in (
            (self.setup_tab, "Setup"),
            (self.crop_tab, "Crop"),
            (self.scalebar_tab, "Scale bar"),
            (self.annotate_tab, "Annotate"),
            (self.export_tab, "Export"),
        ):
            self.tabs.addTab(_scrollable(tab), title)

        bar = QHBoxLayout()
        self.save_btn = QPushButton("Save preset…")
        self.save_btn.setToolTip("Save all Figure Kit settings to a JSON file.")
        self.save_btn.clicked.connect(self._on_save_clicked)
        self.load_btn = QPushButton("Load preset…")
        self.load_btn.setToolTip("Load Figure Kit settings from a JSON file.")
        self.load_btn.clicked.connect(self._on_load_clicked)
        bar.addWidget(self.save_btn)
        bar.addWidget(self.load_btn)
        layout.addLayout(bar)

        self.scalebar_tab.refresh()
        # Auto-save when the widget goes away, whichever way that happens.
        state = self.state

        def saver(*args) -> None:  # signals may pass extra arguments
            _save_state_quietly(state)

        self.destroyed.connect(saver)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(saver)

    # ------------------------------------------------------------ shared
    def reference_layer(self):
        """Layer defining pixel size/shape for the scale bar: first export layer, else
        the first visible image layer, else the first image layer."""
        if self.export_tab is not None:
            layers = self.export_tab.checked_layers()
            if layers:
                return layers[0]
        layers = [lay for lay in image_layers(self.viewer) if not lay.rgb]
        visible = [lay for lay in layers if lay.visible]
        return (visible or layers or [None])[0]

    def on_calibration_changed(self) -> None:
        self.crop_tab.on_calibration_changed()
        self.scalebar_tab.on_calibration_changed()

    def on_export_changed(self) -> None:
        if getattr(self, "scalebar_tab", None) is not None:
            self.scalebar_tab.refresh()

    # ------------------------------------------------------------ presets
    def _restore_last_state(self) -> None:
        path = last_state_path()
        if not path.exists():
            return
        try:
            self.state.update_from(FigureKitState.load_json(path))
        except Exception as err:  # noqa: BLE001 - corrupt file: ignore with a warning
            show_warning(f"Ignoring unreadable Figure Kit state {path}: {err}")

    def load_state_into_ui(self) -> None:
        for tab in (
            self.setup_tab,
            self.crop_tab,
            self.scalebar_tab,
            self.annotate_tab,
            self.export_tab,
        ):
            tab.load_state()

    def save_preset(self, path) -> None:
        self.state.save_json(path)

    def load_preset(self, path) -> None:
        try:
            new = FigureKitState.load_json(path)
        except Exception as err:  # noqa: BLE001
            raise FigureKitError(f"could not read preset {path}: {err}") from err
        self.state.update_from(new)
        self.load_state_into_ui()

    @guarded("Save preset")
    def _on_save_clicked(self, *args) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Figure Kit preset", "figurekit_preset.json", "JSON (*.json)"
        )
        if path:
            self.save_preset(path)
            show_info(f"Preset saved to {path}")

    @guarded("Load preset")
    def _on_load_clicked(self, *args) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Figure Kit preset", "", "JSON (*.json);;All files (*)"
        )
        if path:
            self.load_preset(path)
            show_info(f"Preset loaded from {path}")

    # ------------------------------------------------------------ teardown
    def cleanup(self) -> None:
        """Save state and disconnect from viewer events (idempotent)."""
        if self._closed:
            return
        self._closed = True
        _save_state_quietly(self.state)
        self.watcher.disconnect_all()
        for tab in (self.setup_tab, self.crop_tab, self.annotate_tab):
            tab.disconnect_all()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.cleanup()
        super().closeEvent(event)
