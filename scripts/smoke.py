"""End-to-end smoke test: split → calibrate → crop → annotate → export (headless).

Run:  QT_QPA_PLATFORM=offscreen python scripts/smoke.py [output_folder]
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Keep the smoke run from touching the user's saved settings.
os.environ.setdefault("NAPARI_FIGUREKIT_HOME", tempfile.mkdtemp(prefix="figurekit-smoke-"))

import napari  # noqa: E402
import numpy as np  # noqa: E402
from skimage import data as skdata  # noqa: E402

from napari_figurekit._widget import FigureKitWidget  # noqa: E402


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="figurekit-"))
    viewer = napari.Viewer(show=False)

    # 1. The plugin's sample data contribution loads.
    viewer.open_sample("napari-figurekit", "cells3d")
    assert [lay.name for lay in viewer.layers] == ["membrane", "nuclei"], viewer.layers
    viewer.layers.clear()

    # 2. A raw, uncalibrated ZCYX stack, as a colleague would open it.
    viewer.add_image(skdata.cells3d(), name="cells3d")
    kit = FigureKitWidget(viewer)
    viewer.window.add_dock_widget(kit, name="Figure Kit", area="right")

    setup = kit.setup_tab
    setup.source.set_layer("cells3d")
    assert setup.axis_spin.value() == 1
    setup.set_channel_rows(names=["membrane", "nuclei"])
    channels = setup.split_channels()
    setup.source.set_layer("membrane")
    setup.py_spin.setValue(0.26)
    setup.z_check.setChecked(True)
    setup.z_spin.setValue(0.29)
    setup.apply_calibration()
    print("scale:", [tuple(round(s, 3) for s in lay.scale) for lay in channels])

    # 3. Two crop centres, 30 µm crops.
    crop = kit.crop_tab
    crop.size_spin.setValue(30)
    pts = crop.ensure_points_layer()
    pts.add(np.array([[20.0, 25.0], [45.0, 40.0]]))
    crops = crop.crop()
    print("crops:", [(c.name, c.data.shape) for c in crops])

    # 4. An arrow and a scale bar.
    ann = kit.annotate_tab
    draw = ann.ensure_layers()
    c = crops[0].metadata["figurekit"]["center_world"]
    draw.add_lines(np.array([[c[0] - 5, c[1] - 5], [c[0], c[1]]]))
    kit.scalebar_tab.suggest()

    # 5. Export the crops of point 1, middle Z slice.
    viewer.dims.set_point(0, 30 * 0.29)
    ex = kit.export_tab
    ex.set_checked([lay.name for lay in crops if lay.name.endswith("crop1")])
    ex.folder_edit.setText(str(out))
    ex.upscale_spin.setValue(3)
    for box in (ex.colored_check, ex.gray_check, ex.montage_check):
        box.setChecked(True)
    ex.preview()
    written = ex.export(blocking=True)
    for img, side in written:
        meta = json.loads(side.read_text())
        print(f"{img}  {meta['output_size_px']}  bar={meta['scale_bar']['bar_output_px']} px")
    kit.cleanup()
    viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
