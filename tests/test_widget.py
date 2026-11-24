import json

import numpy as np
import pytest
from PIL import Image

from napari_figurekit._widget import FigureKitWidget
from napari_figurekit.state import FigureKitState, last_state_path


@pytest.fixture
def kit(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = FigureKitWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Figure Kit")
    yield viewer, widget
    widget.cleanup()


def _two_channel(viewer, shape=(2, 64, 64)):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4000, shape, dtype=np.uint16)
    return viewer.add_image(data, name="img"), data


def _split_and_calibrate(viewer, widget, pixel=0.5):
    setup = widget.setup_tab
    setup.source.set_layer("img")
    assert setup.axis_spin.value() == 0
    layers = setup.split_channels()
    setup.py_spin.setValue(pixel)
    setup.apply_calibration()
    return layers


def test_widget_instantiates(kit):
    viewer, widget = kit
    assert widget.tabs.count() == 5
    assert [widget.tabs.tabText(i) for i in range(5)] == [
        "Setup",
        "Crop",
        "Scale bar",
        "Annotate",
        "Export",
    ]


def test_split_channels(kit):
    viewer, widget = kit
    _, data = _two_channel(viewer)
    widget.setup_tab.source.set_layer("img")
    assert widget.setup_tab.table.rowCount() == 2
    widget.setup_tab.set_channel_rows(names=["dna", "actin"])
    new = widget.setup_tab.split_channels()
    assert len(new) == 2
    assert "img" not in [lay.name for lay in viewer.layers]
    assert [lay.name for lay in new] == ["dna", "actin"]
    assert all(str(lay.blending) == "additive" for lay in new)
    assert [lay.colormap.name for lay in new] == ["green", "magenta"]
    np.testing.assert_array_equal(new[1].data, data[1])
    assert new[0].metadata["figurekit"]["channel_name"] == "dna"


def test_split_keep_original_hidden(kit):
    viewer, widget = kit
    src, _ = _two_channel(viewer, (3, 32, 32))
    widget.setup_tab.source.set_layer("img")
    widget.setup_tab.remove_original.setChecked(False)
    new = widget.setup_tab.split_channels()
    assert len(new) == 3 and src in viewer.layers and not src.visible
    assert [lay.colormap.name for lay in new] == ["green", "magenta", "cyan"]


def test_rgb_split_disabled(kit):
    viewer, widget = kit
    viewer.add_image(np.zeros((16, 16, 3), np.uint8), rgb=True, name="photo")
    widget.setup_tab.source.set_layer("photo")
    assert not widget.setup_tab.split_btn.isEnabled()
    assert "RGB" in widget.setup_tab.split_btn.toolTip()


def test_apply_calibration_sets_scale(kit):
    viewer, widget = kit
    _two_channel(viewer)
    layers = _split_and_calibrate(viewer, widget, 0.25)
    for lay in layers:
        np.testing.assert_allclose(lay.scale, (0.25, 0.25))
        assert lay.metadata["figurekit"]["pixel_size"] == [0.25, 0.25]
        assert lay.metadata["figurekit"]["unit"] == "um"
        assert str(lay.units[-1]) == "micrometer"


def test_layer_combo_follows_renames(kit):
    viewer, widget = kit
    layer, _ = _two_channel(viewer)
    layer.name = "renamed"
    items = [widget.setup_tab.source.itemText(i) for i in range(widget.setup_tab.source.count())]
    assert items == ["renamed"]
    viewer.layers.remove(layer)
    assert widget.setup_tab.source.count() == 0


def _add_point(widget, world_yx):
    pts = widget.crop_tab.ensure_points_layer()
    pts.add(np.asarray([world_yx], float))
    return pts


def test_crop_creates_layers(kit):
    viewer, widget = kit
    _, data = _two_channel(viewer, (2, 200, 200))
    layers = _split_and_calibrate(viewer, widget, 0.5)
    widget.crop_tab.size_spin.setValue(20)  # 20 µm at 0.5 µm/px -> 40 px
    _add_point(widget, (50.0, 30.0))  # world µm -> data (100, 60)
    _add_point(widget, (5.0, 80.0))  # data (10, 160): window would start at y = -9
    preview = viewer.layers["crop preview"]
    assert len(preview.data) == 2
    crops = widget.crop_tab.crop()
    assert len(crops) == 4
    first = crops[0]
    assert first.data.shape == (40, 40)
    assert first.data.dtype == np.uint16
    fk = first.metadata["figurekit"]
    y0, x0 = fk["crop_origin_px"]
    assert (y0, x0) == (81, 41)  # start = round_half_up(c - (n - 1) / 2) = 100 - 19.5 -> 81
    np.testing.assert_array_equal(first.data, data[0][y0 : y0 + 40, x0 : x0 + 40])
    np.testing.assert_allclose(first.translate, np.asarray([y0, x0]) * 0.5)
    assert fk["source_layer"] == layers[0].name and fk["crop_size_px"] == [40, 40]
    # Second point is near the top edge: shifted inside, still 40 px.
    assert crops[2].data.shape == (40, 40)
    assert crops[2].metadata["figurekit"]["crop_origin_px"][0] == 0
    assert not layers[0].visible and not layers[1].visible
    assert len(viewer.layers["crop outlines"].data) == 2
    # Preview rectangle = exact cropped pixels.
    rect = preview.data[0]
    np.testing.assert_allclose(rect.min(axis=0), (np.array([y0, x0]) - 0.5) * 0.5)
    np.testing.assert_allclose(rect.max(axis=0), (np.array([y0, x0]) + 40 - 0.5) * 0.5)


def test_crop_nd_keeps_z(kit):
    viewer, widget = kit
    data = np.random.default_rng(1).random((5, 2, 64, 64)).astype(np.float32)
    viewer.add_image(data, name="img")
    setup = widget.setup_tab
    setup.source.set_layer("img")
    assert setup.axis_spin.value() == 1
    setup.split_channels()
    widget.crop_tab.mode_combo.setCurrentIndex(1)  # pixels
    widget.crop_tab.size_spin.setValue(16)
    _add_point(widget, (32.0, 32.0))
    crops = widget.crop_tab.crop()
    assert [c.data.shape for c in crops] == [(5, 16, 16), (5, 16, 16)]


def test_export_writes_png_and_json(kit, tmp_path):
    viewer, widget = kit
    _two_channel(viewer, (2, 200, 200))
    _split_and_calibrate(viewer, widget, 0.5)
    widget.crop_tab.size_spin.setValue(50)  # 100 px
    _add_point(widget, (50.0, 50.0))
    crops = widget.crop_tab.crop()
    ex = widget.export_tab
    assert ex.layer_list.checked_names() == [c.name for c in crops]  # default after crop
    ex.upscale_spin.setValue(2)
    ex.folder_edit.setText(str(tmp_path))
    ex.gray_check.setChecked(True)
    ex.montage_check.setChecked(True)
    sb = widget.scalebar_tab
    sb.length_spin.setValue(10)  # 10 µm at 0.5 µm/px = 20 px, x2 = 40 px
    sb.thick_spin.setValue(6)
    written = ex.export(blocking=True)
    names = sorted(p.name for p, _ in written)
    assert "img_crop1_merge.png" in names
    assert "img_crop1_montage.png" in names
    merge_path = tmp_path / "img_crop1_merge.png"
    img = np.asarray(Image.open(merge_path))
    assert img.shape == (200, 200, 3)
    assert Image.open(merge_path).info["dpi"][0] == pytest.approx(300, abs=0.01)
    side = json.loads((tmp_path / "img_crop1_merge.png.json").read_text())
    assert side["pixel_size"] == {"y": 0.5, "x": 0.5, "unit": "um", "unit_display": "µm"}
    assert side["upscale"] == 2
    assert side["scale_bar"]["bar_output_px"] == 40
    assert side["layers"][0]["crop"]["size_px"] == [100, 100]
    assert side["napari_version"] and side["plugin"]["name"] == "napari-figurekit"
    assert "timestamp" in side
    # Exact bar: 40 white pixels in a row of the bar (bottom right, 20 px margin).
    row = img[200 - 20 - 3]
    white = np.all(row == 255, axis=-1)
    assert white[200 - 20 - 40 : 200 - 20].all() and white.sum() >= 40
    # Montage: 2 channels + merge with 10 px gaps.
    mont = np.asarray(Image.open(tmp_path / "img_crop1_montage.png"))
    assert mont.shape == (200, 3 * 200 + 2 * 10, 3)
    # Never overwrite: second export gets _1 suffixes.
    again = ex.export(blocking=True)
    assert any(p.name == "img_crop1_merge_1.png" for p, _ in again)
    assert merge_path.exists()


def test_export_mismatch_error(kit, tmp_path):
    viewer, widget = kit
    _two_channel(viewer, (2, 200, 200))
    _split_and_calibrate(viewer, widget, 0.5)
    _add_point(widget, (30.0, 30.0))
    _add_point(widget, (70.0, 70.0))
    crops = widget.crop_tab.crop()
    ex = widget.export_tab
    ex.folder_edit.setText(str(tmp_path))
    ex.set_checked([c.name for c in crops])  # crops of different points
    with pytest.raises(Exception, match="does not match"):
        ex.export(blocking=True)


def test_preview_layer_overlays(kit):
    viewer, widget = kit
    _two_channel(viewer, (2, 64, 64))
    _split_and_calibrate(viewer, widget, 0.5)
    ex = widget.export_tab
    ex.upscale_spin.setValue(3)
    widget.scalebar_tab.length_spin.setValue(5)
    layer = ex.preview()
    assert layer.rgb and layer.data.shape == (192, 192, 3)
    np.testing.assert_allclose(layer.scale, (0.5 / 3, 0.5 / 3))
    # Output pixel 1 (centre of first 3x3 block) sits on source pixel 0.
    np.testing.assert_allclose(layer.data_to_world((1, 1)), (0, 0), atol=1e-12)
    assert layer.name not in [n for n, _ in ex.layer_list.entries()]
    again = ex.preview()
    assert sum(lay.name == "figurekit preview" for lay in viewer.layers) == 1
    assert again.rgb


def test_arrows_sync_and_export(kit, tmp_path):
    viewer, widget = kit
    _two_channel(viewer, (2, 100, 100))
    _split_and_calibrate(viewer, widget, 1.0)
    ann = widget.annotate_tab
    draw = ann.ensure_layers()
    draw.add_lines(np.array([[10.0, 10.0], [10.0, 50.0]]))
    vec = viewer.layers["arrows"]
    np.testing.assert_allclose(vec.data, [[[10, 10], [0, 40]]])
    widget.scalebar_tab.apply_combo.setCurrentIndex(2)  # none
    ann.color_pick.set_color("#ff0000")
    ann._changed()
    ex = widget.export_tab
    ex.folder_edit.setText(str(tmp_path))
    ex.set_checked([lay.name for lay in viewer.layers if "img ch" in lay.name])
    for lay in viewer.layers:
        if "img ch" in lay.name:
            lay.contrast_limits = (1e6, 2e6)  # black image
    written = ex.export(blocking=True)
    img = np.asarray(Image.open(written[0][0]))
    red = (img[..., 0] == 255) & (img[..., 1] == 0)
    assert red[10, 50] and not red[:, 51:].any()
    side = json.loads(written[0][1].read_text())
    assert side["scale_bar"] is None and side["arrows"]["world"]


def test_state_autosave_and_presets(kit, tmp_path):
    viewer, widget = kit
    widget.crop_tab.size_spin.setValue(77)
    widget.save_preset(tmp_path / "p.json")
    widget.crop_tab.size_spin.setValue(12)
    widget.load_preset(tmp_path / "p.json")
    assert widget.crop_tab.size_spin.value() == 77
    assert widget.state.crop.size == 77
    widget.cleanup()
    assert FigureKitState.load_json(last_state_path()).crop.size == 77


def test_corrupt_last_state_is_ignored(make_napari_viewer):
    path = last_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    viewer = make_napari_viewer()
    widget = FigureKitWidget(viewer)
    viewer.window.add_dock_widget(widget)
    assert widget.state.crop.size == 50.0
    widget.cleanup()


def test_scale_bar_report_and_overlay(kit):
    viewer, widget = kit
    _two_channel(viewer, (2, 256, 256))
    _split_and_calibrate(viewer, widget, 0.26)
    sb = widget.scalebar_tab
    sb.length_spin.setValue(10)
    assert sb.report_label.text() == "10 µm = 38.46 px → 38 px (9.88 µm)"
    assert "red" in sb.report_label.styleSheet()
    assert viewer.scale_bar.visible
    assert viewer.scale_bar.length == pytest.approx(10)
    assert sb.suggest() in (10.0, 20.0)


def test_sync_contrast(kit):
    viewer, widget = kit
    _two_channel(viewer, (2, 32, 32))
    _split_and_calibrate(viewer, widget, 1.0)
    _add_point(widget, (16.0, 16.0))
    crops = widget.crop_tab.crop()
    src = viewer.layers["img ch0"]
    src.contrast_limits = (100.0, 900.0)
    src.gamma = 0.8
    setup = widget.setup_tab
    setup.ref_combo.set_layer("img ch0")
    targets = setup.sync_contrast()
    assert crops[0] in targets
    assert tuple(crops[0].contrast_limits) == (100.0, 900.0) and crops[0].gamma == 0.8


def test_cells3d_sample_flow(kit, tmp_path):
    pytest.importorskip("skimage")
    from napari_figurekit._sample_data import cells3d_sample

    viewer, widget = kit
    for data, kwargs, _ in cells3d_sample():
        viewer.add_image(data, **kwargs)
    assert [lay.name for lay in viewer.layers] == ["membrane", "nuclei"]
    viewer.dims.set_point(0, 30 * 0.29)
    _add_point(widget, (33.0, 33.0))
    crops = widget.crop_tab.crop()  # default 50 µm at 0.26 -> 192 px
    assert crops[0].data.shape == (60, 192, 192)
    ex = widget.export_tab
    ex.set_checked([c.name for c in crops])
    ex.folder_edit.setText(str(tmp_path))
    written = ex.export(blocking=True)
    side = json.loads(written[0][1].read_text())
    assert side["layers"][0]["slice_indices"] == [30]
    assert written[0][0].name == "cells3d_crop1_merge.png"


def test_crop_groups_quick_select(kit):
    viewer, widget = kit
    _two_channel(viewer, (2, 100, 100))
    _split_and_calibrate(viewer, widget, 1.0)
    _add_point(widget, (30.0, 30.0))
    _add_point(widget, (70.0, 70.0))
    widget.crop_tab.crop()
    ex = widget.export_tab
    assert ex.layer_list.checked_names() == ["img ch0 crop1", "img ch1 crop1"]
    ex.select_crop_group(2)
    assert ex.layer_list.checked_names() == ["img ch0 crop2", "img ch1 crop2"]
    assert [ex.group_combo.itemText(i) for i in range(ex.group_combo.count())][1:] == [
        "crop 1",
        "crop 2",
    ]


def test_threaded_export(kit, tmp_path, qtbot):
    viewer, widget = kit
    _two_channel(viewer, (2, 64, 64))
    _split_and_calibrate(viewer, widget, 0.5)
    ex = widget.export_tab
    ex.folder_edit.setText(str(tmp_path))
    widget.scalebar_tab.length_spin.setValue(5)
    ex.upscale_spin.setValue(4)
    worker = ex.export(blocking=False)
    assert worker is not None and not ex.export_btn.isEnabled()
    qtbot.waitUntil(lambda: ex._worker is None, timeout=20000)
    assert ex.export_btn.isEnabled()
    assert (tmp_path / "img_full_merge.png").exists()
    assert (tmp_path / "img_full_merge.png.json").exists()


def test_scale_bar_too_large_is_reported_before_export(kit, tmp_path):
    viewer, widget = kit
    _two_channel(viewer, (2, 64, 64))
    _split_and_calibrate(viewer, widget, 0.5)
    ex = widget.export_tab
    out = tmp_path / "out"
    ex.folder_edit.setText(str(out))
    widget.scalebar_tab.length_spin.setValue(5)  # label + margins do not fit 64 px
    with pytest.raises(Exception, match="does not fit"):
        ex.export(blocking=False)
    assert ex.export_btn.isEnabled() and not out.exists()
