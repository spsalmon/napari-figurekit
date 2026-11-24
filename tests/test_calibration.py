import numpy as np
import pytest
import tifffile

from napari_figurekit.core import calibration


def test_default_channel_axis():
    assert calibration.default_channel_axis((60, 2, 256, 256)) == 1
    assert calibration.default_channel_axis((3, 64, 64)) == 0
    assert calibration.default_channel_axis((5, 4, 64, 64)) == 1  # smallest size wins
    assert calibration.default_channel_axis((60, 256, 256)) == -1
    assert calibration.default_channel_axis((256, 256)) == -1


def test_updated_scale_keeps_other_axes():
    assert calibration.updated_scale((2.0, 1.0, 1.0, 1.0), 0.5, 0.25) == (2.0, 1.0, 0.5, 0.25)
    assert calibration.updated_scale((2.0, 1.0, 1.0, 1.0), 0.5, 0.5, 0.3) == (2.0, 0.3, 0.5, 0.5)
    assert calibration.updated_scale((1.0, 1.0), 0.5, 0.5, 0.3) == (0.5, 0.5)


def test_rescaled_translate_keeps_pixel_offsets():
    t = calibration.rescaled_translate((10.0, 20.0), (1.0, 1.0), (0.5, 0.25))
    assert t == pytest.approx((5.0, 5.0))


def test_drop_axis():
    assert calibration.drop_axis((0.3, 1.0, 0.2, 0.2), 1) == (0.3, 0.2, 0.2)


def test_read_imagej(tmp_path):
    path = tmp_path / "ij.tif"
    data = np.zeros((4, 32, 32), np.uint16)
    tifffile.imwrite(
        path,
        data,
        imagej=True,
        resolution=(1 / 0.13, 1 / 0.13),
        metadata={"unit": "micron", "spacing": 0.5, "axes": "ZYX"},
    )
    cal = calibration.read_tiff_calibration(path)
    assert cal is not None and cal.source == "imagej"
    assert cal.pixel_size_x == pytest.approx(0.13, rel=1e-5)
    assert cal.pixel_size_y == pytest.approx(0.13, rel=1e-5)
    assert cal.unit == "um" and cal.z_step == pytest.approx(0.5)


def test_read_ome(tmp_path):
    path = tmp_path / "x.ome.tif"
    data = np.zeros((3, 16, 16), np.uint8)
    tifffile.imwrite(
        path,
        data,
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeX": 65.0,
            "PhysicalSizeXUnit": "nm",
            "PhysicalSizeY": 65.0,
            "PhysicalSizeYUnit": "nm",
            "PhysicalSizeZ": 0.2,
            "PhysicalSizeZUnit": "µm",
        },
    )
    cal = calibration.read_tiff_calibration(path)
    assert cal is not None and cal.source == "ome"
    assert cal.unit == "nm" and cal.pixel_size_x == pytest.approx(65.0)
    assert cal.z_step == pytest.approx(200.0)  # converted to the XY unit


def test_no_calibration(tmp_path):
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((8, 8), np.uint8))
    assert calibration.read_tiff_calibration(path) is None
    assert calibration.read_tiff_calibration(tmp_path / "missing.png") is None
