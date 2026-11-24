import numpy as np
import pytest

from napari_figurekit.core import annotations


def test_line_to_vector():
    vec = annotations.lines_to_vectors([np.array([[10, 10], [10, 50]])])
    np.testing.assert_array_equal(vec, [[[10, 10], [0, 40]]])


def test_line_to_vector_with_scale_translate():
    vec = annotations.lines_to_vectors([np.array([[10, 10], [10, 50]])], (0.5, 2), (1, 1))
    np.testing.assert_allclose(vec, [[[6, 21], [0, 80]]])


def test_empty_lines():
    assert annotations.lines_to_vectors([]).shape == (0, 2, 2)


def test_data_to_output_px():
    assert annotations.data_to_output_px((10, 50), 1) == (10, 50)
    assert annotations.data_to_output_px((10, 50), 3) == (31, 151)


@pytest.mark.parametrize("upscale", [1, 3])
def test_arrow_tip_pixel(upscale):
    img = np.zeros((100 * upscale, 100 * upscale, 3), np.uint8)
    tail = annotations.data_to_output_px((10, 10), upscale)
    tip = annotations.data_to_output_px((10, 50), upscale)
    out = annotations.draw_arrows(img, [(tail, tip)], color="#ff0000", width=2 * upscale)
    red = out[..., 0] == 255
    ty, tx = int(tip[0]), int(tip[1])
    assert red[ty, tx]  # tip pixel is drawn
    assert not red[:, tx + 1 :].any()  # nothing beyond the tip
    assert red[ty, int(tail[1])]  # shaft starts at the tail
    cols = np.where(red.any(axis=0))[0]
    assert cols.max() == tx


def test_arrow_polygon_tip_exact():
    shaft, head = annotations.arrow_polygons((0, 0), (30, 40), width=2)
    assert head[0] == (40.0, 30.0)  # (x, y) tip
    assert shaft is not None
