import numpy as np
import pytest

from napari_figurekit.core import crop


def test_window_size_physical_square():
    assert crop.window_size_px(50, "physical", 0.5, 0.5) == (100, 100)


def test_window_size_anisotropic():
    assert crop.window_size_px(50, "physical", 0.5, 0.25) == (100, 200)


def test_window_size_pixels():
    assert crop.window_size_px(64, "pixels") == (64, 64)


def test_window_size_too_small():
    with pytest.raises(ValueError):
        crop.window_size_px(0.1, "physical", 1.0, 1.0)


@pytest.mark.parametrize(
    "center, n, start",
    [
        (10, 5, 8),  # odd: pixels 8..12, centred on 10
        (11, 5, 9),  # next centre -> next window (no banker's rounding)
        (10.4, 5, 8),  # 10.4 lies in pixel 10 -> centred on pixel 10
        (10.6, 5, 9),  # 10.6 lies in pixel 11 -> centred on pixel 11
        (10, 4, 9),  # even, exact tie: pixels 9..12 (centre 10.5)
        (11, 4, 10),
        (9.9, 4, 8),  # pixels 8..11 (centre 9.5, nearest to 9.9)
        (10.4, 4, 9),  # centre 10.5
        (0, 1, 0),
        (7, 1, 7),
        (7.49, 1, 7),
        (7.51, 1, 8),
    ],
)
def test_centering_rule(center, n, start):
    assert crop.centered_start(center, n) == start


def _win(c, n=10, size=100, mode="shift"):
    return crop.crop_window(c, (n, n), (size, size), mode)


@pytest.mark.parametrize(
    "center, origin",
    [
        ((50, 50), (46, 46)),  # n = 10 -> start = round(50 - 4.5) = 46
        ((1, 50), (0, 46)),  # top
        ((98, 50), (90, 46)),  # bottom
        ((50, 1), (46, 0)),  # left
        ((50, 98), (46, 90)),  # right
        ((0, 0), (0, 0)),  # top-left corner
        ((0, 99), (0, 90)),  # top-right corner
        ((99, 0), (90, 0)),  # bottom-left corner
        ((99, 99), (90, 90)),  # bottom-right corner
    ],
)
def test_shift_inside(center, origin):
    w = _win(center)
    assert w.shape == (10, 10)
    assert w.origin == origin
    assert not w.is_padded and not w.fell_back


@pytest.mark.parametrize(
    "center, y, x",
    [
        ((1, 50), (0, 7), (46, 56)),  # top: requested -3..7
        ((98, 50), (94, 100), (46, 56)),  # bottom: requested 94..104
        ((50, 1), (46, 56), (0, 7)),  # left
        ((50, 98), (46, 56), (94, 100)),  # right
        ((0, 0), (0, 6), (0, 6)),  # corner: requested -4..6
        ((99, 99), (95, 100), (95, 100)),  # corner: requested 95..105
    ],
)
def test_clip(center, y, x):
    w = _win(center, mode="clip")
    assert (w.y.start, w.y.stop) == y
    assert (w.x.start, w.x.stop) == x
    assert not w.is_padded


@pytest.mark.parametrize(
    "center, origin, padding",
    [
        ((1, 50), (-3, 46), ((3, 0), (0, 0))),
        ((98, 50), (94, 46), ((0, 4), (0, 0))),
        ((50, 1), (46, -3), ((0, 0), (3, 0))),
        ((50, 98), (46, 94), ((0, 0), (0, 4))),
        ((0, 0), (-4, -4), ((4, 0), (4, 0))),
        ((99, 99), (95, 95), ((0, 5), (0, 5))),
    ],
)
def test_pad(center, origin, padding):
    w = _win(center, mode="pad")
    assert w.shape == (10, 10)
    assert w.origin == origin
    assert w.padding == padding


def test_pad_array_values_and_dtype():
    arr = np.arange(100 * 100, dtype=np.uint16).reshape(100, 100) + 1
    w = _win((0, 0), mode="pad")
    out = crop.crop_array(arr, w)
    assert out.shape == (10, 10) and out.dtype == np.uint16
    assert np.all(out[:4, :] == 0) and np.all(out[:, :4] == 0)
    np.testing.assert_array_equal(out[4:, 4:], arr[0:6, 0:6])


def test_shift_falls_back_to_clip_when_too_large():
    w = crop.crop_window((5, 5), (20, 20), (10, 10), "shift")
    assert w.fell_back and w.mode == "clip"
    assert w.shape == (10, 10)


def test_center_outside_raises():
    with pytest.raises(ValueError):
        _win((-3, 50))
    with pytest.raises(ValueError):
        _win((50, 100))


def test_crop_is_exact_slice_view():
    arr = np.random.default_rng(0).integers(0, 65535, (100, 120), dtype=np.uint16)
    w = crop.crop_window((40.2, 60.7), (21, 30), arr.shape, "shift")
    out = crop.crop_array(arr, w)
    y0, x0 = w.origin
    ref = arr[..., y0 : y0 + 21, x0 : x0 + 30]
    assert out.dtype == arr.dtype
    np.testing.assert_array_equal(out, ref)
    assert np.shares_memory(out, arr)


def test_nd_keeps_leading_axes():
    arr = np.zeros((7, 3, 64, 64), dtype=np.float32)
    w = crop.crop_window((32, 32), (16, 16), (64, 64))
    assert crop.crop_array(arr, w).shape == (7, 3, 16, 16)


def test_dask_stays_lazy():
    da = pytest.importorskip("dask.array")
    arr = da.zeros((5, 64, 64), dtype=np.uint8, chunks=(1, 32, 32))
    w = crop.crop_window((32, 32), (16, 16), (64, 64))
    out = crop.crop_array(arr, w)
    assert isinstance(out, da.Array) and out.shape == (5, 16, 16)
    wp = crop.crop_window((0, 0), (16, 16), (64, 64), "pad")
    outp = crop.crop_array(arr, wp)
    assert isinstance(outp, da.Array) and outp.shape == (5, 16, 16)


def test_crop_translate():
    t = crop.crop_translate((5.0, 10.0, 20.0), (2.0, 0.5, 0.25), (45, 30))
    assert t == pytest.approx((5.0, 10.0 + 45 * 0.5, 20.0 + 30 * 0.25))


def test_rectangle_follows_pixel_edges():
    w = crop.crop_window((50, 50), (10, 10), (100, 100))
    rect = crop.window_rectangle_world(w, (0.5, 0.5), (1.0, 2.0))
    # pixels 46..55 -> data edges 45.5..55.5
    assert rect.min(axis=0) == pytest.approx((45.5 * 0.5 + 1.0, 45.5 * 0.5 + 2.0))
    assert rect.max(axis=0) == pytest.approx((55.5 * 0.5 + 1.0, 55.5 * 0.5 + 2.0))
