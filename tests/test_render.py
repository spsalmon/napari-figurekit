import numpy as np
import pytest

from napari_figurekit.core import render


class LinearCmap:
    """Minimal colormap: black → color (like napari's 'green', 'magenta')."""

    def __init__(self, rgb):
        self.rgb = np.asarray(rgb, float)

    def map(self, v):
        v = np.asarray(v, float)[:, None]
        return np.concatenate([v * self.rgb, np.ones_like(v)], axis=1)


GREEN = LinearCmap((0, 1, 0))
MAGENTA = LinearCmap((1, 0, 1))


def spec(data, cmap, lo=0, hi=1, gamma=1.0, name="c"):
    return render.ChannelSpec(np.asarray(data), (lo, hi), cmap, gamma=gamma, name=name)


def test_green_plus_magenta_is_white():
    full = np.full((4, 4), 100.0)
    rgb = render.composite_additive(
        [
            render.render_channel(spec(full, GREEN, 0, 100)),
            render.render_channel(spec(full, MAGENTA, 0, 100)),
        ]
    )
    out = render.to_uint8(rgb)
    assert out.dtype == np.uint8
    assert np.all(out == 255)


def test_contrast_limits_map_exactly():
    data = np.array([[10, 20, 30, 5, 50]], dtype=np.uint16)
    out = render.to_uint8(render.render_gray(spec(data, GREEN, 10, 30)))
    assert out[0, 0, 0] == 0  # lo -> 0
    assert out[0, 2, 0] == 255  # hi -> 255
    assert out[0, 1, 0] == 128  # linear midpoint: round(127.5) = 128
    assert out[0, 3, 0] == 0 and out[0, 4, 0] == 255  # clipped


def test_gamma_one_is_linear():
    x = np.linspace(0, 1, 11)
    np.testing.assert_allclose(render.normalize(x, (0, 1), 1.0), x)
    np.testing.assert_allclose(render.normalize(x, (0, 1), 2.0), x**2)


def test_equal_contrast_limits_no_division_by_zero():
    v = render.normalize(np.array([1.0, 2.0, 3.0]), (2, 2))
    np.testing.assert_array_equal(v, [0, 1, 1])


def test_upscale_nearest_neighbour_blocks():
    img = np.random.default_rng(1).integers(0, 255, (5, 7, 3), dtype=np.uint8)
    up = render.upscale(img, 3)
    assert up.shape == (15, 21, 3)
    for y in range(5):
        for x in range(7):
            block = up[3 * y : 3 * y + 3, 3 * x : 3 * x + 3]
            assert np.all(block == img[y, x])


def test_upscale_rejects_non_integer():
    with pytest.raises(ValueError):
        render.upscale(np.zeros((2, 2, 3), np.uint8), 0)


def test_opacity_scales_channel():
    rgb = render.render_channel(render.ChannelSpec(np.ones((2, 2)), (0, 1), GREEN, opacity=0.5))
    assert rgb[0, 0, 1] == pytest.approx(0.5)


def test_napari_colormap_object_works():
    from napari.utils.colormaps import ensure_colormap

    out = render.to_uint8(render.render_channel(spec(np.ones((2, 2)), ensure_colormap("magenta"))))
    assert tuple(out[0, 0]) == (255, 0, 255)


def test_render_panels_keys_and_montage():
    a = spec(np.ones((4, 6)), GREEN, name="nuc")
    b = spec(np.ones((4, 6)), MAGENTA, name="mem")
    panels = render.render_panels([a, b], merge=True, colored=True, gray=True, upscale_factor=2)
    assert list(panels) == ["merge", "nuc", "nuc_gray", "mem", "mem_gray"]
    assert all(p.shape == (8, 12, 3) for p in panels.values())
    m = render.montage([panels["nuc"], panels["mem"], panels["merge"]], gap=3, gap_color=(0, 0, 0))
    assert m.shape == (8, 12 * 3 + 6, 3)
    assert np.all(m[:, 12:15] == 0)


def test_slice_indices_and_projection():
    data = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
    idx = render.slice_indices((1.4, 9.0, 2, 2), data.shape)
    assert idx == (1, 3)  # second index clipped
    plane = render.extract_2d(data, idx)
    np.testing.assert_array_equal(plane, data[1, 3])
    proj = render.extract_2d(data, idx, max_project_axis=1)
    np.testing.assert_array_equal(proj, data[1].max(axis=0))
