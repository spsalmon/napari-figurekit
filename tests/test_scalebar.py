import numpy as np
import pytest

from napari_figurekit.core import scalebar


def test_exact_bar_200px():
    geom = scalebar.compute_bar(10, "um", 0.1, "um", upscale=2)
    assert geom.source_px == 100 and geom.bar_px == 200 and geom.is_exact
    img = np.zeros((300, 400, 3), np.uint8)
    out, bar = scalebar.draw_scale_bar(
        img, geom.bar_px, thickness=6, color="#ffffff", label=geom.label, font_size=20
    )
    row = out[bar.y0 + 3]
    assert np.sum(np.all(row == 255, axis=-1)) == 200
    assert bar.x1 - bar.x0 == 200


def test_bar_in_other_unit():
    geom = scalebar.compute_bar(500, "nm", 0.05, "um")
    assert geom.source_px == 10
    assert geom.label == "500 nm"


def test_non_integer_reports_represented_length():
    geom = scalebar.compute_bar(10, "um", 0.26, "um")
    assert geom.exact_px == pytest.approx(38.4615, abs=1e-3)
    assert geom.source_px == 38
    assert geom.represented == pytest.approx(9.88)
    assert not geom.is_exact
    assert geom.label == "10 µm"  # label shows what the user typed
    assert geom.report() == "10 µm = 38.46 px → 38 px (9.88 µm)"
    assert "38 px = 9.88 µm" in geom.warning()
    assert geom.rel_error > scalebar.WARN_REL_ERROR


def test_too_short_bar_raises():
    with pytest.raises(ValueError):
        scalebar.compute_bar(1, "nm", 0.1, "um")


@pytest.mark.parametrize("position", scalebar.POSITIONS)
@pytest.mark.parametrize("box", [False, True])
def test_bar_inside_image_with_margin(position, box):
    h, w, margin = 120, 200, 15
    img = np.zeros((h, w, 3), np.uint8)
    out, bar = scalebar.draw_scale_bar(
        img,
        50,
        thickness=5,
        position=position,
        margin=margin,
        label="10 µm",
        font_size=14,
        box=box,
        box_color="#333333",
    )
    assert margin <= bar.x0 and bar.x1 <= w - margin
    assert margin <= bar.y0 and bar.y1 <= h - margin
    # Nothing drawn inside the margin band.
    drawn = np.any(out != 0, axis=-1)
    assert not drawn[:margin].any() and not drawn[h - margin :].any()
    assert not drawn[:, :margin].any() and not drawn[:, w - margin :].any()
    # Bar flush with the margin on its corner side.
    if position.endswith("right") and not box:
        assert bar.x1 == w - margin
    if position.endswith("left") and not box:
        assert bar.x0 == margin
    # The bar is exactly 50 px of solid colour.
    assert np.sum(np.all(out[bar.y0 + 2] == 255, axis=-1)) == 50


def test_label_above_bar_with_gap():
    img = np.zeros((200, 300, 3), np.uint8)
    out, bar = scalebar.draw_scale_bar(img, 100, thickness=4, label="10 µm", font_size=30)
    drawn_rows = np.where(np.any(out[:, :, 0] > 0, axis=1))[0]
    text_rows = drawn_rows[drawn_rows < bar.y0]
    assert len(text_rows) > 0
    assert bar.y0 - text_rows.max() - 1 == round(0.3 * 30)


def test_bar_too_long_raises():
    with pytest.raises(ValueError):
        scalebar.draw_scale_bar(np.zeros((50, 50, 3), np.uint8), 45, margin=10)


def test_parse_color():
    assert scalebar.parse_color("#ff0080") == (255, 0, 128)
    assert scalebar.parse_color("white") == (255, 255, 255)
    assert scalebar.parse_color((1.0, 0.0, 0.5)) == (255, 0, 128)
    assert scalebar.parse_color((10, 20, 30)) == (10, 20, 30)


def test_check_fits_matches_draw():
    kw = dict(thickness=5, margin=10, label="10 µm", font_size=20, box=True)
    scalebar.check_scale_bar_fits(200, 100, 80, **kw)  # fits
    with pytest.raises(ValueError):
        scalebar.check_scale_bar_fits(90, 100, 80, **kw)
    with pytest.raises(ValueError):
        scalebar.draw_scale_bar(np.zeros((100, 90, 3), np.uint8), 80, **kw)
