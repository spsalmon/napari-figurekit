import pytest

from napari_figurekit.core import units


def test_500nm_is_half_micron():
    assert units.convert(500, "nm", "um") == pytest.approx(0.5)
    assert units.convert(0.5, "um", "nm") == pytest.approx(500)
    assert units.convert(1, "mm", "um") == pytest.approx(1000)


@pytest.mark.parametrize("text", ["µm", "μm", "um", "micron", "Microns", " micrometer ", "UM"])
def test_micron_spellings(text):
    assert units.parse_unit(text) == "um"


def test_bad_unit():
    with pytest.raises(ValueError):
        units.parse_unit("parsec")
    assert units.try_parse_unit("pixel") is None
    assert units.try_parse_unit(None) is None


def test_nice_length_for_cells3d_width():
    # 256 px × 0.26 µm = 66.56 µm; 20 % = 13.3 µm.
    assert units.suggest_bar_length(66.56) in (10.0, 20.0)


@pytest.mark.parametrize(
    "target, expected",
    [
        (13.3, 10.0),
        (17.0, 20.0),
        (3.0, 2.0),
        (3.3, 5.0),
        (4.0, 5.0),
        (0.04, 0.05),
        (700, 500),
        (800, 1000),
    ],
)
def test_nice_length(target, expected):
    assert units.nice_length(target) == pytest.approx(expected)


def test_format_length_uses_micro_sign():
    assert units.format_length(10, "um") == "10 µm"
    assert units.format_length(0.5, "mm") == "0.5 mm"
    assert units.display_unit("um") == "µm"


def test_round_half_up():
    assert units.round_half_up(2.5) == 3
    assert units.round_half_up(3.5) == 4
    assert units.round_half_up(-7.5) == -7
    assert units.round_half_up(2.4999999999999996) == 3
    assert units.round_half_up(38.46) == 38
