import json

from napari_figurekit.state import FigureKitState, last_state_path


def test_roundtrip_default(tmp_path):
    s = FigureKitState()
    assert FigureKitState.from_dict(s.to_dict()) == s
    p = s.save_json(tmp_path / "preset.json")
    assert FigureKitState.load_json(p) == s


def test_roundtrip_non_default(tmp_path):
    s = FigureKitState()
    s.calibration.pixel_size_y = 0.13
    s.calibration.square = False
    s.calibration.unit = "nm"
    s.crop.size = 128
    s.crop.size_mode = "pixels"
    s.crop.edge_mode = "pad"
    s.scale_bar.length = 500.0
    s.scale_bar.unit = "nm"
    s.scale_bar.custom_label = "½ µm"
    s.arrows.color = "#00ff00"
    s.export.upscale = 4
    s.export.fmt = "tiff"
    s.export.montage = True
    d = s.to_dict()
    assert FigureKitState.from_dict(json.loads(json.dumps(d))) == s
    p = s.save_json(tmp_path / "sub" / "p.json")
    assert FigureKitState.load_json(p) == s


def test_unknown_keys_ignored_and_bad_types_default():
    d = FigureKitState().to_dict()
    d["unknown_section"] = {"x": 1}
    d["crop"]["unknown_field"] = 3
    d["crop"]["size"] = "big"  # wrong type -> default
    d["export"]["upscale"] = 3.0  # float for int -> coerced
    s = FigureKitState.from_dict(d)
    assert s.crop.size == 50.0
    assert s.export.upscale == 3 and isinstance(s.export.upscale, int)
    assert FigureKitState.from_dict(None) == FigureKitState()
    assert FigureKitState.from_dict({"crop": "nonsense"}) == FigureKitState()


def test_state_dir_is_isolated_in_tests():
    assert ".napari-figurekit" not in str(last_state_path())
