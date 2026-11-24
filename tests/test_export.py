import json

import numpy as np
import pytest
import tifffile
from PIL import Image

from napari_figurekit.core import export, pipeline, scalebar
from napari_figurekit.core.render import ChannelSpec


class Cmap:
    def __init__(self, rgb):
        self.rgb = np.asarray(rgb, float)

    def map(self, v):
        v = np.asarray(v, float)[:, None]
        return np.concatenate([v * self.rgb, np.ones_like(v)], axis=1)


def test_sanitize_and_pattern():
    assert export.sanitize_filename("cells 3d/µm: crop#1") == "cells_3d_um_crop_1"
    assert export.format_filename("{source}_{panel}", source="a b", panel="merge") == "a_b_merge"
    with pytest.raises(ValueError, match="Bad filename pattern"):
        export.format_filename("{nope}", source="a")


def test_unique_path_never_overwrites(tmp_path):
    p0 = export.unique_path(tmp_path, "x", ".png")
    assert p0.name == "x.png"
    p0.write_bytes(b"")
    p1 = export.unique_path(tmp_path, "x", ".png")
    assert p1.name == "x_1.png"
    export.sidecar_path(p1).write_text("{}")  # an orphan sidecar also blocks the name
    assert export.unique_path(tmp_path, "x", ".png").name == "x_2.png"
    assert export.unique_path(tmp_path, "y", ".png", {tmp_path / "y.png"}).name == "y_1.png"


def test_png_and_tiff_dpi(tmp_path):
    rgb = np.random.default_rng(0).integers(0, 255, (10, 12, 3), dtype=np.uint8)
    png = export.write_image(tmp_path / "a.png", rgb, "png", 600)
    with Image.open(png) as im:
        assert im.info["dpi"][0] == pytest.approx(600, abs=0.01)
        np.testing.assert_array_equal(np.asarray(im), rgb)
    tif = export.write_image(tmp_path / "a.tif", rgb, "tiff", 600)
    with tifffile.TiffFile(tif) as t:
        page = t.pages[0]
        np.testing.assert_array_equal(page.asarray(), rgb)
        num, den = page.tags["XResolution"].value
        assert num / den == pytest.approx(600)
        assert page.tags["ResolutionUnit"].value == 2  # inch


def _job(tmp_path, **kw):
    ch = [
        ChannelSpec(np.full((50, 60), 1.0), (0, 1), Cmap((0, 1, 0)), name="green"),
        ChannelSpec(np.full((50, 60), 1.0), (0, 1), Cmap((1, 0, 1)), name="mag"),
    ]
    bar = scalebar.compute_bar(10, "um", 0.5, "um", upscale=2)
    base = dict(
        channels=ch,
        upscale=2,
        folder=str(tmp_path),
        bar=bar,
        bar_label=bar.label,
        scale_bar={
            "apply_to": "merge",
            "color": "#ff0000",
            "margin": 5,
            "font_size": 10,
            "thickness": 4,
        },
        name_fields={"source": "s", "crop": "crop1"},
        sidecar={"pixel_size": {"y": 0.5, "x": 0.5, "unit": "um"}},
    )
    base.update(kw)
    return pipeline.ExportJob(**base)


def test_pipeline_panels_and_merge_only_bar(tmp_path):
    job = _job(tmp_path, colored=True, gray=True, montage=True, montage_gap=4, gap_color="black")
    imgs = pipeline.render_job(job)
    assert list(imgs) == ["merge", "green", "green_gray", "mag", "mag_gray", "montage"]
    assert imgs["merge"].shape == (100, 120, 3)
    red = lambda im: (im[..., 0] == 255) & (im[..., 1] == 0) & (im[..., 2] == 0)  # noqa: E731
    # Merge is white (green + magenta) with a red bar of exactly 20 * 2 = 40 px.
    assert red(imgs["merge"]).any(axis=1).sum() >= 4
    assert max(red(imgs["merge"]).sum(axis=1)) == 40
    assert not red(imgs["green"]).any()  # merge only
    assert imgs["montage"].shape == (100, 3 * 120 + 2 * 4, 3)
    assert np.all(imgs["montage"][:, 120:124] == 0)
    written = pipeline.write_job(job, imgs)
    assert [p.name for p, _ in written][0] == "s_crop1_merge.png"
    side = json.loads(written[0][1].read_text())
    assert side["scale_bar_drawn"] is True and side["panel"] == "merge"
    assert side["output_size_px"] == {"height": 100, "width": 120}
    side_g = json.loads(written[1][1].read_text())
    assert side_g["scale_bar_drawn"] is False
    side_m = json.loads(written[-1][1].read_text())
    assert side_m["scale_bar_drawn"] is True and side_m["montage"]["order"][-1] == "merge"


def test_pipeline_requires_folder(tmp_path):
    job = _job(tmp_path, folder="")
    with pytest.raises(ValueError, match="folder"):
        pipeline.write_job(job, pipeline.render_job(job))
