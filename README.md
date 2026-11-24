# napari-figurekit

A napari plugin that turns multichannel microscopy images into **publication-ready figure
panels**: split channels, set the pixel size, crop squares around points of interest, add an
exact scale bar and arrows, and export pixel-accurate PNG/TIFF panels — each with a JSON
"receipt" (sidecar) that records exactly how it was made.

Internal tool for the lab; not on PyPI.

---

## Install

You need a Python environment with Python ≥ 3.10 (tested up to 3.14) and napari (≥ 0.6, tested with 0.7.1 and 0.10.0). In that environment:

```bash
pip install git+<repo-url>          # or, from a local copy:  pip install -e .
```

Then start napari and open **Plugins → Figure Kit**. A dock widget with five tabs appears on
the right: **Setup · Crop · Scale bar · Annotate · Export**.

To try it on example data: **File → Open Sample → Figure Kit → cells3d (calibrated, 2
channels)** (needs `pip install scikit-image`; the image is downloaded once).

For development: `pip install -e ".[dev]"`, then `pytest -q` and
`QT_QPA_PLATFORM=offscreen python scripts/smoke.py`.

---

## Walkthrough

### 1. Setup — channels and calibration

**Split channels** (when your file opened as *one* layer with all channels inside):

1. Pick the layer in **Source layer**. The *Shape* line shows the size of every axis, e.g.
   `axis 0: 60 | axis 1: 2 | axis 2: 256 | axis 3: 256` (here axis 1 holds 2 channels).
2. Check the **Channel axis** (it is guessed for you; `-1` means "no channel axis").
3. In the table, give each channel a name (e.g. *DAPI*, *actin*) and a colormap.
   Defaults: 1 channel → gray; 2 → green, magenta; 3 → + cyan; 4 → + yellow.
4. Click **Split channels**. You get one layer per channel, blended additively (like the
   final merge). Untick *Remove original* to keep the original (hidden) instead.

**Calibration** (the pixel size — everything physical depends on it):

1. Type the **Pixel size** (Y; X follows while *square pixels* is ticked) and the **Unit**
   (nm, µm, mm). For z-stacks you can also tick **Z step**.
   *Read from file* fills these in from ImageJ or OME TIFF metadata when the file has them.
2. Choose **Apply to** (*all image layers* by default) and click **Apply**.

The boxes always show the pixel size of the layer selected above, so you can check it.

**Display helpers**

- *Additive blending for all layers* — the export always merges additively; this makes the
  screen match.
- *Contrast* — type exact contrast limits for the source layer (no slider guesswork).
- *Sync contrast* — copy the contrast limits and gamma of one layer to all layers of the
  same channel (or whose name contains some text). Use it to show different conditions with
  identical display settings.

### 2. Crop — squares around points

1. Set the **Crop size** (default 50 µm; or switch to *pixels*) and what happens **at the image
   edge**: *shift inside image* (keeps the size), *clip* (smaller crop) or *pad with zeros*.
2. Click **Create / select center points layer** and click in the image: each point is the
   centre of one crop. Yellow rectangles (layer *crop preview*) show **exactly** the pixels
   that will be cropped.
3. Tick the **Layers to crop** (typically all channels). Layers whose size/pixel size does not
   match are greyed out.
4. Click **Crop**. For every point and every ticked layer a new layer
   `"<layer> crop<i>"` appears, perfectly overlaid on the original. By default the originals
   are hidden, the crop rectangles are kept in a *crop outlines* layer (for inset boxes in an
   overview figure), and the view zooms to the first crop.

Crops are **pure slices** of your data: same pixel values, same data type, nothing resampled.
Z and time axes are kept whole.

### 3. Scale bar

- **Length** + unit (the bar's unit may differ from the pixel unit, e.g. bar in nm).
  **Suggest** picks a round length (1, 2 or 5 × 10ⁿ) about 20 % of the image width.
- **Exact bar** shows how many pixels the bar really is, e.g.
  `10 µm = 38.46 px → 38 px (9.88 µm)`. It turns **red** when rounding to whole pixels changes
  the length by more than 0.5 % — then choose another length (or accept it; the warning is
  stored in the sidecar).
- Label text (auto = the length you typed, e.g. "10 µm"), font size (in exported pixels, with
  its size in points at the export DPI), thickness, color, corner, margin, optional background
  box, and on which panels the bar is drawn (*all panels*, *merge only*, *none*).

> **The canvas overlay is a preview; the exported file is the source of truth.**
> The scale bar you see in napari is napari's own overlay and may look slightly different.
> The exported bar is drawn directly into the pixels and is exact.

### 4. Annotate — arrows

1. Click **New arrow layer**, then draw lines **from the tail to the tip** (layer
   *arrows (draw)*). Arrows appear live in the *arrows* layer.
2. Choose color, width (exported pixels), head length/width (multiples of the width) and on
   which panels arrows are drawn.

Exported arrows are drawn crisply at output resolution; the tip lands exactly at the end of
your line. They may look slightly different from the napari preview.

### 5. Export

1. **Layers to composite** — after a crop, the crops of point 1 are ticked for you. Use
   *select crop…* to switch to another point's crops, or *Select visible layers*.
   All ticked layers must have the same size and position (a clear error tells you which one
   does not).
2. For z-stacks, the **slice currently shown** in napari is exported — or tick *Max projection
   along Z*.
3. **Outputs**: merge (additive composite), each channel colored, each channel grayscale,
   montage (colored channels + merge in a row with a white or black gap).
4. **Upscale** (1–10×, by repeating pixels — never interpolated) and **DPI** (written into the
   file). *Output size* shows pixels and printed millimetres.
5. **Format** PNG or TIFF (8-bit RGB), **Folder**, and **File name** pattern
   (`{source}_{crop}_{panel}` by default; also `{date}`). Existing files are never
   overwritten — `_1`, `_2`, … is appended.
6. **Preview** adds the merged result (with scale bar and arrows) as the layer
   *figurekit preview* so you can check it. **Export** writes everything.

*Export canvas (napari export_figure)* is a quick screenshot of whatever napari shows, for
quick looks — not pixel-exact.

### Presets

**Save preset… / Load preset…** (bottom of the dock) store all settings in a JSON file, so a
whole figure series can use identical settings. Your last settings are also restored
automatically when you reopen Figure Kit (`~/.napari-figurekit/last_state.json`).

---

## Integrity principles

These are enforced by the code and by tests:

1. **Crops never change pixel values.** A crop is a slice of the original array (same dtype,
   no resampling, no normalisation). Lazy (dask) data stays lazy.
2. **Export never interpolates.** Upscaling is integer nearest-neighbour only.
3. **The scale bar is exact in output pixels:**
   `bar_px = round(length / pixel_size) × upscale`. The label always shows the length you
   typed; if rounding was needed, a warning reports the length actually represented.
4. **Contrast mapping is linear** unless gamma ≠ 1, which triggers a warning and is recorded.
5. **Every exported image has a JSON sidecar** (`<image file>.json`, e.g.
   `cells3d_crop1_merge.png.json`) with: source layer names and file paths, pixel size and
   unit, crop origin/size in source pixels, slice indices, per-channel colormap / contrast
   limits / gamma / opacity, upscale, DPI, scale-bar settings and exact bar length, arrows,
   plugin and napari versions, a timestamp and any warnings. Keep it with the image: it makes
   the panel reproducible and defensible to reviewers.

---

## Design decisions

Choices made where the specification was open or where following it literally would have
been wrong:

- **Crop centring rule.** The window start is `start = round_half_up(c − (n − 1) / 2)` where
  `c` is the point in data-pixel coordinates and `n` the window size. In napari the *centre*
  of pixel `i` is at coordinate `i`, so this picks the window whose centre is nearest the
  click (for odd `n`, the middle pixel is the pixel you clicked). The textbook
  `round(c − n/2)` assumes pixel *corners* at integers and would be biased by half a pixel.
- **Rounding** of crop sizes, window starts and bar lengths is "round half up". Python's
  `round()` (banker's rounding: `round(2.5) == 2`) is never used for geometry.
- **Edge handling.** The "window larger than the image → fall back to clip (with a warning)"
  rule applies to *shift* mode; *pad* mode keeps size and centre and pads as needed, and
  *clip* is unaffected. Points outside the image are skipped with a warning.
- **Z step is optional** (checkbox), because the third-to-last axis may be time. It is
  offered for layers with ≥ 3 dimensions.
- **Re-calibrating** also rescales each layer's translate on the calibrated axes, so crops
  stay aligned with their source (their offset in pixels is preserved).
- **Units in napari 0.7.** `viewer.scale_bar.unit` is deprecated in 0.7 (units come from
  `layer.units`). Figure Kit sets `layer.units` on calibration — on images *and* on its helper
  layers (points, shapes, vectors), because napari ignores units when layers disagree. All of
  this goes through `core/napari_compat.py`; only real fields of the installed version are set.
  Since napari 0.9 the overlay lives at `viewer.canvas.overlays["scale_bar"]` (`viewer.scale_bar`
  is a deprecated alias); the adapter uses the new location when it exists.
- **No upper Python bound.** `requires-python = ">=3.10"`: capping it only blocks installs on new
  Python releases (e.g. 3.14) even when everything works.
- **Reference layer.** The scale bar (and arrow mapping) uses the first ticked layer of the
  Export tab (X pixel size); if none is ticked, the first visible image layer.
- **Sidecar name** is `<image file name>.json` (`panel.png.json`), so it is unambiguous even
  if PNG and TIFF of the same panel exist. Name collisions are checked on both files.
- **File-name fields:** `{source}` = the original image name (before splitting/cropping),
  `{crop}` = `crop<i>` or `full`, `{panel}` = `merge`, `<channel>`, `<channel>_gray`,
  `montage`, `{date}` = YYYYMMDD. Names are sanitised (`µ` → `u`, unsafe characters → `_`).
- **Montage** = coloured channel panels + merge. The scale bar and arrows are drawn on each
  sub-panel according to their "on" setting (*merge only* → only on the merge sub-panel).
- **Arrows "on" setting** (all / merge only / none) was added, mirroring the scale bar.
- **Grayscale panels** ignore the colormap and opacity (raw intensity mapping with the same
  contrast limits and gamma).
- **After a crop**, Export ticks the crops of point 1; a *select crop…* menu ticks another
  point's crops. This avoids the most common mismatch error.
- **Preview/outline rectangles** have a width of ~1.5 source pixels in world units: napari
  Shapes have no screen-space edge width.
- **Zoom to first crop** sets the camera directly (napari's `reset_view` ignores layer
  visibility, so "hide others and reset" does not work).
- **Scale bar fit** is checked before exporting: if bar + label + margin do not fit in the
  panel, you get an error immediately instead of a failed export.
- **Extra modules** beyond the planned layout: `core/export.py` (file names, PNG/TIFF + DPI,
  sidecars), `core/pipeline.py` (pure render-and-write job, run in a worker thread) and
  `ui/_common.py` (shared widgets).
- The auto-saved settings folder can be redirected with the environment variable
  `NAPARI_FIGUREKIT_HOME` (used by the tests so they never touch your real settings).

## Known limitations

- Rotated, sheared or affine-transformed layers cannot be cropped or exported (scale and
  translate only). RGB layers cannot be split, cropped or composited.
- Export is 2D: the current slice (or a max projection along Z) of the last two axes, viewed
  in 2D. For time series, the current time point is exported.
- Multiscale (pyramid) images are cropped/exported from the full-resolution level.
- Points, crop rectangles and arrows are 2D and apply to every Z slice. If you re-calibrate
  *after* placing points or arrows, they do not move with the image — calibrate first.
- The napari canvas scale bar and arrows are previews; only the exported pixels are exact.
- Very large lazy (dask) images are sliced in the GUI thread before the export worker starts.
- Tested with napari 0.7.1 (Python 3.12) and 0.10.0 (Python 3.14); 0.6 should work through the version adapter but is untested.

## Future ideas

- *Crop current view* (crop the visible canvas extent).
- Text labels (e.g. "*", "a") burned into the export with the same font.
- Export all crop groups in one click.
- Burn the crop outlines (inset boxes) into an exported overview panel.
- Panel letters (A, B, C) and channel name labels on panels.
- Raw 16-bit TIFF export of the crops (for quantification) next to the RGB panels.
