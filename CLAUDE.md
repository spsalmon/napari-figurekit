# CLAUDE.md — napari-figurekit

A napari plugin for turning multichannel microscopy images into publication-ready figure panels:
define channels, calibrate pixel size, crop squares around points, add an exact scale bar and arrows,
and export pixel-accurate PNG/TIFF panels.

**Audience:** internal tool for a small lab group (a few PhD students/postdocs). Not published to PyPI.
Install via `pip install -e .` or `pip install git+<repo-url>`.

**Goal for Claude Code:** implement the whole plugin in one pass, following this file. Do not stop to
ask questions. Where something is ambiguous, pick the simplest reasonable option, implement it,
and record the decision in `README.md` under "Design decisions".

---

## 1. Tech stack and constraints

- Python ≥ 3.10 (no upper bound; tested on 3.14). Target napari **0.7.x–0.10.x** but keep working on **≥ 0.6** (see §8 "Version adapter").
- Plugin engine: **npe2** (`napari.yaml` manifest). No npe1 hooks.
- UI: `qtpy` widgets (QWidget + QTabWidget + QFormLayout). `magicgui` is allowed for small sub-forms,
  but the main dock widget is a single `QWidget` subclass.
- Runtime deps: `napari`, `numpy`, `qtpy`, `magicgui`, `tifffile`, `pillow>=10.1`.
- Dev deps (optional extra `[dev]`): `pytest`, `pytest-qt`, `napari[pyqt6]` (or pyqt5), `scikit-image` (sample data).
- Build: `setuptools` via `pyproject.toml`, src layout. `napari.yaml` must be included as package data.
- **Use only public napari APIs.** Do not import from `napari._qt`, `napari._vispy`, or call `layer._*`.
  If something seems to require a private API, find another way or drop the sub-feature and note it in the README.
- **Separate logic from UI.** All maths and image operations go in `core/` as pure functions with no Qt
  and no viewer, so they can be unit-tested headless. UI modules only read widgets, call core, and update layers.

## 2. Repository layout

```
napari-figurekit/
├── pyproject.toml
├── README.md                 # user guide for colleagues (GUI walkthrough + design decisions)
├── CLAUDE.md
├── src/napari_figurekit/
│   ├── __init__.py
│   ├── napari.yaml
│   ├── _widget.py            # FigureKitWidget: dock widget with tabs, shared state
│   ├── _sample_data.py       # cells3d sample with correct scale
│   ├── state.py              # dataclasses for all settings + JSON (de)serialisation
│   ├── core/
│   │   ├── units.py          # unit parsing/conversion (nm, µm, mm), "nice" lengths
│   │   ├── calibration.py    # pixel size helpers, metadata reading (tifffile)
│   │   ├── channels.py       # split along channel axis, default colormaps
│   │   ├── crop.py           # crop-window maths + array cropping
│   │   ├── render.py         # deterministic RGB compositing (additive), upscale
│   │   ├── scalebar.py       # scale bar geometry + drawing onto RGB arrays
│   │   ├── annotations.py    # arrows: line → vector conversion, drawing onto RGB
│   │   └── napari_compat.py  # version adapter for viewer.scale_bar, layer.units, etc.
│   └── ui/
│       ├── setup_tab.py
│       ├── crop_tab.py
│       ├── scalebar_tab.py
│       ├── annotate_tab.py
│       └── export_tab.py
└── tests/
    ├── test_units.py
    ├── test_crop.py
    ├── test_render.py
    ├── test_scalebar.py
    ├── test_annotations.py
    ├── test_state.py
    └── test_widget.py        # uses make_napari_viewer fixture
```

### napari.yaml (skeleton)

```yaml
name: napari-figurekit
display_name: Figure Kit
contributions:
  commands:
    - id: napari-figurekit.widget
      python_name: napari_figurekit._widget:FigureKitWidget
      title: Open Figure Kit
    - id: napari-figurekit.sample_cells3d
      python_name: napari_figurekit._sample_data:cells3d_sample
      title: Load calibrated cells3d sample
  widgets:
    - command: napari-figurekit.widget
      display_name: Figure Kit
  sample_data:
    - command: napari-figurekit.sample_cells3d
      key: cells3d
      display_name: cells3d (calibrated, 2 channels)
```

Entry point in `pyproject.toml`:

```toml
[project.entry-points."napari.manifest"]
napari-figurekit = "napari_figurekit:napari.yaml"

[tool.setuptools.package-data]
"*" = ["*.yaml"]
```

`FigureKitWidget.__init__(self, napari_viewer: "napari.viewer.Viewer")`. npe2 injects the viewer by that parameter name.

## 3. Core principles (publication integrity)

These are non-negotiable. Tests must enforce them.

1. **Crops never change pixel values.** A crop is a pure slice of the original array: same dtype, no
   resampling, no normalisation. Dask/lazy arrays stay lazy until needed.
2. **Export never interpolates.** Upscaling is integer nearest-neighbour only (`np.repeat`).
3. **The scale bar length is exact in output pixels:** `bar_px = round(length_in_px_unit / pixel_size) * upscale`.
   The label always shows the length the user typed (e.g. "10 µm"), never a rounded display value.
   If `length / pixel_size` is not an integer, round it and show a non-blocking warning with the actual
   represented length (e.g. "bar = 38 px = 9.88 µm").
4. **Contrast mapping is linear** unless gamma ≠ 1. Gamma ≠ 1 shows a warning and is recorded.
5. **Every export writes a JSON sidecar** (`<file>.json`) containing source layer names, file paths if
   known, pixel size + unit, crop origin/size in source pixels, per-channel colormap / contrast limits /
   gamma, upscale factor, scale bar settings, plugin + napari versions, and a timestamp. This is what makes
   a panel reproducible and defensible to reviewers.

## 4. Shared state

`state.py` defines dataclasses (`CalibrationSettings`, `CropSettings`, `ScaleBarSettings`,
`ArrowSettings`, `ExportSettings`, and `FigureKitState` holding all of them) with `to_dict` / `from_dict`
and `save_json(path)` / `load_json(path)`.

- The widget holds one `FigureKitState` instance shared by all tabs.
- A bar at the bottom of the dock widget (below the tabs) has **Save preset…** and **Load preset…**
  buttons for the whole state.
- The last-used state is auto-saved to `~/.napari-figurekit/last_state.json` when the widget closes,
  and restored on open (wrapped in try/except; a corrupt file is ignored with a warning).

Per-layer info lives in `layer.metadata["figurekit"]` (a dict): `pixel_size`, `unit`, `channel_name`,
and for crops: `source_layer`, `crop_origin_px` (y, x), `crop_size_px` (ny, nx), `center_world`.

## 5. Features (tabs)

Priority tags: **[MUST]** is required for done. **[SHOULD]** gets implemented in the same pass after all
MUSTs work and tests pass. **[NICE]** is only done if trivial; otherwise list it under "Future ideas" in the README.

All layer combo boxes must stay in sync with the viewer: connect to `viewer.layers.events.inserted`,
`.removed`, `.moved`, and each layer's `events.name`.

### Tab 1 — Setup (channels + calibration)

**Channel axis [MUST]**
- Source image layer combo. Below it, a label showing the shape with axis indices, e.g.
  `axis 0: 60 | axis 1: 2 | axis 2: 256 | axis 3: 256`.
- "Channel axis" spin box (`-1` = none) with a sensible default: the smallest axis with size ≤ 6 that is
  not one of the last two axes, else none.
- **Split channels** button: removes the source layer and calls
  `viewer.add_image(data, channel_axis=k, scale=..., translate=..., name=[...], colormap=[...], blending="additive")`.
  Keep the source's scale/translate, dropping the channel axis entry. Preserve the original `metadata`
  plus a `figurekit` entry. If the user unchecks "Remove original", keep the source layer and hide it.
- Before splitting, a small table (QTableWidget) lists each channel: editable **name**
  (default `"<layer> ch<i>"`) and **colormap** combo. Defaults by channel count:
  1 → gray; 2 → green, magenta; 3 → green, magenta, cyan; 4 → + yellow; >4 → cycle.
- RGB layers (`layer.rgb is True`): splitting is disabled, with a tooltip explaining why.

**Calibration [MUST]**
- Pixel size Y and X (QDoubleSpinBox, 6 decimals, min 1e-6), with a "square pixels" checkbox (default on;
  when on, X mirrors Y). Optional Z step for 3D data (shown only when the layer has ≥ 3 spatial dims).
- Unit combo: `nm`, `µm`, `mm` (store internally as `"nm" | "um" | "mm"`; display "µm").
- "Apply to" combo: *selected layer* / *all image layers* (default) / *layers with the same shape*.
- **Apply** sets `layer.scale` (last 2 or 3 axes; other axes keep their scale) and
  `layer.metadata["figurekit"]["pixel_size"/"unit"]`, tries to set `layer.units` via `napari_compat`,
  and sets `viewer.scale_bar.unit`.
- **Read from file [SHOULD]**: if `layer.source.path` is a TIFF, read with `tifffile`:
  - ImageJ TIFF: `XResolution`/`YResolution` are pixels per unit, so pixel size = 1 / value.
    The unit comes from `imagej_metadata["unit"]` ("micron", "µm", "um" → um).
  - OME-TIFF: `PhysicalSizeX/Y/Z` + `PhysicalSizeXUnit`.
  - Fill the spin boxes and let the user press Apply. If nothing is found, show a message and leave the values unchanged.
- Also pre-fill the spin boxes from the current `layer.scale` whenever the selected layer changes.

**Display helpers [SHOULD]**
- "Set all image layers to additive blending" button.
- **Sync contrast**: pick a reference layer and copy its `contrast_limits` and `gamma` to all layers whose
  `metadata["figurekit"]["channel_name"]` matches (or whose name contains a user-given substring).
  Used to compare conditions with identical display settings.
- Show current contrast limits of the selected layer as two editable spin boxes (exact values typed, not dragged).

### Tab 2 — Crop around point [MUST]

- **Crop size** spin box + mode combo: *physical units* (uses the calibration unit) or *pixels*.
  Default 50 µm.
- **Center points layer**: button "Create / select center points layer" creates (or reuses) a Points
  layer named `crop centers`, makes it active, and sets mode to `add`. Each point = one crop.
- **Live preview [MUST]**: a Shapes layer `crop preview` (rectangles, no fill, edge color yellow,
  edge width 2 screen px if supported, else a small world value) shows the square around every point.
  Update on `points.events.data`, crop-size change, and source change. Preview rectangles use the
  exact pixel-snapped window that will be cropped, not the idealised square.
- **Layers to crop**: a checklist of image layers (default: all visible image layers with the same Y/X
  shape and scale as the first checked one). Layers that are not compatible are shown disabled with a tooltip.
- **Crop** button: for every point × every checked layer, create a new image layer:
  - Window maths (in `core/crop.py`, fully unit-tested):
    - `n_y = round(size / pixel_size_y)`, `n_x = round(size / pixel_size_x)` (so it is physically square
      even with anisotropic pixels); in pixel mode `n_y = n_x = size`.
    - The center in data pixel coords comes from `layer.world_to_data(point)`, using the last two axes.
    - `start = int(round(c - n / 2))`, `stop = start + n`. Document this rule in the docstring.
  - **Edge handling** combo: *shift inside image* (default; keeps size n, moves the window), *clip*
    (smaller crop), *pad with zeros* (keeps size and centre; padded area recorded in metadata).
    If n > image size, fall back to clip and warn.
  - Crop only the last two axes; all other axes (Z, T) stay whole. Multiscale layers use `data[0]`
    and warn once.
  - New layer: name `"<source name> crop<i>"` (i = point index + 1). It copies colormap,
    contrast_limits, gamma, blending, opacity, scale, and `metadata`, and gets
    `translate = source.translate + start * scale` on the cropped axes, so it overlays exactly on the source.
    Metadata gets `figurekit` crop info (§4).
  - Options: "Hide source layers after crop" (default on), "Add outline of crop to source" (default on:
    keeps a Shapes layer `crop outlines` with the rectangles, for inset boxes in the overview image),
    "Zoom to first crop" (default on: `viewer.reset_view()` restricted to the crop by temporarily
    hiding others, or `viewer.camera` center/zoom).
- **Crop current view [NICE]**: a crop window = the visible canvas extent.

### Tab 3 — Scale bar [MUST]

Controls (all stored in `ScaleBarSettings`):
- Visible (checkbox).
- **Length** + unit (unit defaults to the calibration unit; may differ, e.g. pixel size in µm and bar in nm,
  so convert with `core/units.py`). Button **Suggest** sets a "nice" length (1, 2, 5 × 10^k) close to 20 %
  of the reference layer's width.
- Label: show text (checkbox), custom label override (line edit; empty = auto, e.g. "10 µm").
  Use "µm" with the micro sign in labels.
- Font size (px in the exported image) with a read-only helper showing `= X pt at <dpi> dpi`
  (pt = px × 72 / dpi, dpi from the Export tab).
- Bar thickness (px in the exported image), color (white / black / custom via QColorDialog),
  position (bottom-right [default], bottom-left, top-right, top-left), margin (px), optional background box + box color.
- A label below shows the computed exact bar: `"10 µm = 38.46 px → 38 px (9.88 µm)"`, in red when rounding error > 0.5 %.

Behaviour:
- **Live napari overlay** mirrors these settings through `napari_compat.apply_scale_bar(viewer, settings)`
  for preview: visible, unit, fixed length (converted to the layers' world unit), font size, color, box,
  position, ticks off. Set only attributes that exist in the installed napari version.
- The **exported** scale bar is drawn by `core/scalebar.py` directly into the output pixels (§Tab 5), so
  it is exact regardless of napari's overlay behaviour. The README must say clearly: *the canvas overlay
  is a preview; the exported file is the source of truth.*
- "Scale bar on": *merge only* / *all panels* (default) / *none*. This applies to export.

### Tab 4 — Annotate (arrows) [SHOULD]

- **New arrow layer** creates a Shapes layer `arrows (draw)` in `add_line` mode. The user draws lines from tail to tip.
- A Vectors layer `arrows` (`vector_style="arrow"`) is kept in sync live: on Shapes `events.data`,
  convert each line to `[start, end - start]` (`core/annotations.py`), using the Shapes layer's scale/translate.
  The draw layer stays visible but with opacity 0.3 so it can still be edited.
- Controls: arrow color, width (px in export), head length (× width, default 3), head width (× width, default 2.5).
- Export draws arrows with PIL as a line plus a filled triangular head whose tip lands exactly at the end
  point (mapped world → output pixels through the reference layer's `world_to_data`, then × upscale).
  The exported look may differ slightly from the napari preview; document this.
- **Text labels [NICE]**: Points layer with text (e.g. "*", "a") burned into export with the same font.

### Tab 5 — Export [MUST]

- **Layers**: checklist of image layers to composite (default: visible image layers, in layer-list order).
  They must share Y/X shape, scale, and translate. Otherwise show a clear error naming the mismatched
  layer (typical case: tick the crops of one point together).
- **Slice**: for nD layers, export the 2D slice currently shown in the viewer. For each non-displayed
  axis, the index = `round(layer.world_to_data(viewer.dims.point)[axis])`, clipped to bounds.
  Optional **max projection along Z [SHOULD]** checkbox.
- **Outputs** (checkboxes): Merge (additive composite) [default on], each channel coloured, each channel
  grayscale, **montage [SHOULD]** (channels + merge in one row with a gap of N px, gap color white/black).
- **Upscale factor**: integer 1–10 (nearest neighbour). Shows the output size in px and, with the DPI,
  the printed width in mm.
- DPI (default 300), written into PNG/TIFF metadata.
- Format: PNG (8-bit RGB, default) or TIFF (8-bit RGB via tifffile, with resolution tags).
- Output folder (QFileDialog) + filename pattern, default `{source}_{crop}_{panel}` where panel ∈
  `merge`, `<channel name>`, `<channel name>_gray`, `montage`. Sanitise filenames. Never overwrite
  silently: append `_1`, `_2`, …
- **Preview** button: adds the merged render (with scale bar and arrows) as an RGB layer `figurekit preview`
  (replacing any previous one), so the user can check before saving.
- **Export** button writes all files + JSON sidecars (§3.5) and shows a summary notification
  (`napari.utils.notifications.show_info`) listing the files.
- **napari-native export [SHOULD]**: secondary button "Export canvas (napari export_figure)" calling
  `viewer.export_figure(path, scale_factor=...)`, for quick looks that include any layer type.

#### Rendering algorithm (`core/render.py`, pure numpy)

For each channel layer, take the 2D slice as float64 → `v = clip((x - lo) / (hi - lo), 0, 1)` →
`v = v ** gamma` → `rgba = layer.colormap.map(v.ravel()).reshape(..., 4)` (pass the napari Colormap
object, or a function, so core stays viewer-free) → `rgb = rgba[..., :3] * opacity`.
Merge = `clip(sum(rgb_i), 0, 1)` (additive, as napari's additive blending). Convert to uint8 with
`np.round(x * 255)`. Then upscale with `np.repeat` on both axes. Then draw arrows, then the scale bar
(PIL `ImageDraw`), so annotations are crisp at output resolution. Single-channel grayscale panels
ignore the colormap.

Font: try `ImageFont.truetype("DejaVuSans.ttf")`, then `"Arial.ttf"`/`"arial.ttf"`, then matplotlib's
bundled DejaVuSans path if matplotlib is importable, then `ImageFont.load_default(size=...)`. Allow an
optional custom TTF path in ScaleBarSettings. Scale bar label: centred above the bar, gap = 0.3 × font size.

## 6. UX details

- Dock area: right. Minimum width ~ 340 px. Tabs: Setup · Crop · Scale bar · Annotate · Export.
- Every button that can fail shows a `show_warning`/`show_error` notification with a clear message.
  Exceptions are never swallowed silently, and the GUI never crashes.
- Tooltips on every control (one sentence).
- Long operations (export with many files) run in `napari.qt.threading.thread_worker` with the button
  disabled meanwhile. Crops of normal-size data can stay synchronous.
- No modal dialogs except file pickers and the color dialog.

## 7. Sample data

`cells3d_sample()` returns `skimage.data.cells3d()` (shape (60, 2, 256, 256), ZCYX) as two layers via
`channel_axis=1`, with scale `(0.29, 0.26, 0.26)` µm, names "membrane"/"nuclei", green/magenta, additive.
Import skimage lazily; if missing, raise a clear error saying `pip install scikit-image`.
Use this sample for manual checks and the widget tests.

## 8. napari version adapter (`core/napari_compat.py`)

napari's overlay and units APIs changed between 0.5 → 0.7. Isolate all version-sensitive calls here:

- `apply_scale_bar(viewer, settings, world_unit)`: for each field (`visible`, `unit`, `length`,
  `font_size`, `color`, `colored`, `box`, `box_color`, `position`, `ticks`), set it only if
  `hasattr(viewer.scale_bar, name)`, in try/except. Convert position names to the installed
  enum's values (inspect them at runtime; e.g. `"bottom_right"`).
- `set_layer_units(layer, unit, spatial_axes)`: only if `hasattr(layer, "units")`. Build a full-length
  tuple, keeping the existing units for non-spatial axes. Try/except; on failure, log and continue.
  Scale is the source of truth.
- `napari_version()` for the sidecar.

**Before writing UI code, run** `python -c "import napari; print(napari.__version__); v=napari.components.ViewerModel(); print(v.scale_bar)"`
to see the real scale bar fields of the installed version, and adapt.

## 9. Tests (must pass: `pytest -q`)

Headless: set `QT_QPA_PLATFORM=offscreen` in `tests/conftest.py`. Core tests must not need Qt.

Required test cases (add more as needed):
- `units`: 500 nm == 0.5 µm; nice-length suggestions (e.g. width 66.56 µm → 10 µm or 20 µm);
  "µm"/"um"/"micron" parse to the same unit.
- `crop`:
  - size 50 µm at 0.5 µm/px → 100 × 100 window; with pixel sizes (0.5, 0.25) → 100 × 200.
  - centring rule for odd/even n (explicit expected start indices).
  - shift-inside / clip / pad behaviour at all four edges and corners.
  - the crop result equals `arr[..., y0:y1, x0:x1]` exactly (same dtype, `np.shares_memory` or equal values).
  - nD (Z, Y, X) keeps Z intact. Dask input stays a dask array.
  - translate of the new layer = source translate + start × scale.
- `render`:
  - green (0,1,0) + magenta (1,0,1) at full intensity → white (255,255,255).
  - contrast limits map lo→0 and hi→255 exactly. Gamma 1 is linear.
  - upscale ×3 is exact nearest-neighbour (each source pixel becomes a 3×3 block of identical values).
- `scalebar`:
  - 10 µm at 0.1 µm/px, upscale 2 → exactly 200 bar pixels counted in the rendered image (count
    the bar-colored pixels in one row of the bar).
  - non-integer case reports the represented length correctly.
  - the bar stays inside the image with the margin, for all four positions.
- `annotations`: line [(10,10),(10,50)] → vector [[10,10],[0,40]]; the drawn arrow tip pixel is at the expected location.
- `state`: round-trip `to_dict`/`from_dict`/JSON for default and non-default states; unknown keys ignored.
- `widget` (with `make_napari_viewer`): the widget instantiates; split channels on a (2, 64, 64) image
  gives 2 layers with additive blending; apply calibration sets scale; add a point + crop gives the
  expected number of new layers with the right shape; export to `tmp_path` writes PNG + JSON with the expected size.

## 10. Implementation order for Claude Code

1. Scaffold the package, `pyproject.toml`, `napari.yaml`, empty widget. Run `pip install -e ".[dev]"`
   and check that `npe2 validate napari-figurekit` (or `python -c "import napari_figurekit"`) works.
2. Inspect the napari version + scale bar fields (§8).
3. Implement `core/` modules with their tests. Run `pytest` until green.
4. Implement `state.py` + tests.
5. Implement the UI tabs in order Setup → Crop → Scale bar → Export → Annotate, plus the widget tests.
6. Run the full test suite. Then run a smoke script (`scripts/smoke.py`, not packaged) that opens a viewer
   with the sample data, adds the widget, programmatically splits/calibrates/crops/exports to a temp dir,
   and prints the output files. It must run without errors under `QT_QPA_PLATFORM=offscreen`.
7. Write `README.md`: install, a 1-page GUI walkthrough per tab, the integrity principles (§3),
   "Design decisions", "Known limitations", "Future ideas".

## 11. Definition of done

- [ ] `pip install -e .` works; "Figure Kit" appears under Plugins in napari.
- [ ] All [MUST] features work from the GUI with no console needed.
- [ ] `pytest -q` passes headless; the smoke script runs clean.
- [ ] Exported scale bars are pixel-exact (test-verified), and every export has a JSON sidecar.
- [ ] No private napari API imports (`grep -r "napari\._" src/` returns nothing).
- [ ] README written for non-programmer colleagues.

## 12. Code style

- Type hints everywhere; docstrings (NumPy style) on public core functions, stating units and coordinate conventions
  (data px vs world; axis order is always napari's, i.e. …, Y, X).
- `ruff` clean (line length 100). No print statements in the package; use `logging` + napari notifications.
- Keep functions small; no global mutable state except the widget's `FigureKitState`.
