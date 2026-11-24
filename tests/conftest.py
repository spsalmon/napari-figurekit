"""Test configuration: headless Qt and an isolated state folder."""

import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Never touch the user's real ~/.napari-figurekit during tests (set before any import).
os.environ["NAPARI_FIGUREKIT_HOME"] = tempfile.mkdtemp(prefix="figurekit-test-home-")


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_state_home(tmp_path, monkeypatch):
    """Each test gets its own empty state folder (no auto-restored settings)."""
    home = tmp_path / "figurekit-home"
    home.mkdir()
    monkeypatch.setenv("NAPARI_FIGUREKIT_HOME", str(home))
    yield home
