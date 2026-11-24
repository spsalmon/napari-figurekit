"""Length units (nm, µm, mm): parsing, conversion, formatting and "nice" lengths.

Units are stored internally as the ASCII codes ``"nm" | "um" | "mm"``. The micro sign
is only used for display (``"µm"``).
"""

from __future__ import annotations

import math

UNITS: tuple[str, ...] = ("nm", "um", "mm")

#: Size of one unit expressed in nanometres.
_NM_PER_UNIT: dict[str, float] = {"nm": 1.0, "um": 1_000.0, "mm": 1_000_000.0}

_DISPLAY: dict[str, str] = {"nm": "nm", "um": "µm", "mm": "mm"}

#: Unit names understood by pint (used for ``layer.units`` in napari >= 0.6).
_PINT: dict[str, str] = {"nm": "nanometer", "um": "micrometer", "mm": "millimeter"}

_ALIASES: dict[str, str] = {
    "nm": "nm",
    "nanometer": "nm",
    "nanometers": "nm",
    "nanometre": "nm",
    "nanometres": "nm",
    "um": "um",
    "µm": "um",  # micro sign
    "μm": "um",  # greek small letter mu
    "micron": "um",
    "microns": "um",
    "micrometer": "um",
    "micrometers": "um",
    "micrometre": "um",
    "micrometres": "um",
    "\\u00b5m": "um",  # literal escape as written by some ImageJ versions
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
}


def parse_unit(text: str) -> str:
    """Parse a unit string into one of ``"nm" | "um" | "mm"``.

    Parameters
    ----------
    text : str
        E.g. ``"µm"``, ``"um"``, ``"micron"``, ``"micrometer"``, ``"nm"``, ``"mm"``.
        Case and surrounding whitespace are ignored.

    Returns
    -------
    str
        The canonical unit code.

    Raises
    ------
    ValueError
        If the unit is not a supported length unit.
    """
    key = str(text).strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(f"Unsupported length unit: {text!r} (use nm, µm or mm)")


def try_parse_unit(text: object) -> str | None:
    """Like :func:`parse_unit` but return ``None`` instead of raising."""
    if text is None:
        return None
    try:
        return parse_unit(str(text))
    except ValueError:
        return None


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a length between units.

    Parameters
    ----------
    value : float
        Length expressed in ``from_unit``.
    from_unit, to_unit : str
        Any spelling accepted by :func:`parse_unit`.

    Returns
    -------
    float
        Length expressed in ``to_unit``.
    """
    f = _NM_PER_UNIT[parse_unit(from_unit)]
    t = _NM_PER_UNIT[parse_unit(to_unit)]
    return float(value) * f / t


def display_unit(unit: str) -> str:
    """Return the display form of a unit (``"um"`` → ``"µm"``)."""
    return _DISPLAY[parse_unit(unit)]


def pint_unit(unit: str) -> str:
    """Return the pint unit name of a unit (``"um"`` → ``"micrometer"``)."""
    return _PINT[parse_unit(unit)]


def format_number(value: float, max_decimals: int = 6) -> str:
    """Format a number without trailing zeros (``10.0`` → ``"10"``, ``0.25`` → ``"0.25"``)."""
    text = f"{float(value):.{max_decimals}f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def format_length(value: float, unit: str) -> str:
    """Format a length for a label, e.g. ``format_length(10, "um")`` → ``"10 µm"``."""
    return f"{format_number(value)} {display_unit(unit)}"


def round_half_up(x: float) -> int:
    """Round to the nearest integer, halves away from minus infinity (``2.5`` → ``3``).

    Python's built-in :func:`round` uses banker's rounding (``round(2.5) == 2``), which
    would make crop windows and bar lengths depend on the parity of the value. A tiny
    tolerance absorbs floating-point noise such as ``2.4999999999999996``.
    """
    return int(math.floor(float(x) + 0.5 + 1e-9))


def nice_length(target: float) -> float:
    """Return the "nice" length (1, 2 or 5 × 10^k) closest to ``target`` on a log scale.

    Parameters
    ----------
    target : float
        Desired length (any unit, must be > 0).

    Returns
    -------
    float
        E.g. ``13.3`` → ``10.0``, ``17`` → ``20.0``, ``0.04`` → ``0.05``.
    """
    if not math.isfinite(target) or target <= 0:
        raise ValueError("target length must be a positive number")
    k = math.floor(math.log10(target))
    best = None
    best_dist = math.inf
    for exp in (k - 1, k, k + 1):
        for mantissa in (1, 2, 5):
            cand = mantissa * 10.0**exp
            dist = abs(math.log10(cand / target))
            if dist < best_dist - 1e-12:
                best, best_dist = cand, dist
    assert best is not None
    # Remove floating noise such as 0.30000000000000004.
    return float(f"{best:.12g}")


def suggest_bar_length(image_width: float, fraction: float = 0.2) -> float:
    """Suggest a nice scale-bar length close to ``fraction`` of the image width.

    Parameters
    ----------
    image_width : float
        Physical width of the image, in the unit wanted for the bar.
    fraction : float
        Target fraction of the width (default 20 %).
    """
    return nice_length(image_width * fraction)
