"""Deprecated: use ``plotting/serve_animation.run_full_serve_animation`` instead."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_PLOTTING_DIR = Path(__file__).resolve().parents[2] / "plotting"
if str(_PLOTTING_DIR) not in sys.path:
    sys.path.insert(0, str(_PLOTTING_DIR))

from serve_animation import run_full_serve_animation  # noqa: E402

from src.markers.io import load_serve_markers  # noqa: E402


def run_ace_animation(ace_markers_path: str | Path, *, title: str | None = None, speed: int = 1) -> None:
    """Backward-compatible wrapper around the standard ACE animation renderer."""
    warnings.warn(
        "run_ace_animation is deprecated; use plotting/serve_animation.run_full_serve_animation",
        DeprecationWarning,
        stacklevel=2,
    )
    path = Path(ace_markers_path)
    markers = load_serve_markers(path)
    run_full_serve_animation(markers, title or path.parent.name, speed=speed)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="View ACE markers using the standard ACE animation renderer.")
    parser.add_argument("ace_markers", help="Path to ace_markers.npz or output directory")
    parser.add_argument("--speed", type=int, default=1)
    args = parser.parse_args()
    run_ace_animation(args.ace_markers, speed=args.speed)


if __name__ == "__main__":
    main()
