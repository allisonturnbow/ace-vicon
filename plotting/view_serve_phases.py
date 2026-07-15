"""
Interactive serve phase viewer with timeline, overlay, and phase selection.

Usage (from repo root or plotting/):
    python plotting/view_serve_phases.py firstserve
    python plotting/view_serve_phases.py generated_motionbert/andy
    python plotting/view_serve_phases.py generated_motionbert/andy/ace_markers.npz
    python plotting/view_serve_phases.py firstserve --speed 4

Select view via radio buttons: Full Serve or any phase.
Keys: + / = faster, - slower, R reset speed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from playback import ALLOWED_SPEEDS
from segmentation_viz import run_interactive_viewer
from skeleton_viz import load_serve_markers

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
INDIVIDUAL_DIR = os.path.join(PLOT_DIR, "markers", "individual")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive ACE serve phase 3D viewer")
    parser.add_argument(
        "serve",
        nargs="?",
        default="firstserve",
        help="Serve folder, ace_markers.npz, or generated_motionbert output directory",
    )
    parser.add_argument(
        "--speed",
        type=int,
        choices=ALLOWED_SPEEDS,
        default=None,
        help="Playback multiplier for all views (1, 2, 4, or 8)",
    )
    args = parser.parse_args()
    serve_arg = args.serve
    individual_dir = os.path.join(INDIVIDUAL_DIR, serve_arg)
    if os.path.isdir(individual_dir):
        serve_path = individual_dir
        serve_name = serve_arg
    else:
        serve_path = serve_arg
        serve_name = Path(serve_arg).parent.name if str(serve_arg).endswith(".npz") else Path(serve_arg).name

    try:
        markers = load_serve_markers(serve_path)
    except FileNotFoundError:
        print(f"Serve not found: {serve_arg}")
        sys.exit(1)

    run_interactive_viewer(serve_name=serve_name, markers=markers, cli_speed=args.speed)


if __name__ == "__main__":
    main()
