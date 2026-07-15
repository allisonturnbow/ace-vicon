"""
Full-serve 3D skeleton animation (original ACE behavior).

Usage:
    python plotting/plot.py firstserve
    python plotting/plot.py firstserve --speed 4

Keys: + / = faster, - slower, R reset speed.

For phase segmentation, timeline, and per-phase views use:
    python plotting/view_serve_phases.py firstserve
    python plotting/generate_segmentation_validation.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from serve_animation import run_full_serve_animation
from skeleton_viz import load_serve_markers

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
INDIVIDUAL_DIR = os.path.join(PLOT_DIR, "markers", "individual")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-serve 3D skeleton animation")
    parser.add_argument(
        "serve",
        nargs="?",
        default="firstserve",
        help="Serve folder under plotting/markers/individual/, path to ace_markers.npz, or output dir",
    )
    parser.add_argument("--speed", type=int, choices=(1, 2, 4, 8), default=1)
    args = parser.parse_args()

    serve_path = args.serve
    individual_dir = os.path.join(INDIVIDUAL_DIR, serve_path)
    if os.path.isdir(individual_dir):
        serve_path = individual_dir

    try:
        markers = load_serve_markers(serve_path)
    except FileNotFoundError:
        print(f"Serve not found: {args.serve}")
        raise SystemExit(1) from None

    title = Path(serve_path).stem if os.path.isdir(serve_path) or str(serve_path).endswith(".npz") else args.serve
    run_full_serve_animation(markers, title, speed=args.speed)


if __name__ == "__main__":
    main()
