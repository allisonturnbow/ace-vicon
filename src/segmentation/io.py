from __future__ import annotations

from pathlib import Path


def load_serve_from_folder(serve_dir: str | Path) -> dict:
    """Load ACE markers from a Vicon CSV folder."""
    from src.markers.io import load_serve_markers

    return load_serve_markers(serve_dir)
