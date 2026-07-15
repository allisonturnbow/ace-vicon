import json
import warnings
from pathlib import Path

import numpy as np

from src.markers.io import save_serve_markers
from src.motionbert.ace_adapter import motionbert_to_ace_markers
from src.motionbert.motionbert_runner import MOTIONBERT_JOINT_NAMES
from src.motionbert.view_ace_animation import run_ace_animation


def _sample_markers(frames: int = 3) -> dict:
    pose = np.zeros((frames, len(MOTIONBERT_JOINT_NAMES), 3), dtype=float)
    return motionbert_to_ace_markers(pose, scale=1.0)


def test_run_ace_animation_delegates_to_standard_renderer(tmp_path, monkeypatch):
    markers = _sample_markers()
    save_serve_markers(tmp_path, markers)
    called = {}

    def fake_animation(loaded, title, *, speed=1):
        called["title"] = title
        called["frames"] = len(loaded["frames"])
        called["speed"] = speed

    monkeypatch.setattr(
        "src.motionbert.view_ace_animation.run_full_serve_animation",
        fake_animation,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        run_ace_animation(tmp_path / "ace_markers.npz", title="demo", speed=2)

    assert called["title"] == "demo"
    assert called["frames"] == 3
    assert called["speed"] == 2
