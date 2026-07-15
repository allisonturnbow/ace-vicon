"""Convert MotionBERT 3D skeleton output into the canonical ACE marker dictionary."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.markers.io import ACE_MARKER_NAMES, save_serve_markers
from src.motionbert.motionbert_runner import MOTIONBERT_JOINT_NAMES

# Normalized skeleton coordinates use shoulder width ~= 1.0.
# Scale to approximate Vicon millimetre units so segmentation thresholds apply.
DEFAULT_VICON_MM_SCALE = 400.0

ACE_TO_MOTIONBERT = {
    "head": "head",
    "chest": "thorax",
    "left_shoulder": "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_hand": "left_wrist",
    "right_hand": "right_wrist",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
    "left_foot": "left_ankle",
    "right_foot": "right_ankle",
}


def _validate_pose_3d(poses_3d: np.ndarray) -> np.ndarray:
    pose = np.asarray(poses_3d, dtype=float)
    if pose.ndim != 3 or pose.shape[1:] != (len(MOTIONBERT_JOINT_NAMES), 3):
        raise ValueError(
            "poses_3d must have shape "
            f"(frames, {len(MOTIONBERT_JOINT_NAMES)}, 3); got {pose.shape}"
        )
    return pose


def motionbert_to_ace_markers(
    poses_3d: np.ndarray,
    *,
    frame_start: int = 1,
    scale: float = DEFAULT_VICON_MM_SCALE,
) -> dict[str, dict[str, np.ndarray] | np.ndarray]:
    """Convert 17-joint MotionBERT output into ACE's Vicon marker dictionary.

    Expects **normalized** skeleton input (``normalize_skeleton`` output) where
    canonical Y points up (head Y > foot Y).

    Coordinate mapping (normalized body frame → ACE/Vicon axes):
      TX ← X  (lateral)
      TY ← Z  (depth)
      TZ ← Y  (vertical; Vicon Z is up)

    The returned dict matches ``load_single_serve()`` output exactly.
    """
    pose = _validate_pose_3d(poses_3d)
    frames = np.arange(frame_start, frame_start + pose.shape[0], dtype=int)
    markers: dict[str, dict[str, np.ndarray] | np.ndarray] = {"frames": frames}

    for ace_name in ACE_MARKER_NAMES:
        mb_name = ACE_TO_MOTIONBERT[ace_name]
        idx = MOTIONBERT_JOINT_NAMES.index(mb_name)
        coords = pose[:, idx, :] * float(scale)
        markers[ace_name] = {
            "TX": coords[:, 0].astype(float),
            "TY": coords[:, 2].astype(float),
            "TZ": coords[:, 1].astype(float),
        }

    return markers


def save_ace_markers(
    output_dir: str | Path,
    markers: dict[str, dict[str, np.ndarray] | np.ndarray],
    *,
    filename: str = "ace_markers.npz",
) -> Path:
    """Persist markers; delegates to ``save_serve_markers()``."""
    return save_serve_markers(output_dir, markers, filename=filename)


def load_ace_markers(npz_path: str | Path) -> dict[str, dict[str, np.ndarray] | np.ndarray]:
    """Load markers from NPZ; delegates to ``load_serve_markers()``."""
    from src.markers.io import load_serve_markers

    return load_serve_markers(npz_path)


def convert_file_to_ace_markers(
    poses_3d_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    scale: float = DEFAULT_VICON_MM_SCALE,
) -> Path:
    path = Path(poses_3d_path)
    out = Path(output_dir) if output_dir is not None else path.parent
    pose = np.load(path)
    markers = motionbert_to_ace_markers(pose, scale=scale)
    return save_ace_markers(out, markers)
