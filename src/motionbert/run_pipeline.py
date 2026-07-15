from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLOTTING_DIR = _REPO_ROOT / "plotting"
if str(_PLOTTING_DIR) not in sys.path:
    sys.path.insert(0, str(_PLOTTING_DIR))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from segmentation import SegmentationConfig, segment_serve  # noqa: E402
from segmentation.cli import format_phase_frames  # noqa: E402
from serve_animation import run_full_serve_animation  # noqa: E402
from segmentation_viz import run_interactive_viewer  # noqa: E402

from src.motionbert.mediapipe_extractor import extract_video  # noqa: E402
from src.motionbert.motionbert_runner import (  # noqa: E402
    DEFAULT_MOTIONBERT_CHECKPOINT,
    DEFAULT_MOTIONBERT_DIR,
    run_motionbert_stage,
)
from src.video_discovery import find_videos, latest_video  # noqa: E402


def process_video(
    video_path: str | Path,
    *,
    output_root: str | Path = "generated_motionbert",
    backend: str = "auto",
    motionbert_command: str | None = None,
    pose_model_path: str | Path | None = None,
    motionbert_dir: str | Path = DEFAULT_MOTIONBERT_DIR,
    checkpoint_path: str | Path | None = DEFAULT_MOTIONBERT_CHECKPOINT,
) -> tuple[Path, dict[str, Any]]:
    """Run video → MediaPipe → MotionBERT → ACE marker dictionary."""
    output_dir = extract_video(video_path, output_root, pose_model_path=pose_model_path)
    markers = run_motionbert_stage(
        output_dir,
        backend=backend,
        motionbert_command=motionbert_command,
        motionbert_dir=motionbert_dir,
        checkpoint_path=checkpoint_path,
        video_path=video_path,
    )
    return output_dir, markers


def segment_markers(
    markers: dict[str, Any],
    *,
    use_v2: bool = True,
) -> Any:
    """Run the existing ACE segmentation pipeline on a marker dictionary."""
    cfg = SegmentationConfig(use_legacy_detection=not use_v2)
    return segment_serve(markers, cfg)


def print_segmentation_report(
    markers: dict[str, Any],
    *,
    serve_name: str = "video",
    use_v2: bool = True,
) -> Any:
    """Segment a serve and print detected phases to stdout."""
    result = segment_markers(markers, use_v2=use_v2)
    print(f"\n{'=' * 60}")
    print(f"ACE SERVE SEGMENTATION — {serve_name}")
    print(f"{'=' * 60}")
    if result.warnings:
        for warning in result.warnings:
            print(f"  WARNING: {warning}")
    print("\nPhases (Vicon frame ranges):")
    print(format_phase_frames(result.phases))
    print("\nEvents:")
    for name, frame in result.events.items():
        conf = result.event_confidence.get(name, 0.0)
        print(f"  {name:40s} frame {frame:5d}  conf={conf:.2f}")
    return result


def run_unified_demo(
    video_path: str | Path,
    *,
    output_root: str | Path = "generated_motionbert",
    backend: str = "geometric",
    animate: bool = False,
    phases_view: bool = False,
    use_v2: bool = True,
    speed: int = 1,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any], Any]:
    """Process a video through the unified ACE pipeline and optionally visualize."""
    video = Path(video_path)
    serve_name = video.stem
    print(f"Processing {video}")
    output_dir, markers = process_video(
        video,
        output_root=output_root,
        backend=backend,
        **kwargs,
    )
    print(f"ACE markers ready: {len(markers['frames'])} frames → {output_dir / 'ace_markers.npz'}")

    result = print_segmentation_report(markers, serve_name=serve_name, use_v2=use_v2)

    if phases_view:
        run_interactive_viewer(serve_name=serve_name, markers=markers, cli_speed=speed)
    elif animate:
        run_full_serve_animation(markers, serve_name, speed=speed)

    return output_dir, markers, result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified pipeline: video → MotionBERT → ACE markers → segmentation → visualization."
    )
    parser.add_argument("--video", default=None, help="Path to a single video file.")
    parser.add_argument("--video-dir", default="2d_video", help="Directory to scan when --video is omitted.")
    parser.add_argument("--output-root", default="generated_motionbert", help="Directory for generated outputs.")
    parser.add_argument(
        "--view",
        action="store_true",
        help="After processing, open the standard ACE full-serve animation (plotting/plot.py renderer).",
    )
    parser.add_argument(
        "--phases",
        action="store_true",
        help="After processing, open the standard ACE phase viewer (plotting/view_serve_phases.py renderer).",
    )
    parser.add_argument(
        "--legacy-segmentation",
        action="store_true",
        help="Use legacy 8-phase segmentation instead of v2.",
    )
    parser.add_argument("--speed", type=int, default=1, help="Playback speed for --view / --phases.")
    parser.add_argument("--backend", choices=("auto", "external", "geometric"), default="auto")
    parser.add_argument("--pose-model", default=None, help="Optional MediaPipe PoseLandmarker .task model path.")
    parser.add_argument("--motionbert-dir", default=str(DEFAULT_MOTIONBERT_DIR))
    parser.add_argument("--checkpoint", default=str(DEFAULT_MOTIONBERT_CHECKPOINT))
    parser.add_argument("--motionbert-command", default=None)
    args = parser.parse_args()

    use_v2 = not args.legacy_segmentation

    if args.video:
        videos = [Path(args.video)]
    elif args.view:
        videos = [latest_video(args.video_dir)]
    else:
        videos = find_videos(args.video_dir)
        if not videos:
            raise SystemExit(
                f"No videos found in {args.video_dir}. Add an .mp4 file or pass --video PATH."
            )

    last_result = None
    last_markers = None
    last_name = None

    try:
        for video in videos:
            output_dir, markers = process_video(
                video,
                output_root=args.output_root,
                backend=args.backend,
                motionbert_command=args.motionbert_command,
                pose_model_path=args.pose_model,
                motionbert_dir=args.motionbert_dir,
                checkpoint_path=args.checkpoint,
            )
            last_markers = markers
            last_name = video.stem
            last_result = print_segmentation_report(markers, serve_name=last_name, use_v2=use_v2)
            print(f"\nSaved outputs under {output_dir}")
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    if last_markers is not None and (args.view or args.phases):
        if args.phases:
            run_interactive_viewer(serve_name=last_name, markers=last_markers, cli_speed=args.speed)
        else:
            run_full_serve_animation(last_markers, last_name or "serve", speed=args.speed)


if __name__ == "__main__":
    main()
