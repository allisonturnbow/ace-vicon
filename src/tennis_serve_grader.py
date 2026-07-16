"""
ACE Tennis Serve Grader — Multi-Joint, Snapshot-Based
------------------------------------------------------
File location: src/tennis_serve_grader.py

Compares a customer's serve against the DTW barycenter (pro reference)
at each of Maximiliano's 8 named serve snapshots, across 5 metrics:

    elbow           — right/racket arm joint angle (shoulder-elbow-hand)
    shoulder        — right/racket arm joint angle (chest-shoulder-elbow)
    hip_rotation    — BOTH hips treated as one unit: angle of the
                      hip-to-hip line relative to the swing's forward
                      direction. Flags if the customer opens their hips
                      earlier/later in the swing than the pro.
    knee_bend       — BOTH knees treated as one unit: flexion angle
                      (hip-knee-foot) for each leg, averaged into one
                      number. Flags if the customer isn't loading/bending
                      their knees as much as the pro at a given snapshot.
    foot_lift       — how far each foot has risen above its own
                      start_pose ("ground") height, compared to how far
                      the pro's foot rose at the same point in the swing.
                      This is a HEIGHT comparison (mm), not an angle.

Each snapshot's score is the average of whichever of the 5 metrics have
valid data at that snapshot; the existing per-snapshot weights (Contact
20%, etc.) then apply to that combined number, same as before.

CHANGE FROM PREVIOUS VERSION:
    Max's old segmentation.pipeline.segment_serve() (8 phases, ranges)
    is replaced by Maximiliano's single-CSV pipeline
    ("formatdata and render"/format_data.py + find_snapshots.py), which
    auto-identifies markers from ONE raw Vicon CSV (no per-marker
    folder, no MARKER_ORDER sidecar needed) and returns 8 named,
    single-frame snapshots:

        start_pose, hand_cross, flat_racket_arm, peak_racket_arm,
        contact, hand_cross_2, racket_deceleration, finish_pose

    DTW alignment to barycenter2.npy is UNCHANGED — still needed to
    translate "this frame in the customer's serve" into "the
    corresponding frame in the pro's swing" for a fair comparison.

KNOWN OPEN ISSUES (flagged in chat, not yet fixed):
    - The DTW-alignment array is still built via dtw.load_data.load_multi_serve()
      + MARKER_ORDER, which assumes markers are already in a fixed column
      order. This is inconsistent with Maximiliano's auto-identification
      and can KeyError or silently mislabel markers on CSVs that don't
      match that assumption (e.g. fewer than 14 tracks, no *_order.py
      sidecar). Someday this should be rebuilt from the SAME
      auto-identified labels format_data.py already produces.
    - Snapshot frames with marker dropout (NaN at that exact frame) give
      `nan` metric values. interpolate_missing.py exists in Maximiliano's
      folder for this and isn't wired in yet.

Pipeline:
    barycenter2.npy (pro reference)
        + format_data.py + find_snapshots.py   (Maximiliano — one raw CSV in)
        + DTW alignment to barycenter          (unchanged)
        → grade_serve()
        → print_report()
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

# ─────────────────────────────────────────────
# 0. LOAD MAXIMILIANO'S MODULES
#    "formatdata and render" has a space in the folder name, so it
#    can't be imported normally (`from formatdata and render import x`
#    is a syntax error). Loaded by file path instead.
#    TODO(Jaime): ask Max to rename the folder to `formatdata_and_render`
#    so this can become a normal import.
# ─────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_FORMATDATA_DIR = os.path.join(_HERE, "..", "formatdata and render")


def _load_module_from_path(module_name: str, filename: str):
    path = os.path.join(_FORMATDATA_DIR, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_format_data = _load_module_from_path("ace_format_data", "format_data.py")
_find_snapshots = _load_module_from_path("ace_find_snapshots", "find_snapshots.py")

# ─────────────────────────────────────────────
# 0b. dtw/ MODULES (marker order + raw CSV loading for DTW alignment)
# ─────────────────────────────────────────────

_DTW_DIR = os.path.join(_HERE, "..", "dtw")
if _DTW_DIR not in sys.path:
    sys.path.insert(0, _DTW_DIR)

from constants import MARKER_ORDER  # noqa: E402
from load_data import load_multi_serve  # noqa: E402
from prepare_data import (  # noqa: E402
    interpolate_nans,
    filter_nan_frames,
    normalize_serve,
    convert,
)

# ─────────────────────────────────────────────
# 1. BARYCENTER PATH
# ─────────────────────────────────────────────

DEFAULT_BARYCENTER = os.path.join(_HERE, "..", "dtw", "barycenter2.npy")

# ─────────────────────────────────────────────
# 2. SNAPSHOT NAMES + WEIGHTS
#    Order matches find_snapshots.SNAPSHOT_NAMES (canonical order).
# ─────────────────────────────────────────────

SNAPSHOT_NAMES = [
    "start_pose",
    "hand_cross",
    "flat_racket_arm",
    "peak_racket_arm",
    "contact",
    "hand_cross_2",
    "racket_deceleration",
    "finish_pose",
]

SNAPSHOT_WEIGHTS = {
    "start_pose":          0.05,
    "hand_cross":          0.15,
    "flat_racket_arm":     0.25,
    "peak_racket_arm":     0.20,
    "contact":             0.20,
    "hand_cross_2":        0.05,
    "racket_deceleration": 0.05,
    "finish_pose":         0.05,
}

SNAPSHOT_CONTEXT = {
    "start_pose":          "your starting pose at the top of the ball toss",
    "hand_cross":          "the moment your racket hand overtakes your toss hand",
    "flat_racket_arm":     "the trophy position — forearm level with the elbow",
    "peak_racket_arm":     "the top of your racket-arm swing, just before the drop into acceleration",
    "contact":             "ball contact — elbow should be near full extension",
    "hand_cross_2":        "the follow-through moment your hands cross again",
    "racket_deceleration": "your deceleration — controlled arm slowdown after contact",
    "finish_pose":         "your follow-through and finish position",
}

# ─────────────────────────────────────────────
# 3. METRICS
#    Each metric has a human label, unit ("deg" or "mm"), and its own
#    scoring thresholds (same 4-tier shape, different scale for mm).
# ─────────────────────────────────────────────

METRIC_LABELS = {
    "elbow":        "Elbow angle",
    "shoulder":     "Shoulder angle",
    "hip_rotation": "Hip rotation",
    "knee_bend":    "Knee bend",
    "foot_lift":    "Foot lift off ground",
}

ANGLE_THRESHOLDS = [       # (diff_deg, score, tier)
    (5,   95, "Excellent"),
    (15,  80, "Good"),
    (30,  60, "Fair"),
    (999, 35, "Poor"),
]

DISTANCE_THRESHOLDS = [    # (diff_mm, score, tier) — tune once you have real data
    (20,    95, "Excellent"),
    (50,    80, "Good"),
    (100,   60, "Fair"),
    (99999, 35, "Poor"),
]

METRIC_THRESHOLDS = {
    "elbow":        ANGLE_THRESHOLDS,
    "shoulder":     ANGLE_THRESHOLDS,
    "hip_rotation": ANGLE_THRESHOLDS,
    "knee_bend":    ANGLE_THRESHOLDS,
    "foot_lift":    DISTANCE_THRESHOLDS,
}

TIER_ICON = {"Excellent": "✓", "Good": "~", "Fair": "!", "Poor": "✗"}

FEEDBACK_TEMPLATES = {
    "Excellent": "{metric} matches the pro at {phase_label} — excellent.",
    "Good":      "{metric} is slightly off at {phase_label}. Minor adjustment needed for {context}.",
    "Fair":      "{metric} deviates noticeably at {phase_label}. Focus on {context}.",
    "Poor":      "{metric} is significantly off at {phase_label}. Prioritise drills for {context}.",
}


def score_metric(diff: float, metric: str) -> tuple:
    for threshold, points, tier in METRIC_THRESHOLDS[metric]:
        if diff < threshold:
            return points, tier
    return 35, "Poor"


# ─────────────────────────────────────────────
# 4. MARKER COLUMN INDICES (barycenter array only — 42-length rows,
#    MARKER_ORDER layout from dtw/constants.py)
# ─────────────────────────────────────────────

_BC_SLICE = {
    "head":           slice(0, 3),
    "left_shoulder":  slice(3, 6),
    "right_elbow":    slice(6, 9),
    "left_elbow":     slice(9, 12),
    "chest":          slice(12, 15),
    "right_shoulder": slice(15, 18),
    "right_knee":     slice(18, 21),
    "right_hand":     slice(21, 24),
    "left_foot":      slice(24, 27),
    "left_knee":      slice(27, 30),
    "right_hip":      slice(30, 33),
    "left_hand":      slice(33, 36),
    "left_hip":       slice(36, 39),
    "right_foot":     slice(39, 42),
}

_NEEDED_MARKERS = [
    "chest", "right_shoulder", "right_elbow", "right_hand",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_foot", "right_foot",
]


def _markers_from_barycenter_row(frame: np.ndarray) -> dict:
    return {name: frame[sl] for name, sl in _BC_SLICE.items()}


def _markers_from_labeled_row(row) -> dict:
    out = {}
    for name in list(_BC_SLICE.keys()):
        tx_col, ty_col, tz_col = f"{name}_TX", f"{name}_TY", f"{name}_TZ"
        if tx_col in row.index:
            out[name] = np.array([row[tx_col], row[ty_col], row[tz_col]], dtype=float)
    return out


# ─────────────────────────────────────────────
# 5. GEOMETRY HELPERS
# ─────────────────────────────────────────────

def _angle_at_vertex(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    v1 = a - vertex
    v2 = b - vertex
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _signed_rotation_deg(vec_xy: np.ndarray, axis_xy: np.ndarray) -> float:
    """Signed angle (degrees) from axis_xy to vec_xy, in the XY plane."""
    dot   = axis_xy[0] * vec_xy[0] + axis_xy[1] * vec_xy[1]
    cross = axis_xy[0] * vec_xy[1] - axis_xy[1] * vec_xy[0]
    return float(np.degrees(np.arctan2(cross, dot)))


def _bc_forward_right_axis(barycenter: np.ndarray):
    """
    Forward/right axis for the pro barycenter, computed the same way
    format_data.py computes it for the customer: direction the racket
    (right) hand travels between its own ball-toss-equivalent and
    follow-through-equivalent peaks.
    """
    lhan_tz = barycenter[:, _BC_SLICE["left_hand"]][:, 2]
    rhan_tz = barycenter[:, _BC_SLICE["right_hand"]][:, 2]
    peak1 = int(np.argmax(lhan_tz))
    peak2 = int(np.argmax(rhan_tz))

    start_xy = barycenter[peak1, _BC_SLICE["right_hand"]][:2]
    end_xy   = barycenter[peak2, _BC_SLICE["right_hand"]][:2]
    dx, dy = float(end_xy[0] - start_xy[0]), float(end_xy[1] - start_xy[1])

    if abs(dx) >= abs(dy):
        forward = np.array([1.0, 0.0]) if dx >= 0 else np.array([-1.0, 0.0])
    else:
        forward = np.array([0.0, 1.0]) if dy >= 0 else np.array([0.0, -1.0])
    right = np.array([forward[1], -forward[0]])
    return forward, right


# ─────────────────────────────────────────────
# 6. PER-METRIC CALCULATORS
#    Each takes a marker dict (+ right_axis where needed, + a ground
#    baseline for foot_lift) and returns a float in the metric's unit.
# ─────────────────────────────────────────────

def metric_elbow(m: dict) -> float:
    return _angle_at_vertex(m["right_shoulder"], m["right_elbow"], m["right_hand"])


def metric_shoulder(m: dict) -> float:
    return _angle_at_vertex(m["chest"], m["right_shoulder"], m["right_elbow"])


def metric_hip_rotation(m: dict, right_axis_xy: np.ndarray) -> float:
    vec = (m["right_hip"] - m["left_hip"])[:2]
    return _signed_rotation_deg(vec, right_axis_xy)


def metric_knee_bend(m: dict) -> float:
    """
    Knee flexion angle (hip-knee-foot) for each leg, averaged into one
    number — 180° = leg straight, smaller = more bent/loaded.
    """
    right_bend = _angle_at_vertex(m["right_hip"], m["right_knee"], m["right_foot"])
    left_bend  = _angle_at_vertex(m["left_hip"],  m["left_knee"],  m["left_foot"])
    return (right_bend + left_bend) / 2.0


def metric_foot_lift(m: dict, ground_z: float) -> float:
    """Height (mm) of the higher foot above the start_pose ground baseline."""
    return float(max(m["right_foot"][2], m["left_foot"][2]) - ground_z)


# ─────────────────────────────────────────────
# 7. MAIN GRADING FUNCTION
# ─────────────────────────────────────────────

def grade_serve(snapshot_metrics: dict) -> dict:
    """
    Grade a customer's serve across 5 metrics at each of the 8 snapshots.

    Parameters
    ----------
    snapshot_metrics : dict
        {
          snapshot_name: {
            "elbow":         (customer_val, pro_val) | None,
            "shoulder":       ...,
            "hip_rotation":   ...,
            "knee_rotation":  ...,
            "foot_lift":      ...,
          },
          ...
        }
        Any metric missing/None at a snapshot is simply skipped for that
        snapshot's combined score.

    Returns
    -------
    dict with per-snapshot, per-metric results, plus overall score/grade.
    """
    results = {"snapshots": {}}
    weighted_total = 0.0
    weight_used    = 0.0

    for snap in SNAPSHOT_NAMES:
        metrics = snapshot_metrics.get(snap, {})
        metric_results = {}
        metric_scores  = []

        for metric_name in METRIC_LABELS:
            pair = metrics.get(metric_name)
            if pair is None:
                continue
            customer_val, pro_val = pair
            if customer_val is None or pro_val is None or np.isnan(customer_val) or np.isnan(pro_val):
                continue

            diff = abs(customer_val - pro_val)
            score, tier = score_metric(diff, metric_name)
            snap_label = snap.replace("_", " ")
            feedback = FEEDBACK_TEMPLATES[tier].format(
                metric=METRIC_LABELS[metric_name],
                phase_label=snap_label,
                context=SNAPSHOT_CONTEXT[snap],
            )
            metric_results[metric_name] = {
                "customer": round(customer_val, 1),
                "pro":      round(pro_val, 1),
                "diff":     round(diff, 1),
                "score":    score,
                "tier":     tier,
                "feedback": feedback,
            }
            metric_scores.append(score)

        if not metric_scores:
            results["snapshots"][snap] = {"score": None, "metrics": {}, "summary": f"{snap} — no data"}
            continue

        combined_score = round(sum(metric_scores) / len(metric_scores), 1)
        results["snapshots"][snap] = {"score": combined_score, "metrics": metric_results}

        w = SNAPSHOT_WEIGHTS[snap]
        weighted_total += combined_score * w
        weight_used    += w

    if weight_used == 0:
        results["overall_score"] = None
        results["overall_grade"] = "No data provided."
        return results

    overall = round(weighted_total / weight_used, 1)
    results["overall_score"] = overall

    if overall >= 90:
        results["overall_grade"] = "A — Pro-level serve"
    elif overall >= 75:
        results["overall_grade"] = "B — Strong serve, minor adjustments needed"
    elif overall >= 60:
        results["overall_grade"] = "C — Developing serve, focused practice recommended"
    else:
        results["overall_grade"] = "D — Fundamentals need significant work"

    return results


# ─────────────────────────────────────────────
# 8. REPORT PRINTER
# ─────────────────────────────────────────────

def print_report(results: dict) -> None:
    sep = "─" * 70
    print(f"\n{'ACE TENNIS SERVE — MULTI-JOINT REPORT':^70}")
    print(sep)

    for snap in SNAPSHOT_NAMES:
        s = results["snapshots"].get(snap)
        if not s or s.get("score") is None:
            print(f"\n  {snap.replace('_', ' ')}  — no data")
            continue

        weight_pct = int(SNAPSHOT_WEIGHTS[snap] * 100)
        print(f"\n  {snap.replace('_', ' ')}  ({weight_pct}% of grade)  — combined score: {s['score']}/100")
        for metric_name, p in s["metrics"].items():
            icon = TIER_ICON[p["tier"]]
            unit = "°" if METRIC_THRESHOLDS[metric_name] is ANGLE_THRESHOLDS else "mm"
            print(f"    [{icon}] {METRIC_LABELS[metric_name]:<22s} {p['tier']:9s}  "
                  f"Score: {p['score']:3d}/100   Customer: {p['customer']}{unit}  "
                  f"Pro: {p['pro']}{unit} ")
            print(f"        → {p['feedback']}")

    print(f"\n{sep}")
    print(f"  OVERALL SCORE : {results['overall_score']}/100")
    print(f"  GRADE         : {results['overall_grade']}")
    print(sep)


# ─────────────────────────────────────────────
# 9. DTW ALIGNMENT (unchanged in spirit — also returns the alignment
#    object so a specific customer frame can be mapped to its
#    corresponding barycenter frame)
# ─────────────────────────────────────────────

def align_to_barycenter(customer_array, barycenter):
    from dtw import dtw
    from scipy.spatial.distance import cdist

    print("Aligning customer serve to pro barycenter via DTW...")
    dist_mat  = cdist(customer_array, barycenter)
    alignment = dtw(dist_mat, distance_only=False)

    n_bc  = barycenter.shape[0]
    accum = [[] for _ in range(n_bc)]
    for cust_idx, bc_idx in zip(alignment.index1, alignment.index2):
        accum[bc_idx].append(customer_array[cust_idx])

    aligned = np.zeros_like(barycenter)
    for j in range(n_bc):
        if accum[j]:
            aligned[j] = np.mean(accum[j], axis=0)
        else:
            aligned[j] = customer_array[
                alignment.index1[np.argmin(np.abs(alignment.index2 - j))]
            ]

    print(f"  Customer: {customer_array.shape[0]} frames -> aligned to {n_bc} frames")
    return aligned, alignment


def map_customer_idx_to_barycenter_idx(alignment, customer_idx: int) -> int:
    index1 = np.asarray(alignment.index1)
    index2 = np.asarray(alignment.index2)

    matches = index2[index1 == customer_idx]
    if len(matches) > 0:
        return int(round(matches.mean()))

    nearest = np.argmin(np.abs(index1 - customer_idx))
    return int(index2[nearest])


# ─────────────────────────────────────────────
# 10. CUSTOMER-SIDE PIPELINE (Maximiliano's single-CSV marker + snapshot
#     identification — replaces Max's old segmentation.pipeline)
# ─────────────────────────────────────────────

def run_maximiliano_pipeline(raw_csv_path: str, tmp_dir: str) -> dict:
    """
    Run format_data.py's marker-identification + orientation logic, then
    find_snapshots.py's snapshot-finding logic, on one raw multi-marker
    Vicon CSV.

    Returns
    -------
    dict with:
        "snapshots"     : {snapshot_name: frame_number}  (0 = not found)
        "formatted_df"  : labeled DataFrame (right_shoulder_TX, etc.)
        "right_axis_xy" : customer's own forward-swing "right" axis (2,)
    """
    fd = _format_data

    print(f"\n[Maximiliano pipeline] Parsing raw CSV: {raw_csv_path}")
    df, tz_cols, tracks = fd.parse_vicon_csv(raw_csv_path)

    excluded = set()
    head, _        = fd.identify_head(df, tracks, excluded)
    excluded.add(head)
    ball_hand, _   = fd.identify_ball_hand(df, tracks, head)
    excluded.add(ball_hand)

    bh_tz = df[ball_hand + "_TZ"].dropna()
    peak1_fi = int(bh_tz.idxmax())

    racket_hand, _ = fd.identify_racket_hand(df, tracks, excluded)
    excluded.add(racket_hand)

    rh_tz = df[racket_hand + "_TZ"].dropna()
    peak2_fi = int(rh_tz.idxmax())

    elbow1, _ = fd.identify_elbow1(df, tracks, excluded, peak2_fi)
    excluded.add(elbow1)

    peaks = [
        {"col": ball_hand + "_TZ",   "frame_idx": peak1_fi, "label": "Ball Toss"},
        {"col": racket_hand + "_TZ", "frame_idx": peak2_fi, "label": "Follow Through"},
    ]
    peaks.sort(key=lambda x: x["frame_idx"])

    labels_map = {
        "head": head, "ball_hand": ball_hand,
        "racket_hand": racket_hand, "elbow1": elbow1,
    }

    swing_start = min(p["frame_idx"] for p in peaks)
    swing_end   = max(p["frame_idx"] for p in peaks)

    body_parts, _, _, _ = fd.assign_body_parts(df, tracks, excluded, swing_start, swing_end)

    # Customer's own forward/right axis, computed the same way format_data
    # computes it internally for left/right orientation.
    forward = fd.determine_forward_axis(df, racket_hand, peaks[0]["frame_idx"], peaks[1]["frame_idx"])
    right_axis_xy = fd.right_axis_from_forward(forward)[:2]

    labels_map, body_parts = fd.orient_labels(
        df, labels_map, body_parts, peaks[0]["frame_idx"], peaks[1]["frame_idx"]
    )

    base = os.path.splitext(os.path.basename(raw_csv_path))[0]
    formatted_path = os.path.join(tmp_dir, base + "_formatted.csv")
    fd.export_annotated_csv(raw_csv_path, peaks, labels_map, body_parts, out_path=formatted_path)

    print("[Maximiliano pipeline] Finding snapshot frames...")
    fs = _find_snapshots
    fs_df, fs_tz_cols, fs_part_names, fs_peaks, _, _, _ = fs.read_formatted_csv(formatted_path)
    snapshots = fs.find_snapshots(fs_df, fs_tz_cols, fs_part_names, fs_peaks)

    return {"snapshots": snapshots, "formatted_df": fs_df, "right_axis_xy": right_axis_xy}


def _frame_to_df_idx(df, frame_num: int) -> int:
    matches = df.index[df["Frame"] == frame_num].tolist()
    return matches[0] if matches else min(frame_num, len(df) - 1)


# ─────────────────────────────────────────────
# 11. TOP-LEVEL RUNNERS
# ─────────────────────────────────────────────

def run_from_csv(raw_csv_path: str) -> dict:
    """
    Run the full grading pipeline on one raw, unlabeled, multi-marker
    Vicon CSV (Maximiliano's expected input format).
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        mx = run_maximiliano_pipeline(raw_csv_path, tmp_dir)
        snapshots         = mx["snapshots"]        # {name: frame_number, 0 = not found}
        formatted_df      = mx["formatted_df"]     # labeled columns: right_shoulder_TX, etc.
        cust_right_axis   = mx["right_axis_xy"]

        # Customer's own "ground" baseline for foot_lift — start_pose frame.
        ground_frame_num = snapshots.get("start_pose", 0)
        ground_z = None
        if ground_frame_num:
            g_idx = _frame_to_df_idx(formatted_df, ground_frame_num)
            g_row = formatted_df.iloc[g_idx]
            m_ground = _markers_from_labeled_row(g_row)
            if "right_foot" in m_ground and "left_foot" in m_ground:
                ground_z = float(max(m_ground["right_foot"][2], m_ground["left_foot"][2]))

        print("\nSnapshot frames found:")
        snapshot_customer_frame_idx = {}
        snapshot_customer_markers   = {}
        for name in SNAPSHOT_NAMES:
            frame_num = snapshots.get(name, 0)
            if frame_num == 0:
                print(f"  {name:20s} — not found, skipping")
                continue
            df_idx = _frame_to_df_idx(formatted_df, frame_num)
            snapshot_customer_frame_idx[name] = df_idx
            snapshot_customer_markers[name]   = _markers_from_labeled_row(formatted_df.iloc[df_idx])
            print(f"  {name:20s} frame={frame_num:5d}")

        # ── DTW alignment (separate from Maximiliano's pipeline) ──
        print("\nLoading raw CSV for DTW alignment...")
        serve = load_multi_serve(raw_csv_path)
        serve = filter_nan_frames(serve)
        serve = interpolate_nans(serve)
        serve = normalize_serve(serve)

        barycenter   = np.load(DEFAULT_BARYCENTER)
        marker_array = convert(serve)
        _, alignment = align_to_barycenter(marker_array, barycenter)
        pro_right_axis = _bc_forward_right_axis(barycenter)[1]

        serve_frames = serve["frames"]

        # Pro's own ground baseline (start_pose's corresponding barycenter frame)
        pro_ground_z = None
        if "start_pose" in snapshot_customer_frame_idx:
            frame_num = snapshots["start_pose"]
            nearest_serve_idx = int(np.argmin(np.abs(serve_frames - frame_num)))
            bc_idx = map_customer_idx_to_barycenter_idx(alignment, nearest_serve_idx)
            m_pro_ground = _markers_from_barycenter_row(barycenter[bc_idx])
            pro_ground_z = float(max(m_pro_ground["right_foot"][2], m_pro_ground["left_foot"][2]))

        snapshot_metrics = {}
        for name, df_idx in snapshot_customer_frame_idx.items():
            frame_num = snapshots[name]
            nearest_serve_idx = int(np.argmin(np.abs(serve_frames - frame_num)))
            bc_idx = map_customer_idx_to_barycenter_idx(alignment, nearest_serve_idx)

            m_cust = snapshot_customer_markers[name]
            m_pro  = _markers_from_barycenter_row(barycenter[bc_idx])

            metrics = {}
            try:
                metrics["elbow"] = (metric_elbow(m_cust), metric_elbow(m_pro))
            except KeyError:
                pass
            try:
                metrics["shoulder"] = (metric_shoulder(m_cust), metric_shoulder(m_pro))
            except KeyError:
                pass
            try:
                metrics["hip_rotation"] = (
                    metric_hip_rotation(m_cust, cust_right_axis),
                    metric_hip_rotation(m_pro, pro_right_axis),
                )
            except KeyError:
                pass
            try:
                metrics["knee_bend"] = (
                    metric_knee_bend(m_cust),
                    metric_knee_bend(m_pro),
                )
            except KeyError:
                pass
            if ground_z is not None and pro_ground_z is not None:
                try:
                    metrics["foot_lift"] = (
                        metric_foot_lift(m_cust, ground_z),
                        metric_foot_lift(m_pro, pro_ground_z),
                    )
                except KeyError:
                    pass

            snapshot_metrics[name] = metrics

        return grade_serve(snapshot_metrics)


def run_dummy() -> dict:
    """Run grader with dummy data for testing without a real CSV."""
    snapshot_metrics = {
        "start_pose":          {"elbow": (168.0, 165.0), "shoulder": (40.0, 38.0), "hip_rotation": (5.0, 4.0),  "knee_bend": (170.0, 172.0), "foot_lift": (2.0, 0.0)},
        "hand_cross":          {"elbow": (148.0, 150.0), "shoulder": (55.0, 52.0), "hip_rotation": (20.0, 12.0), "knee_bend": (155.0, 150.0), "foot_lift": (10.0, 15.0)},
        "flat_racket_arm":     {"elbow": (110.0, 99.0),  "shoulder": (85.0, 88.0), "hip_rotation": (45.0, 40.0), "knee_bend": (120.0, 115.0), "foot_lift": (30.0, 45.0)},
        "peak_racket_arm":     {"elbow": (128.0, 124.0), "shoulder": (100.0, 102.0), "hip_rotation": (60.0, 58.0), "knee_bend": (115.0, 110.0), "foot_lift": (60.0, 80.0)},
        "contact":             {"elbow": (172.0, 143.0), "shoulder": (150.0, 152.0), "hip_rotation": (85.0, 82.0), "knee_bend": (150.0, 155.0), "foot_lift": (90.0, 110.0)},
        "hand_cross_2":        {"elbow": (150.0, 143.0), "shoulder": (120.0, 118.0), "hip_rotation": (95.0, 90.0), "knee_bend": (160.0, 158.0), "foot_lift": (40.0, 50.0)},
        "racket_deceleration": {"elbow": (145.0, 128.0), "shoulder": (90.0, 92.0), "hip_rotation": (100.0, 98.0), "knee_bend": (165.0, 162.0), "foot_lift": (15.0, 20.0)},
        "finish_pose":         {"elbow": (95.0, 99.0),   "shoulder": (60.0, 58.0), "hip_rotation": (105.0, 100.0), "knee_bend": (168.0, 165.0), "foot_lift": (0.0, 0.0)},
    }
    return grade_serve(snapshot_metrics)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ACE Tennis Serve Grader")
    parser.add_argument(
        "--csv", metavar="FILE", default=None,
        help="Path to ONE raw multi-marker Vicon CSV. Omit to use dummy data.",
    )
    args = parser.parse_args()

    if args.csv:
        results = run_from_csv(args.csv)
    else:
        print("No CSV provided — running with dummy data.")
        results = run_dummy()

    print_report(results)