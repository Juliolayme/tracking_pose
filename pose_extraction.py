from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import NUM_POSE_LANDMARKS


def _require_mediapipe() -> Any:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is required for keypoint extraction. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc
    return mp


def cache_path_for_video(video_path: Path, cache_dir: Path) -> Path:
    video_path = Path(video_path)
    stat = video_path.stat()
    digest = hashlib.sha1(
        f"{video_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:16]
    safe_stem = "".join(ch if ch.isalnum() else "_" for ch in video_path.stem)[:80]
    return Path(cache_dir) / f"{safe_stem}_{digest}.npz"


def extract_pose_sequence(
    video_path: Path,
    sample_fps: float = 15.0,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract only MediaPipe pose coordinates, never image pixels."""
    mp = _require_mediapipe()
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(fps / sample_fps))) if fps > 0 and sample_fps > 0 else 1

    frames: list[np.ndarray] = []
    sampled = 0
    failed = 0
    pose_solution = mp.solutions.pose
    with pose_solution.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride != 0:
                frame_index += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = pose.process(rgb)
            if result.pose_landmarks:
                keypoints = np.array(
                    [[landmark.x, landmark.y, landmark.z] for landmark in result.pose_landmarks.landmark],
                    dtype=np.float32,
                )
            else:
                keypoints = np.full((NUM_POSE_LANDMARKS, 3), np.nan, dtype=np.float32)
                failed += 1
            frames.append(keypoints)
            sampled += 1
            frame_index += 1

    capture.release()

    if not frames:
        raise RuntimeError(f"No frames could be sampled from video: {video_path}")

    metadata = {
        "video_path": str(video_path),
        "source_fps": fps,
        "sample_fps": sample_fps,
        "sample_stride": stride,
        "total_frames": total_frames,
        "sampled_frames": sampled,
        "missing_pose_frames": failed,
        "missing_pose_fraction": failed / max(sampled, 1),
    }
    return np.stack(frames, axis=0).astype(np.float32), metadata


def load_cached_sequence(cache_path: Path) -> np.ndarray:
    with np.load(cache_path, allow_pickle=False) as data:
        return data["keypoints"].astype(np.float32)


def write_pose_cache(
    video_path: Path,
    cache_path: Path,
    label: int,
    class_dir: str,
    group_id: str,
    sample_fps: float,
    refresh: bool = False,
) -> Path:
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    keypoints, metadata = extract_pose_sequence(video_path, sample_fps=sample_fps)
    metadata.update(
        {
            "label": int(label),
            "class_dir": class_dir,
            "group_id": group_id,
        }
    )
    np.savez_compressed(
        cache_path,
        keypoints=keypoints,
        label=np.array(label, dtype=np.int64),
        metadata=json.dumps(metadata),
    )
    return cache_path
