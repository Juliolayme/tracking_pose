from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .config import COORD_DIMS, NUM_POSE_LANDMARKS


LEFT_RIGHT_LANDMARK_PAIRS = (
    (1, 4),
    (2, 5),
    (3, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20),
    (21, 22),
    (23, 24),
    (25, 26),
    (27, 28),
    (29, 30),
    (31, 32),
)

ANGLE_TRIPLETS = (
    (11, 13, 15),  # left elbow
    (12, 14, 16),  # right elbow
    (13, 11, 23),  # left shoulder
    (14, 12, 24),  # right shoulder
    (11, 23, 25),  # left hip
    (12, 24, 26),  # right hip
    (23, 25, 27),  # left knee
    (24, 26, 28),  # right knee
)


def validate_keypoint_sequence(sequence: np.ndarray) -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D keypoint array, got shape {array.shape}.")
    if array.shape[1:] != (NUM_POSE_LANDMARKS, COORD_DIMS):
        raise ValueError(
            "Expected keypoints shaped "
            f"(T, {NUM_POSE_LANDMARKS}, {COORD_DIMS}), got {array.shape}."
        )
    return array


def fill_missing_keypoints(sequence: np.ndarray) -> np.ndarray:
    """Interpolate missing MediaPipe frames marked as NaN."""
    sequence = validate_keypoint_sequence(sequence).copy()
    if sequence.shape[0] == 0:
        return np.zeros((1, NUM_POSE_LANDMARKS, COORD_DIMS), dtype=np.float32)

    frame_idx = np.arange(sequence.shape[0], dtype=np.float32)
    flat = sequence.reshape(sequence.shape[0], -1)
    for col in range(flat.shape[1]):
        values = flat[:, col]
        valid = np.isfinite(values)
        if valid.all():
            continue
        if not valid.any():
            values[:] = 0.0
        elif valid.sum() == 1:
            values[:] = values[valid][0]
        else:
            values[:] = np.interp(frame_idx, frame_idx[valid], values[valid])
    return flat.reshape(sequence.shape).astype(np.float32)


def resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    sequence = validate_keypoint_sequence(sequence)
    if target_length <= 0:
        raise ValueError("target_length must be positive.")
    if sequence.shape[0] == target_length:
        return sequence.astype(np.float32, copy=True)
    if sequence.shape[0] == 0:
        return np.zeros((target_length, NUM_POSE_LANDMARKS, COORD_DIMS), dtype=np.float32)
    if sequence.shape[0] == 1:
        return np.repeat(sequence, target_length, axis=0).astype(np.float32)

    old_t = np.linspace(0.0, 1.0, sequence.shape[0], dtype=np.float32)
    new_t = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    flat = sequence.reshape(sequence.shape[0], -1)
    resized = np.empty((target_length, flat.shape[1]), dtype=np.float32)
    for col in range(flat.shape[1]):
        resized[:, col] = np.interp(new_t, old_t, flat[:, col])
    return resized.reshape(target_length, NUM_POSE_LANDMARKS, COORD_DIMS)


def normalize_keypoints(sequence: np.ndarray) -> np.ndarray:
    """Remove translation and body scale so pixels/background cannot leak in."""
    sequence = validate_keypoint_sequence(sequence)

    hip_center = 0.5 * (sequence[:, 23:24, :] + sequence[:, 24:25, :])
    shoulder_center = 0.5 * (sequence[:, 11:12, :] + sequence[:, 12:13, :])

    shoulder_width = np.linalg.norm(sequence[:, 11, :2] - sequence[:, 12, :2], axis=1)
    hip_width = np.linalg.norm(sequence[:, 23, :2] - sequence[:, 24, :2], axis=1)
    torso_height = np.linalg.norm(shoulder_center[:, 0, :2] - hip_center[:, 0, :2], axis=1)
    scale = np.maximum.reduce([shoulder_width, hip_width, torso_height])

    valid_scale = scale[np.isfinite(scale) & (scale > 1e-6)]
    fallback_scale = float(np.median(valid_scale)) if valid_scale.size else 1.0
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, fallback_scale)

    normalized = (sequence - hip_center) / scale[:, None, None]
    return normalized.astype(np.float32)


def compute_velocity(sequence: np.ndarray) -> np.ndarray:
    sequence = validate_keypoint_sequence(sequence)
    velocity = np.diff(sequence, axis=0, prepend=sequence[:1])
    return velocity.astype(np.float32)


def compute_joint_angles(sequence: np.ndarray, triplets: Iterable[tuple[int, int, int]] = ANGLE_TRIPLETS) -> np.ndarray:
    sequence = validate_keypoint_sequence(sequence)
    angles = []
    for first, mid, last in triplets:
        a = sequence[:, first, :] - sequence[:, mid, :]
        b = sequence[:, last, :] - sequence[:, mid, :]
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        denom = np.maximum(denom, 1e-6)
        cosine = np.sum(a * b, axis=1) / denom
        cosine = np.clip(cosine, -1.0, 1.0)
        angles.append(np.arccos(cosine) / math.pi)
    return np.stack(angles, axis=1).astype(np.float32)


def keypoint_variance_score(sequence: np.ndarray) -> float:
    sequence = validate_keypoint_sequence(sequence)
    return float(np.mean(np.var(sequence, axis=0)))


def build_feature_sequence(
    raw_sequence: np.ndarray,
    sequence_length: int,
    include_angles: bool = True,
) -> np.ndarray:
    """Build LSTM features from normalized pose, velocity, and temporal variance."""
    sequence = fill_missing_keypoints(raw_sequence)
    sequence = resample_sequence(sequence, sequence_length)
    sequence = normalize_keypoints(sequence)

    velocity = compute_velocity(sequence)
    variance = np.var(sequence, axis=0, keepdims=True).repeat(sequence.shape[0], axis=0)

    per_frame_speed = np.linalg.norm(velocity, axis=2).mean(axis=1, keepdims=True)
    overall_variance = np.full(
        (sequence.shape[0], 1),
        keypoint_variance_score(sequence),
        dtype=np.float32,
    )

    features = [
        sequence.reshape(sequence.shape[0], -1),
        velocity.reshape(sequence.shape[0], -1),
        variance.reshape(sequence.shape[0], -1),
        per_frame_speed.astype(np.float32),
        overall_variance,
    ]
    if include_angles:
        features.append(compute_joint_angles(sequence))
    return np.concatenate(features, axis=1).astype(np.float32)


def mirror_keypoints(sequence: np.ndarray) -> np.ndarray:
    sequence = validate_keypoint_sequence(sequence).copy()
    sequence[..., 0] = 1.0 - sequence[..., 0]
    for left, right in LEFT_RIGHT_LANDMARK_PAIRS:
        sequence[:, [left, right], :] = sequence[:, [right, left], :]
    return sequence


def augment_keypoints(
    sequence: np.ndarray,
    rng: np.random.Generator,
    jitter_std: float = 0.006,
) -> np.ndarray:
    """Label-preserving augmentations that do not introduce background cues."""
    sequence = validate_keypoint_sequence(sequence).copy()
    if sequence.shape[0] > 8:
        keep_ratio = float(rng.uniform(0.75, 1.0))
        crop_len = max(8, int(round(sequence.shape[0] * keep_ratio)))
        if crop_len < sequence.shape[0]:
            start = int(rng.integers(0, sequence.shape[0] - crop_len + 1))
            sequence = sequence[start : start + crop_len]

    if rng.random() < 0.5:
        sequence = mirror_keypoints(sequence)

    if rng.random() < 0.2:
        sequence = sequence[::-1].copy()

    if jitter_std > 0:
        jitter = rng.normal(0.0, jitter_std, size=sequence.shape).astype(np.float32)
        sequence = np.where(np.isfinite(sequence), sequence + jitter, sequence)
    return sequence.astype(np.float32)


def feature_dim(include_angles: bool = True) -> int:
    base = NUM_POSE_LANDMARKS * COORD_DIMS
    dim = base + base + base + 2
    if include_angles:
        dim += len(ANGLE_TRIPLETS)
    return dim
