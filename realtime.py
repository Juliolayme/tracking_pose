from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import NUM_POSE_LANDMARKS
from .inference import device_from_arg, load_checkpoint_model, predict_from_keypoints
from .pose_extraction import _require_mediapipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time POSING/NORMAL prediction from a webcam or video stream.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", default="0", help="Camera index, video path, or stream URL.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--warmup-frames", type=int, default=16)
    parser.add_argument("--predict-every", type=int, default=3)
    parser.add_argument("--draw-pose", action="store_true")
    return parser.parse_args()


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def frame_to_keypoints(frame: np.ndarray, pose: Any) -> tuple[np.ndarray, Any]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = pose.process(rgb)
    if not result.pose_landmarks:
        return np.full((NUM_POSE_LANDMARKS, 3), np.nan, dtype=np.float32), result
    keypoints = np.array(
        [[landmark.x, landmark.y, landmark.z] for landmark in result.pose_landmarks.landmark],
        dtype=np.float32,
    )
    return keypoints, result


def draw_prediction(frame: np.ndarray, prediction: dict[str, Any], ready: bool) -> None:
    if not ready:
        label = "COLLECTING MOTION"
        confidence = 0.0
        color = (80, 180, 255)
    else:
        label = str(prediction["label"])
        confidence = float(prediction["confidence"])
        color = (0, 180, 0) if label == "NORMAL" else (0, 120, 255)

    text = f"{label}  {confidence:.2f}" if ready else label
    cv2.rectangle(frame, (12, 12), (410, 82), (0, 0, 0), thickness=-1)
    cv2.putText(frame, text, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    if ready:
        cv2.putText(
            frame,
            f"variance={prediction['variance_score']:.5f}",
            (24, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    device = device_from_arg(args.device)
    model, checkpoint = load_checkpoint_model(args.checkpoint, device)
    sequence_length = int(checkpoint["sequence_length"])

    mp = _require_mediapipe()
    capture = cv2.VideoCapture(parse_source(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    keypoint_buffer: deque[np.ndarray] = deque(maxlen=sequence_length)
    prediction: dict[str, Any] = {"label": "NORMAL", "confidence": 0.0, "variance_score": 0.0}
    frame_count = 0

    pose_solution = mp.solutions.pose
    drawing_utils = mp.solutions.drawing_utils
    with pose_solution.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            keypoints, result = frame_to_keypoints(frame, pose)
            keypoint_buffer.append(keypoints)
            ready = len(keypoint_buffer) >= max(2, min(args.warmup_frames, sequence_length))

            if ready and frame_count % max(1, args.predict_every) == 0:
                sequence = np.stack(list(keypoint_buffer), axis=0)
                prediction = predict_from_keypoints(model, checkpoint, sequence, device)

            if args.draw_pose and result.pose_landmarks:
                drawing_utils.draw_landmarks(frame, result.pose_landmarks, pose_solution.POSE_CONNECTIONS)
            draw_prediction(frame, prediction, ready)
            cv2.imshow("Pose Behavior Classifier", frame)

            frame_count += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
