from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_SAMPLE_FPS
from .inference import device_from_arg, load_checkpoint_model, predict_from_keypoints
from .pose_extraction import extract_pose_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict POSING or NORMAL for one video.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt from training.")
    parser.add_argument("--video", type=Path, required=True, help="Video file to classify.")
    parser.add_argument("--sample-fps", type=float, default=None, help="Override extraction FPS.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = device_from_arg(args.device)
    model, checkpoint = load_checkpoint_model(args.checkpoint, device)
    sample_fps = args.sample_fps
    if sample_fps is None:
        sample_fps = float(checkpoint.get("args", {}).get("sample_fps", DEFAULT_SAMPLE_FPS))

    keypoints, metadata = extract_pose_sequence(args.video, sample_fps=sample_fps)
    prediction = predict_from_keypoints(model, checkpoint, keypoints, device)

    print(f"Video: {args.video}")
    print(f"Prediction: {prediction['label']}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    print("Probabilities:")
    for label, probability in prediction["probabilities"].items():
        print(f"  {label}: {probability:.4f}")
    print(f"Temporal variance score: {prediction['variance_score']:.6f}")
    print(f"Missing pose frames: {metadata['missing_pose_frames']}/{metadata['sampled_frames']}")


if __name__ == "__main__":
    main()
