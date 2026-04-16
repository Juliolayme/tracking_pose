from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import LABEL_TO_CLASS
from .features import build_feature_sequence, keypoint_variance_score
from .model import PoseLSTMClassifier


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[PoseLSTMClassifier, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PoseLSTMClassifier(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint.get("hidden_size", 128)),
        num_layers=int(checkpoint.get("num_layers", 2)),
        dropout=float(checkpoint.get("dropout", 0.3)),
        bidirectional=bool(checkpoint.get("bidirectional", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict_from_keypoints(
    model: PoseLSTMClassifier,
    checkpoint: dict[str, Any],
    keypoints: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    sequence_length = int(checkpoint["sequence_length"])
    include_angles = bool(checkpoint.get("include_angles", True))
    features = build_feature_sequence(keypoints, sequence_length=sequence_length, include_angles=include_angles)
    tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_label = int(np.argmax(probabilities))

    label_to_class = {int(key): value for key, value in checkpoint.get("label_to_class", LABEL_TO_CLASS).items()}
    probability_by_class = {
        label_to_class.get(index, str(index)): float(probabilities[index])
        for index in range(probabilities.shape[0])
    }
    label_name = label_to_class.get(pred_label, str(pred_label))
    return {
        "label": label_name,
        "confidence": float(probabilities[pred_label]),
        "probabilities": probability_by_class,
        "variance_score": keypoint_variance_score(
            build_feature_sequence(
                keypoints,
                sequence_length=sequence_length,
                include_angles=False,
            )[:, : 33 * 3].reshape(sequence_length, 33, 3)
        ),
    }


def device_from_arg(choice: str) -> torch.device:
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
