from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CLASS_TO_LABEL, DEFAULT_SAMPLE_FPS, DEFAULT_SEQUENCE_LENGTH, LABEL_TO_CLASS
from .data import (
    CachedRecord,
    PoseSequenceDataset,
    discover_videos,
    save_split_manifest,
    split_records,
    summarize_split,
)
from .features import feature_dim
from .model import PoseLSTMClassifier
from .pose_extraction import cache_path_for_video, write_pose_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a posing vs normal behavior classifier from MediaPipe pose motion."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Folder containing posing/ and no_posing/.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".pose_cache"), help="Pose keypoint cache folder.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Where checkpoints and metrics are written.")
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-angles", action="store_true", help="Disable joint-angle features.")
    parser.add_argument("--no-augment", action="store_true", help="Disable train-time keypoint augmentation.")
    parser.add_argument("--no-group-split", action="store_true", help="Split clips randomly instead of by source video.")
    parser.add_argument("--refresh-cache", action="store_true", help="Re-extract MediaPipe keypoints.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(choice: str) -> torch.device:
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_cached_records(
    records,
    cache_dir: Path,
    sample_fps: float,
    refresh: bool,
) -> list[CachedRecord]:
    cached: list[CachedRecord] = []
    for record in tqdm(records, desc="Extracting MediaPipe pose"):
        cache_path = cache_path_for_video(record.path, cache_dir)
        write_pose_cache(
            video_path=record.path,
            cache_path=cache_path,
            label=record.label,
            class_dir=record.class_dir,
            group_id=record.group_id,
            sample_fps=sample_fps,
            refresh=refresh,
        )
        cached.append(
            CachedRecord(
                path=record.path,
                label=record.label,
                class_dir=record.class_dir,
                group_id=record.group_id,
                cache_path=cache_path,
            )
        )
    return cached


def class_weights(records: list[CachedRecord], device: torch.device) -> torch.Tensor:
    counts = torch.zeros(len(LABEL_TO_CLASS), dtype=torch.float32)
    for record in records:
        counts[record.label] += 1.0
    if torch.any(counts == 0):
        return torch.ones_like(counts, device=device)
    weights = counts.sum() / (len(counts) * counts)
    return weights.to(device)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    total = int(labels.numel())

    positive = CLASS_TO_LABEL["posing"]
    tp = ((predictions == positive) & (labels == positive)).sum().item()
    fp = ((predictions == positive) & (labels != positive)).sum().item()
    fn = ((predictions != positive) & (labels == positive)).sum().item()
    tn = ((predictions != positive) & (labels != positive)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": correct / max(total, 1),
        "precision_posing": precision,
        "recall_posing": recall,
        "f1_posing": f1,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return total_loss / max(len(loader.dataset), 1), compute_metrics(logits, labels)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(features)
        loss = criterion(logits, labels)

        total_loss += float(loss.item()) * labels.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    if not all_logits:
        return 0.0, {}
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return total_loss / max(len(loader.dataset), 1), compute_metrics(logits, labels)


def make_loader(dataset: PoseSequenceDataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    input_size: int,
    best_metrics: dict[str, float],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "sequence_length": args.sequence_length,
        "include_angles": not args.no_angles,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "bidirectional": True,
        "class_to_label": CLASS_TO_LABEL,
        "label_to_class": LABEL_TO_CLASS,
        "metrics": best_metrics,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    include_angles = not args.no_angles

    records = discover_videos(args.data_dir)
    splits = split_records(
        records,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        group_aware=not args.no_group_split,
    )

    print("Dataset split:")
    for name, items in splits.items():
        print(f"  {name}: {summarize_split(items)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.runs_dir / f"pose_lstm_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cached_splits = {
        name: prepare_cached_records(items, args.cache_dir, args.sample_fps, args.refresh_cache)
        for name, items in splits.items()
    }
    save_split_manifest(cached_splits, run_dir / "splits.json")

    train_dataset = PoseSequenceDataset(
        cached_splits["train"],
        sequence_length=args.sequence_length,
        include_angles=include_angles,
        augment=not args.no_augment,
        seed=args.seed,
    )
    val_dataset = PoseSequenceDataset(
        cached_splits["val"],
        sequence_length=args.sequence_length,
        include_angles=include_angles,
        augment=False,
        seed=args.seed,
    )
    test_dataset = PoseSequenceDataset(
        cached_splits["test"],
        sequence_length=args.sequence_length,
        include_angles=include_angles,
        augment=False,
        seed=args.seed,
    )

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, device)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, device)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, device)

    input_size = feature_dim(include_angles=include_angles)
    model = PoseLSTMClassifier(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights(cached_splits["train"], device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, args.patience // 3),
    )

    best_score = -1.0
    best_metrics: dict[str, float] = {}
    bad_epochs = 0
    best_path = run_dir / "best.pt"
    history = []

    print(f"Training on {device} with input_size={input_size}.")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        score = val_metrics.get("f1_posing", 0.0)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train": train_metrics,
            "val": val_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_metrics.get('accuracy', 0):.3f} "
            f"val_f1_posing={score:.3f}"
        )

        if score > best_score + 1e-4:
            best_score = score
            best_metrics = {"val_loss": val_loss, **val_metrics, "epoch": float(epoch)}
            save_checkpoint(best_path, model, args, input_size, best_metrics)
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"Early stopping after {bad_epochs} epochs without validation F1 improvement.")
            break

    metrics_path = run_dir / "metrics.json"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    summary = {
        "best_validation": best_metrics,
        "test": {"loss": test_loss, **test_metrics} if test_metrics else {},
        "history": history,
        "config": {
            "input_size": input_size,
            "sequence_length": args.sequence_length,
            "include_angles": include_angles,
            "class_to_label": CLASS_TO_LABEL,
            "label_to_class": LABEL_TO_CLASS,
            "split_summary": {name: summarize_split(items) for name, items in cached_splits.items()},
        },
    }
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Best checkpoint: {best_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Test metrics: {summary['test']}")


if __name__ == "__main__":
    main()
