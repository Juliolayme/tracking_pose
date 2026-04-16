from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import CLASS_TO_LABEL, LABEL_TO_CLASS, VIDEO_EXTENSIONS
from .features import augment_keypoints, build_feature_sequence


@dataclass(frozen=True)
class VideoRecord:
    path: Path
    label: int
    class_dir: str
    group_id: str


@dataclass(frozen=True)
class CachedRecord(VideoRecord):
    cache_path: Path


def source_group_id(path: Path) -> str:
    """Group clips from the same source video to avoid validation leakage."""
    stem = re.sub(r"\s+", " ", path.stem).strip()
    match = re.search(r"(?i)(?:[_\s-]+start\d+(?:\.\d+)?)", stem)
    source = stem[: match.start()] if match else stem
    source = re.sub(r"[^0-9a-zA-Z]+", "_", source).strip("_").lower()
    return source or path.stem.lower()


def dataset_root(data_dir: Path) -> Path:
    data_dir = Path(data_dir)
    nested = data_dir / "dataset"
    if (nested / "posing").is_dir() and (nested / "no_posing").is_dir():
        return nested
    return data_dir


def discover_videos(data_dir: Path) -> list[VideoRecord]:
    root = dataset_root(Path(data_dir))
    records: list[VideoRecord] = []
    for class_dir, label in CLASS_TO_LABEL.items():
        folder = root / class_dir
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing class folder: {folder}. Expected posing/ and no_posing/."
            )
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                records.append(
                    VideoRecord(
                        path=path,
                        label=label,
                        class_dir=class_dir,
                        group_id=source_group_id(path),
                    )
                )
    if not records:
        raise FileNotFoundError(f"No videos found under {root}.")
    return records


def label_counts(records: Iterable[VideoRecord]) -> dict[str, int]:
    counts = Counter(record.label for record in records)
    return {LABEL_TO_CLASS[label]: counts.get(label, 0) for label in sorted(LABEL_TO_CLASS)}


def summarize_split(records: Iterable[VideoRecord]) -> dict[str, object]:
    records = list(records)
    return {
        "clips": len(records),
        "groups": len({record.group_id for record in records}),
        "labels": label_counts(records),
    }


def _group_records(records: list[VideoRecord]) -> list[list[VideoRecord]]:
    groups: dict[str, list[VideoRecord]] = defaultdict(list)
    for record in records:
        groups[record.group_id].append(record)
    return list(groups.values())


def _select_groups_for_fraction(
    remaining_groups: list[list[VideoRecord]],
    target_fraction: float,
    total_counts: np.ndarray,
    rng: random.Random,
) -> list[list[VideoRecord]]:
    if target_fraction <= 0 or not remaining_groups:
        return []

    total_clips = int(total_counts.sum())
    target_clips = max(1, int(round(total_clips * target_fraction)))
    target_counts = total_counts * target_fraction
    selected: list[list[VideoRecord]] = []
    current_counts = np.zeros_like(total_counts, dtype=np.float32)
    current_clips = 0

    while remaining_groups and current_clips < target_clips:
        rng.shuffle(remaining_groups)

        def score(group: list[VideoRecord]) -> float:
            group_counts = np.zeros_like(total_counts, dtype=np.float32)
            for record in group:
                group_counts[record.label] += 1.0
            after_counts = current_counts + group_counts
            after_clips = current_clips + len(group)
            class_loss = float(
                np.sum(((after_counts - target_counts) / np.maximum(total_counts, 1.0)) ** 2)
            )
            size_loss = ((after_clips - target_clips) / max(total_clips, 1)) ** 2
            return class_loss + size_loss

        best_group = min(remaining_groups, key=score)
        remaining_groups.remove(best_group)
        selected.append(best_group)
        for record in best_group:
            current_counts[record.label] += 1.0
        current_clips += len(best_group)

    return selected


def split_records(
    records: list[VideoRecord],
    val_fraction: float = 0.2,
    test_fraction: float = 0.1,
    seed: int = 42,
    group_aware: bool = True,
) -> dict[str, list[VideoRecord]]:
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 0.8:
        raise ValueError("Use non-negative split fractions with train fraction >= 0.2.")

    rng = random.Random(seed)
    records = list(records)
    if not group_aware:
        return _stratified_clip_split(records, val_fraction, test_fraction, rng)

    groups = _group_records(records)
    if len(groups) < 3:
        return _stratified_clip_split(records, val_fraction, test_fraction, rng)

    rng.shuffle(groups)
    total_counts = np.zeros(len(LABEL_TO_CLASS), dtype=np.float32)
    for record in records:
        total_counts[record.label] += 1.0

    test_groups = _select_groups_for_fraction(groups, test_fraction, total_counts, rng)
    val_groups = _select_groups_for_fraction(groups, val_fraction, total_counts, rng)
    train_groups = groups

    splits = {
        "train": [record for group in train_groups for record in group],
        "val": [record for group in val_groups for record in group],
        "test": [record for group in test_groups for record in group],
    }
    if not splits["train"] or not splits["val"]:
        return _stratified_clip_split(records, val_fraction, test_fraction, rng)
    return {name: sorted(items, key=lambda item: str(item.path)) for name, items in splits.items()}


def _stratified_clip_split(
    records: list[VideoRecord],
    val_fraction: float,
    test_fraction: float,
    rng: random.Random,
) -> dict[str, list[VideoRecord]]:
    by_label: dict[int, list[VideoRecord]] = defaultdict(list)
    for record in records:
        by_label[record.label].append(record)

    splits = {"train": [], "val": [], "test": []}
    for label_records in by_label.values():
        rng.shuffle(label_records)
        n = len(label_records)
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        if n >= 3 and test_fraction > 0:
            n_test = max(1, n_test)
        if n - n_test >= 2 and val_fraction > 0:
            n_val = max(1, n_val)
        splits["test"].extend(label_records[:n_test])
        splits["val"].extend(label_records[n_test : n_test + n_val])
        splits["train"].extend(label_records[n_test + n_val :])
    return {name: sorted(items, key=lambda item: str(item.path)) for name, items in splits.items()}


def save_split_manifest(splits: dict[str, list[VideoRecord]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        name: [
            {
                **asdict(record),
                "path": str(record.path),
                "cache_path": str(record.cache_path) if isinstance(record, CachedRecord) else None,
            }
            for record in split_records
        ]
        for name, split_records in splits.items()
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class PoseSequenceDataset(Dataset):
    def __init__(
        self,
        records: list[CachedRecord],
        sequence_length: int,
        include_angles: bool,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.records = list(records)
        self.sequence_length = sequence_length
        self.include_angles = include_angles
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        with np.load(record.cache_path, allow_pickle=False) as data:
            keypoints = data["keypoints"].astype(np.float32)

        if self.augment:
            keypoints = augment_keypoints(keypoints, self.rng)

        features = build_feature_sequence(
            keypoints,
            sequence_length=self.sequence_length,
            include_angles=self.include_angles,
        )
        return (
            torch.from_numpy(features),
            torch.tensor(record.label, dtype=torch.long),
        )
