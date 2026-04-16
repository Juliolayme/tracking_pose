from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_SAMPLE_FPS
from .data import discover_videos
from .train import prepare_cached_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-extract MediaPipe pose keypoints into a cache.")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--cache-dir", type=Path, default=Path(".pose_cache"))
    parser.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = discover_videos(args.data_dir)
    cached = prepare_cached_records(records, args.cache_dir, args.sample_fps, args.refresh_cache)
    print(f"Cached {len(cached)} pose sequences in {args.cache_dir}")


if __name__ == "__main__":
    main()
