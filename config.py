from pathlib import Path

CLASS_TO_LABEL = {
    "no_posing": 0,
    "posing": 1,
}

LABEL_TO_CLASS = {
    0: "NORMAL",
    1: "POSING",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
}

NUM_POSE_LANDMARKS = 33
COORD_DIMS = 3
DEFAULT_SEQUENCE_LENGTH = 64
DEFAULT_SAMPLE_FPS = 15.0
DEFAULT_DATA_DIR = Path(".")
