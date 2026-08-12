from __future__ import annotations

import random
import argparse
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv
import numpy as np
import torch
from torch import nn
from torch.utlis.data import Dataloader, Dataset

try:
    import mediapipe as mp
except ImportError:
    mp = None

project_name = "CubeMoveAI"
dataset_dir = Path("dataset")
cache_dir = Path("mediapipe_cache")
model_path = Path("cube_move_ai_model_v2.pt")
video_extensions = {".mp4",".avi",".mov",".mkv"}
supported_labels = {"R", "L", "U", "D", "F", "B", "Rp", "Lp", "Up", "Dp", "Fp", "Bp"}

max_hands = 2
landmark_count = 21
hand_feature_size = 1 + 3 + (landmark_count * 3)
frame_feature_size = hand_feature_size * max_hands
model_input_size = frame_feature_size * 2

default_sequence_length = 32
min_recording_frame = 12
feature_cache_version = 2
min_active_sequence_frames = 12
active_motion_threshold = 0.01

train_epochs = 50
train_batch_size = 8
learning_rate = 0.001
weight_decay = 1e-4
validation_ratio = 0.2
early_stopping_patience = 10
random_seed = 42 
graident_clip_norm = 1.0

augment_noise_std = 0.01
augment_scale_min = 0.97
augment_scale_max = 1.03
augment_frame_drop_probability = 0.4
augment_max_dropped_frames = 3

camera_width = 1280
camera_height = 720
target_fps = 30
default_camera_index = -1 
camera_search_order = [1, 0, 2, 3, 4, 5]
countdown_seconds = 3
window_name = "CubeMoveAI"

mp_model_complexity = 1
min_detection_confidence = 0.5
min_tracking_confidence = 0.5

class ClipExample:
    label: str
    video_path: Path
    raw_sequence: np.ndarray

class SequenceDataset(Dataset):
    def __init__(
            self,
            examples: list[ClipExample],
            label_to_index: dict[str, int],
            feature_mean: np.ndarray,
            feature_std: np.ndarray,
            sequence_length: int,
            augment: bool
    ) -> None:
        self.examples = examples
        self.label_to_index = label_to_index
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.sequence_length = sequence_length
        self.augment = augment

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.example[index]
        features = prepare_sequence_for_model(
            example.raw_sequence,
            sequence_length=self.sequence_length,
            augment=self.augment
        )

        features = normalize_features(features, self.feature_mean, self.feature_std)
        label = self.label_to_index[example.label]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class MoveClassifier(nn.Module):
    def __init__(self, input_size: int, num_class: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.encoder = nn.GRU(input_size=input_size, hidden_size=96, num_layers=2, batch_first=True, dropout=0.25, bidirectional=True)

        self.classifier = nn.Sequential(
            nn.Linear(96*2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_class)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.input_norm(inputs)
        encoded, _ = self.encoder(inputs)
        mean_pool = encoded.mean(dim=1)
        max_pool = encoded.max(dim=1).values
        merged = torch.cat([mean_pool, max_pool], dim=1)
        return self.classifier(merged)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_label_list(label_text: Optional[str]) -> Optional[list[str]]:
    if not label_text:
        return None

    labels = [item.strip() for item in label_text.split(",") if item.strip()]
    invalid =[label for label in labels if label not in supported_labels]
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid labels: {",".join(invalid)}.")
    return labels

def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CURA was requested, but PyTorch cannot see a GPU.")

    return torch.device(device_name)

def require_mediapipe() -> None:
    if mp is None:
        raise RuntimeError("Mediapipe is needed!")

def create_hands_tracker(static_image_mode: bool) -> object:
    require_mediapipe()
    return mp.solutions.hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_hands,
        model_complexity=mp_model_complexity,
        max_num_hands=max_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )

def list_video_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    files = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in video_extensions]
    return sorted(files)

def discover_available_labels(dataset_dir: Path, requested_labels: Optional[list[str]]) -> list[str]:
    labels = requested_labels or supported_labels
    available = [label for label in labels if list_video_files(dataset_dir / label)]
    return available

def cache_path_for_video(video_path: Path) -> Path:
    label_dir = cache_dir / video_path.parent.name
    label_dir.mkdir(parents=True, exist_ok=True)
    return label_dir / f"{video_path.stem}_mpcache_v{feature_cache_version}.npz"

def load_cached_features(video_path: Path) -> Optional[np.ndarray]:
    cache_path = cache_path_for_video(video_path)
    return cache_path if cache_path.exists() else None

def load_cached_sequence(video_path: Path) -> Optional[np.ndarray]:
    cache_path = cache_path_for_video(video_path)
    if not cache_path.exists():
        return None

    source_stat = video_path.stat()
    try:
        with np.load(cache_path, allow_pickle=False) as cache_data:
            cache_stat = cache_path.stat()
            cached_size = int(cache_data["source_size"])
            cache_mtime_ns = int(cache_data["source_mtime_ns"])
            cached_version = int(cache_data["cache_version"])
            if cached_version != feature_cache_version or cached_size != source_stat.st_size or cache_mtime_ns != source_stat.st_mtime_ns:
                return None
            sequence = cache_data["sequence"].astype(np.float32)
    except (OSError, KeyError, ValueError):
        return None
    return sequence

def save_cached_sequence(video_path: Path, sequence: np.ndarray) -> None:
    cache_path = cache_path_for_video(video_path)
    cache_path.parent.mkdir(parents=True, exists_ok=True)
    source_stat = video_path.stat()
    np.savez_compressed(
        cache_path,
        cache_version=feature_cache_version,
        source_size=source_stat.st_size,
        source_mtime_ns=source_stat.st_mtime_ns,
        sequence=sequence.astype(np.float32)
    )

def hand_landmarks_to_features(hand_landmarks: object) -> tuple[float, np.ndarray]:
    points = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark], dtype=np.float32)
    wrist = points[0]
    centered_points = points - wrist
    scale = np.linalg.norm(centered_points[:, :2], axis=1).max()
    if scale > 1e-6:
        centered_points = centered_points / scale

    feature = np.concatenate(
        [
            np.array([1.0], dtype=np.float32),
            wrist.astype(np.float32),
            centered_points.reshape(-1).astype(np.float32)
        ]
    )
    return float(wrist[0]), feature

def extract_frame_feature(result: object) -> np.ndarray:
    frame_feature = np.zeros(frame_feature_size, dtype=np.float32)
    if not getattr(result, "multi_hand_landmarks", None):
        return frame_feature
    ordered_hands = []
    for hand_landmark in result.multi_hand_landmarks:
        wrist_x, hand_feature = hand_landmarks_to_features(hand_landmark)
        ordered_hands.append((wrist_x, hand_feature))
    ordered_hands.sort(key=lambda x: x[0])

    for hand_index, (_, hand_feature) in enumerate(ordered_hands[:max_hands]):
        start = hand_index * hand_feature_size
        end = start + hand_feature_size
        frame_feature[start:end] = hand_feature
    return frame_feature

def sequence_has_landmarks(sequence: np.ndarray) -> bool:
    if sequence.size == 0:
        return False
    presence_sum = sequence[:, 0].sum()
    return bool(np.count_nonzero(presence_sum > 0.0))

def extract_video_feature(
        video_path: Path,
        hands_tracker: object,
        refresh_cache: bool,
) -> tuple[Optional[np.ndarray], bool]:
    if not refresh_cache:
        cached_sequence = load_cached_sequence(video_path)
        if cached_sequence is not None:
            return cached_sequence, True

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Failed to open video: {video_path}")
        return None, False

    frame_feature:list[np.ndarray] = []

    try:
        while True:
            frame_ok, frame= capture.read()
            if not frame_ok:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_tracker.process(rgb_frame)
            frame_feature.append(extract_frame_feature(results))

    finally:
        capture.release()

    if not frame_feature:
        return None, False

    sequence = np.asarray(frame_feature, dtype=np.float32)
    save_cached_sequence(video_path, sequence)
    return sequence, False

def trim_sequence_to_motion(sequence: np.ndarray, motion_threshold: float) -> np.ndarray:
    if len(sequence) < min_active_sequence_frames:
        return sequence
    motion = np.linalg.norm(sequence[1:] - sequence[:-1], axis=1)
    if motion.size == 0:
        return sequence

    start = 
    
