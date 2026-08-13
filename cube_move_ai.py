import argparse
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
from matplotlib.style import available
import numpy as np
import torch
from torch import mode, nn
from torch.utils.data import DataLoader, Dataset

import mediapipe as mp

project_name = "CubeMoveAI"
dataset_dir = Path("dataset")
cache_dir = Path("mediapipe_cache")
model_path = Path("cube_move_ai_model.pth")
video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
supported_labels = ["R", "L", "U", "D", "F", "B", "Rp", "Lp", "Up", "Dp", "Fp", "Bp"]

max_hands = 2
landmark_count = 21
hand_feature_size = 1 + 3 + (landmark_count * 3)  # handedness + visibility + (x, y, z) for each landmark
frame_feature_size = hand_feature_size * max_hands
model_input_size = frame_feature_size * 2  # Two frames as input

default_sequence_length = 32
min_recording_frame = 12
feature_cache_version = 2
min_active_sequence_frames = 12
active_motion_padding = 4
min_motion_threshold = 0.01

train_epochs = 50
train_batch_size = 8
learning_rate = 0.001
weight_decay = 1e-4
validation_ratio = 0.2
early_stopping_patience = 10
random_seed = 42
gradient_clip_norm = 1.0

augment_noise_std = 0.01
augment_scale_min = 0.97
augment_scale_max =1.03
augment_frame_drop_probability = 0.4
augment_max_dropped_frames = 3

camera_width = 1280
camera_height = 720
target_fps = 30
default_camera_index = -1
camera_search_order = [1, 0, 2, 3, 4, 5]
countdown_seconds = 3
window_name = "CubeMoveA"

mp_model_complexity = 1
min_detection_confidence = 0.5
min_tracking_confidence = 0.5

@dataclass
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
            augment: bool,
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
        example =  self.examples[index]
        features = prepare_sequence_for_model(
            example.raw_sequence,
            sequence_length = self.sequence_length,
            augment = self.augment
        )
        features = normalize_features(features, self.feature_mean, self.feature_std)
        label = self.label_to_index[example.label]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class MoveClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.encoder = nn.GRU(
            input_size = input_size,
            hidden_size = 96,
            num_layers = 2,
            batch_first = True,
            dropout = 0.25,
            bidirectional = True
        )

        self.classifier = nn.Sequential(
            nn.Linear(96*4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.input_norm(inputs)
        encoded, _ = self.encoder(inputs)
        mean_pool = encoded.mean(dim=1)
        max_pool = encoded.max(dim=1).values
        merged = torch.cat([mean_pool, max_pool], dim=1)
        return self.classifier(merged)

def set_Seed(seed : int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_label_list(label_text: Optional[str]) -> Optional[list[str]]:
    if not label_text:
        return None

    labels = [item.strip() for item in label_text.split(",") if item.strip()]
    invalid = [label for label in labels if label not in supported_labels]
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid labels: {', '.join(invalid)}")
    return labels

def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot see a GPU.")

    return torch.device(device_name)

def require_mediapipe() -> None:
    if mp is None:
        raise RuntimeError("Mediapipe is needed! ")

def create_hands_tracker(static_image_mode: bool) -> object:
    require_mediapipe()
    return mp.solutions.hands.Hands(
        static_image_mode = static_image_mode,
        max_num_hands = max_hands,
        model_complexity = mp_model_complexity,
        min_detection_confidence = min_detection_confidence,
        min_tracking_confidence = min_tracking_confidence
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

def load_cached_sequence(video_path: Path) -> Optional[np.ndarray]:
    cache_path = cache_path_for_video(video_path)
    return cache_path if cache_path.exists() else None

def load_cached_sequence(video_path: Path) -> Optional[np.ndarray]:
    cache_path = cache_path_for_video(video_path)
    if not cache_path.exists():
        return None

    source_stat = video_path.stat()
    try:
        with np.load(cache_path, allow_pickle = False) as cache_data:
            cache_stat = cache_path.stat()
            cached_size = int(cache_data["source_size"])
            cached_mtime_ns = int(cache_data["source_mtime_ns"])
            cached_version = int(cache_data["cache_version"])
            if cached_version != feature_cache_version or cached_size != source_stat.st_size or cached_mtime_ns != source_stat.st_mtime_ns:
                return None
            sequence = cache_data["sequence"].astype(np.float32)
    except (OSError, KeyError, ValueError):
        return None

    return sequence

def save_cached_sequence(video_path: Path, sequence: np.ndarray) -> None:
    cache_path = cache_path_for_video(video_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    source_stat = video_path.stat()
    np.savez_compressed(
        cache_path,
        cache_version = feature_cache_version,
        source_size = source_stat.st_size,
        source_mtime_ns = source_stat.st_mtime_ns,
        sequence = sequence.astype(np.float32)
    )

def hand_landmarks_to_feature(hand_landmarks: object) -> tuple[float, np.ndarray]:
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
    for hand_landmarks in result.multi_hand_landmarks:
        wrist_x, hand_feature = hand_landmarks_to_feature(hand_landmarks)
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
    presence_sum = sequence[:,0] + sequence[:, hand_feature_size]
    return bool(np.count_nonzero(presence_sum > 0.0))

def extract_video_features(
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

    frame_features:list[np.ndarray] = []

    try:
        while True:
            frame_ok, frame = capture.read()
            if not frame_ok:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_tracker.process(rgb_frame)
            frame_features.append(extract_frame_feature(results))
    finally:
        capture.release()

    if not frame_features:
        return None, False

    sequence = np.asarray(frame_features, dtype=np.float32)
    save_cached_sequence(video_path, sequence)
    return sequence, False

def trim_sequence_to_motion(sequence: np.ndarray) -> np.ndarray:
    if len(sequence) <= min_active_sequence_frames:
        return sequence
    motion = np.linalg.norm(sequence[1:] - sequence[:-1], axis=1)
    if motion.size == 0:
        return sequence

    threshold = max(min_motion_threshold, float(motion.max())*0.25)
    active_indices = np.flatnonzero(motion >= threshold)
    if active_indices.size == 0:
        return sequence

    start = max(int(active_indices[0]) - active_motion_padding, 0)
    end = min(int(active_indices[-1]) + active_motion_padding + 2, len(sequence))
    trimmed = sequence[start:end]
    if len(trimmed) < min_active_sequence_frames:
        return sequence
    return trimmed

def maybe_drop_frames(sequence: np.ndarray, sequence_length: int) -> np.ndarray:
    if len(sequence) <= sequence_length + 2:
        return sequence
    if random.random() >= augment_frame_drop_probability:
        return sequence

    max_drop = min(augment_max_dropped_frames, len(sequence) - sequence_length)
    if max_drop <= 0:
        return sequence

    drop_count = random.randint(1, max_drop)
    drop_indices = sorted(random.sample(range(len(sequence)), drop_count))
    mask = np.ones(len(sequence), dtype=bool)
    mask[drop_indices] = False
    return sequence[mask]

def apply_coordinate_jitter(sequence: np.ndarray) -> np.ndarray:
    jittered = sequence.copy()
    scale = np.random.uniform(augment_scale_min, augment_scale_max)
    noise = np.random.normal(0.0, augment_noise_std, size=jittered.shape).astype(np.float32)

    for hand_index in range(max_hands):
        start = hand_index * hand_feature_size
        wrist_slice = slice(start + 1, start + 4)
        landmark_slice = slice(start + 4, start + hand_feature_size)
        jittered[:, wrist_slice] *= scale
        jittered[:, landmark_slice] *= scale
        jittered[:, wrist_slice] += noise[:, wrist_slice]
        jittered[:, landmark_slice] += noise[:, landmark_slice]

    return jittered

def resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    if len(sequence) == 0:
        return np.zeros((target_length, frame_feature_size), dtype=np.float32)
    if len(sequence) == 1:
        return np.repeat(sequence, target_length, axis=0).astype(np.float32)

    positions = np.linspace(0, len(sequence) - 1, num=target_length)
    left_indices = np.floor(positions).astype(int)
    right_indices = np.ceil(positions).astype(int)
    blend = (positions - left_indices).astype(np.float32).reshape(-1, 1)
    resampled = sequence[left_indices] * (1.0 - blend) + sequence[right_indices] * blend
    return resampled.astype(np.float32)

def prepare_sequence_for_model(
    raw_sequence: np.ndarray,
    sequence_length: int,
    augment: bool,
) -> np.ndarray:
    if raw_sequence.ndim != 2 or raw_sequence.shape[1] != frame_feature_size:
        raise ValueError(f"Invalid raw_sequence shape: {raw_sequence.shape}, expected (N, {frame_feature_size})")

    sequence = raw_sequence.astype(np.float32, copy = True)
    sequence = trim_sequence_to_motion(sequence)
    if augment:
        sequence = maybe_drop_frames(sequence, sequence_length=sequence_length)
        sequence = apply_coordinate_jitter(sequence)
    base_sequence = resample_sequence(sequence, sequence_length)
    motion_sequence = np.diff(base_sequence, axis=0, prepend=base_sequence[:1])
    combined = np.concatenate([base_sequence, motion_sequence], axis = 1)
    return combined.astype(np.float32)

def normalize_features(features: np.ndarray, feature_mean: np.ndarray, feature_std: np.ndarray) -> np.ndarray:
    return ((features - feature_mean) / feature_std).astype(np.float32)

def compute_feature_stats(examples: list[ClipExample], sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    prepare = [
        prepare_sequence_for_model(example.raw_sequence, sequence_length=sequence_length, augment=False)
        for example in examples
    ]

    stacked = np.concatenate(prepare, axis=0)
    feature_mean = stacked.mean(axis=0).astype(np.float32)
    feature_std = stacked.std(axis=0).astype(np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)
    return feature_mean, feature_std

def split_examples_stratified(examples: list[ClipExample],validation_ratio: float) -> tuple[list[ClipExample], list[ClipExample]]:
    grouped: dict[str, list[ClipExample]] = defaultdict(list)
    for example in examples:
        grouped[example.label].append(example)

    training_examples: list[ClipExample] = []
    validation_examples: list[ClipExample] = []

    for label, label_examples in grouped.items():
        if len(label_examples) < 2:
            raise ValueError(f"Not enough examples for label '{label}' to split into training and validation sets.")

        random.shuffle(label_examples)
        split_index = int(round(len(label_examples) * (1 - validation_ratio)))
        training_examples.extend(label_examples[:split_index])
        validation_examples.extend(label_examples[split_index:])

    random.shuffle(training_examples)
    random.shuffle(validation_examples)
    return training_examples, validation_examples

def compute_class_weights(examples: list[ClipExample], labels: list[str]) -> torch.Tensor:
    counts = Counter(example.label for example in examples)
    total_count = sum(counts.values())
    weights = []
    for label in labels:
        label_count = counts[label]
        weights.append(total_count / (len(labels) * label_count))
    return torch.tensor(weights, dtype=torch.float32)

def save_checkpoint(mode_path: Path, model: MoveClassifier, labels: list[str], feature_mean: np.ndarray, feature_std: np.ndarray, sequence_length: int, validation_accuracy: float) -> None:
    mode_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "labels": labels,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "sequence_length": sequence_length,
        "validation_accuracy": validation_accuracy
    }

    torch.save(checkpoint, mode_path)

def load_model_bundle(model_path: Path, device: torch.device) -> tuple[MoveClassifier, list[str], np.ndarray, np.ndarray, int]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    labels = list(checkpoint["labels"])
    input_size = int(checkpoint.get("input_size", model_input_size))
    sequence_length = int(checkpoint.get("sequence_length", default_sequence_length))
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)

    model = MoveClassifier(input_size=input_size, num_classes=len(labels))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, labels, feature_mean, feature_std, sequence_length

def predict_move(model: MoveClassifier, raw_sequence:np.ndarray, labels: list[str], feature_mean: np.ndarray, feature_std: np.ndarray, sequence_length: int, device) -> tuple[str, float, list[tuple[str,float]]]:
    prepared = prepare_sequence_for_model(raw_sequence, sequence_length=sequence_length, augment=False)
    prepared = normalize_features(prepared, feature_mean, feature_std)
    inputs = torch.from_numpy(prepared).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    best_index = int(np.argmax(probabilities))
    ranked = sorted(((label, float(probabilities[index])) for index, label in enumerate(labels)), key=lambda item: item[1], reverse=True)
    return labels[best_index], float(probabilities[best_index]), ranked

def run_epoch(model: MoveClassifier, data_loader: DataLoader, criterion: nn.Module, optimizer: Optional[torch.optim.Optimizer], device:torch.device) -> tuple[float, float]:
    training = optimizer is not None
    if training: 
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, targets)

            if training: 
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

        predictions = logits.argmax(dim=1)
        total_loss += float(loss.item()) * len(inputs)
        total_correct += int((predictions == targets).sum().item())
        total_examples += len(inputs)

    average_loss = total_loss / max(total_examples, 1)
    accuracy = total_correct / max(total_examples, 1)
    return average_loss, accuracy

def load_example(dataset_dir: Path, labels: list[str], refresh_cache: bool) -> list[ClipExample]:
    example: list[ClipExample] = []
    print("Extracting features from video files...")
    with create_hands_tracker(static_image_mode=True) as hands_tracker:
        for label in labels: 
            video_paths = list_video_files(dataset_dir / label)
            print(f"\n{label}: {len(video_paths)} video(s) found.")

            for video_path in video_paths:
                sequence, used_cache = extract_video_features(video_path, hands_tracker, refresh_cache=refresh_cache)
                if sequence is None:
                    continue
                if not sequence_has_landmarks(sequence):
                    print(f"Warning: No hand landmarks detected in video: {video_path}")
                    continue
                detected_frames =int(np.count_nonzero(sequence[:, 0] + sequence[:, hand_feature_size]))
                source_text = int(np.count_nonzero(sequence[:,0] + sequence[:, hand_feature_size]))
                print(f"  Loaded {video_path.name} "
                    f"({detected_frames}/{len(sequence)} frames with landmarks, {source_text})")
                example.append(ClipExample(label=label, video_path=video_path, raw_sequence=sequence))
    return example

def summarize_label_counts(examples: Iterable[ClipExample]) -> str:
    counts = Counter(example.label for example in examples)
    return ", ".join(f"{label}: {count}" for label, count in counts.items())

def command_train(args: argparse.Namespace) -> int:
    require_mediapipe()
    set_Seed(args.seed)

    labels = discover_available_labels(args.dataset, args.labels)
    if len(labels) < 2:
        print("Error: At least two labels with video examples are required for training.")
        return 1

    print(f"Training labels: {', '.join(labels)}")
    examples = load_example(args.dataset, labels, refresh_cache=args.refresh_cache)
    if len(examples) < len(labels) * 2:
        print("Error: Not enough video examples for training. Each label should have at least two videos.")
        return 1

    print(f"\nLoaded {len(examples)} video examples. Label counts: {summarize_label_counts(examples)}")

    try:
        train_examples, validation_examples = split_examples_stratified(examples, validation_ratio=args.validation_ratio)
    except ValueError as e:
        print(f"Dataset split error: {e}")
        return 1

    print(f"Training examples: {len(train_examples)}, Validation examples: {len(validation_examples)}")
    feature_mean, feature_std = compute_feature_stats(train_examples, sequence_length=args.sequence_length)

    label_to_index = {label: index for index, label in enumerate(labels)}
    train_dataset = SequenceDataset(train_examples, label_to_index=label_to_index, feature_mean=feature_mean, feature_std=feature_std, sequence_length=args.sequence_length, augment=True)
    validation_dataset = SequenceDataset(validation_examples, label_to_index=label_to_index, feature_mean=feature_mean, feature_std=feature_std, sequence_length=args.sequence_length, augment=False)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

    device = choose_device(args.device)
    model = MoveClassifier(input_size=model_input_size, num_classes=len(labels)).to(device)
    class_weights = compute_class_weights(train_examples, labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5)

    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0 

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device)
        validation_loss, validation_accuracy = run_epoch(model, validation_loader, criterion, optimizer=None, device=device)
        scheduler.step(validation_accuracy)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | train acc {train_accuracy:.3f} | "
            f"val loss {validation_loss:.4f} | val acc {validation_accuracy:.3f}"
        )

        improved = (
            (validation_accuracy > best_validation_accuracy) or (abs(validation_accuracy - best_validation_accuracy) < 1e-6 and validation_loss < best_validation_loss)
        )

        if improved:
            best_validation_accuracy = validation_accuracy
            best_validation_loss = validation_loss 
            epochs_without_improvement = 0
            save_checkpoint(mode_path=args.model, model=model, labels=labels, feature_mean=feature_mean, feature_std=feature_std, sequence_length=args.sequence_length, validation_accuracy=validation_accuracy)
            print(f"  New best model saved to {args.model}")

        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement.")
            break

    print(f"\nTraining completed. Best validation accuracy: {best_validation_accuracy:.3f}")
    print(f"Checkpoint: {args.model}")
    return 0

def configure_camera(camera: cv2.VideoCapture) -> None:
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    camera.set(cv2.CAP_PROP_FPS, target_fps)

def try_open_camera(camera_index: int) -> Optional[cv2.VideoCapture]:
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    backends.append(cv2.CAP_ANY)

    for backend in backends:
        camera = cv2.VideoCapture(camera_index, backend)
        configure_camera(camera)
        frame_ok, _ = camera.read()
        if frame_ok:
            return camera
        camera.release()
    return None

def open_camera(requested_index: int) -> tuple[cv2.VideoCapture, int]:
    attempted_indices = []

    if requested_index >= 0:
        candidate_indices = [requested_index]
    else:
        candidate_indices = camera_search_order

    seen = set()

    for camera_index in candidate_indices:
        if camera_index in seen:
            continue
        seen.add(camera_index)
        attempted_indices.append(camera_index)
        camera = try_open_camera(camera_index)
        if camera is not None:
            return camera, camera_index

    raise RuntimeError(f"Failed to open camera. Attempted indices: {attempted_indices}")

def draw_text_lines(frame: np.ndarray, lines: list[str], origin: tuple[int, int], color: tuple[int, int, int], line_height: int = 28, scale: float = 0.65) -> None:
    x_pos, y_pos = origin
    for line in lines:
        cv2.putText(frame, line, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y_pos += line_height

def show_countdown(camera: cv2.VideoCapture, seconds: int) -> None:
    for seconds_left in range(seconds, 0, -1):
        started_at = time.time()
        while time.time() - started_at < 1.0:
            frame_ok, frame = camera.read()
            if not frame_ok:
                continue
            display_frame = frame.copy()
            draw_text_lines(display_frame, [f"Recording starts in {seconds_left}...", "Get Ready..."], origin=(30,50), color=(0,255, 255))
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
    return True

def format_probability_line(ranked_probabilities: list[tuple[str, float]], top_k: int=4) -> str:
    top_predictions = ranked_probabilities[:top_k]
    return " | ".join(f"{label}: {prob:.2f}" for label, prob in top_predictions)

def draw_overlay(
        frame: np.ndarray,
        recording: bool,
        frame_count: int,
        camera_index: int,
        last_prediction: Optional[str],
        last_confidence: Optional[float],
        history: list[str],
        ranked_probabilities: list[tuple[str, float]]
)-> np.ndarray:
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (frame.shape[1]-12, 200), (245, 245, 245), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.30, 0.0, frame)

    status_text = "Recording..." if recording else "IDLE"
    status_color = (0, 0, 255) if recording else (0, 160, 0)

    if last_prediction is None:
        prediction_line = "Last move: -" 
    else:
        prediction_line = f"Last move: {last_prediction} ({last_confidence:.2f})" 

    history_text = "Recent moves: " + (" ".join(history[-8:]) if history else "-") 
    probability_text = "Top guesses: " + (
        format_probability_line(ranked_probabilities) if ranked_probabilities else "-"
    )

    lines = [
        f"{project_name} | Camera {camera_index} | {status_text}",
        f"Frames in current clip: {frame_count}",
        prediction_line,
        history_text,
        probability_text,
        "Space=start/stop    c=clear history    r=reset history  z=undo last   q=quit"
    ]
    draw_text_lines(frame, lines, origin=(25,40), color=status_color if recording else (40,40,0))
    return frame

def command_live(args: argparse.Namespace) -> int:
    require_mediapipe()
    device = choose_device(args.device)

    try:
        model, labels, feature_mean, feature_std, sequence_length = load_model_bundle(args.model, device)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(f"Could not load model: {error}")
        return 1

    try:
        camera, active_camera_index = open_camera(args.camera)
    except RuntimeError as error:
        print(error)
        return 1

    print(f"Loaded model from {args.model}")
    print(f"Model labels: {', '.join(labels)}")
    print(f"Using camera_index: {active_camera_index}")

    last_prediction: Optional[str] = None
    last_confidence: Optional[float] = None
    ranked_probabilities: list[tuple[str, float]] = []
    prediction_history: list[str] = []
    recorded_features: list[np.ndarray] = []
    recording = False

    drawing_utils = mp.solutions.drawing_utils if mp is not None else None
    hands_connections = mp.solutions.hands.HAND_CONNECTIONS if mp is not None else None

    try:
        with create_hands_tracker(static_image_mode=False) as hands_tracker:
            while True:
                frame_ok, frame = camera.read()
                if not frame_ok:
                    print("Warning: could not read a frame from the webcam.")
                    continue
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands_tracker.process(rgb_frame)
                frame_feature = extract_frame_feature(results)

                if recording:
                    recorded_features.append(frame_feature)
                    if drawing_utils is not None and getattr(results, "multi_hand_landmarks", None):
                        for hand_landmarks in results.multi_hand_landmarks:
                            drawing_utils.draw_landmarks(frame, hand_landmarks, hands_connections)
                frame = cv2.flip(frame, 1)
                frame = draw_overlay(frame, recording=recording, frame_count=len(recorded_features), camera_index=active_camera_index, last_prediction=last_prediction, last_confidence=last_confidence, history=prediction_history, ranked_probabilities=ranked_probabilities)
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    prediction_history.clear()
                    last_prediction = None
                    last_confidence = None
                    ranked_probabilities = []
                elif key in (ord("z"), 8, 127):
                    if prediction_history:
                        prediction_history.pop()
                    if prediction_history:
                        last_prediction = prediction_history[-1]
                    else:
                        last_prediction = None
                        last_confidence = None
                        ranked_probabilities = []
                elif key == ord(" "):
                    if not recording:
                        if not show_countdown(camera, args.countdown):
                            break
                        recorded_features = []
                        recording = True
                        print("Recording started...")
                    else:
                        recording = False
                        raw_sequence = np.asarray(recorded_features, dtype = np.float32)
                        if len(raw_sequence) < args.min_frames:
                            print(f"Clip too short: {len(raw_sequence)} frames.")
                            recorded_features = []
                            continue
                        predicted_label, confidence, ranked_probabilities = predict_move(model=model, raw_sequence=raw_sequence, labels=labels, feature_mean=feature_mean, feature_std=feature_std, sequence_length=sequence_length, device=device)
                        last_prediction = predicted_label
                        last_confidence = confidence
                        prediction_history.append(predicted_label)
                        print(f"Predicted move: {predicted_label} | confidence {confidence:.3f} | {format_probability_line(ranked_probabilities)}")
                        recorded_features = []
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0

def command_check(args: argparse.Namespace) -> int:
    print(f"{project_name}")
    print(f"Dataset path: {args.dataset.resolve()}")

    if not args.dataset.exists():
        print("Dataset folder does not exist yet.")
        return 1
    available_labels = discover_available_labels(args.dataset, args.labels)
    if not available_labels:
        print("No dataset clips were found for the requested labels.")
        return 1
    print("Dataset clip count:")
    sample_video: Optional[Path] = None
    for label in available_labels:
        videos = list_video_files(args.dataset / label)
        print(f" {label}: {len(videos)} clips")
        if sample_video is None and videos:
            sample_video = videos[0]
    if mp is None:
        print("Mediapipe: missing")
    else:
        print("Mediapipe: available")
    if sample_video is not None and mp is not None:
        with create_hands_tracker(static_image_mode=True) as hands_tracker:
            sequence, used_cache = extract_video_features(sample_video, hands_tracker=hands_tracker, refresh_cache=False)
            if sequence is None:
                print(f"Sample extraction failed for {sample_video.name}")
            else:
                prepared = prepare_sequence_for_model(sequence, sequence_length=default_sequence_length, augment=False)
                print(f"Sample Clip: {sample_video}")
                print(f"Raw sequence shape: {sequence.shape}")
                print(f"Prepared input shape: {prepared.shape}")
                print(f"Landmarks detected: {'Yes' if sequence_has_landmarks(sequence) else 'No'}")
                print(f"Source: {'cache' if used_cache else 'video'}")
    if args.model.exists():
        try:
            device = torch.device("cpu")
            _, labels, _, _, sequence_length = load_model_bundle(args.model, device)
            print(f"Labels: {','.join(labels)}")
            print(f"Sequence length: {sequence_length}")
        except Exception as e:
            print(f"Failed to load model: {e}")
    else:
        print(f"Model file not found: {args.model}")
    return 0

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cube Motion training and live testing tool.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a move classifier from dataset videos.")
    train_parser.add_argument("--dataset", type=Path, default=dataset_dir, help="Dataset folder Path.")
    train_parser.add_argument("--model", type=Path, default=model_path, help="Checkpoint output path")
    train_parser.add_argument("--labels", type=parse_label_list, default=None, help="Comma-seperated labels to train, like R,Rp,U,Up. Default: auto-detect avaliable labels.")
    train_parser.add_argument("--epochs", type=int, default=train_epochs, help="Number of training epochs.")
    train_parser.add_argument("--batch-size", type=int, default=train_batch_size, help="Batch size for training and validation.")
    train_parser.add_argument("--learning-rate", type=float, default=learning_rate, help="Optimizer learning rate.")
    train_parser.add_argument("--weight-decay", type=float, default=weight_decay, help="AdamW weight decay.")
    train_parser.add_argument("--validation-ratio", type=float, default=validation_ratio, help="Fraction of dataset to use for validation.")
    train_parser.add_argument("--sequence-length", type=int, default=default_sequence_length, help="Fixed number of frames per clip after resampling.")
    train_parser.add_argument("--patience", type=int, default=early_stopping_patience, help="Early stopping patience.")
    train_parser.add_argument("--refresh-cache", action="store_true", help="Ignore cached Mediapipe features and recompute them from the videos.")
    train_parser.add_argument("--device", choices=["auto","cpu","cuda"], default="auto", help="Training device.")
    train_parser.add_argument("--seed", type=int, default=random_seed, help="Random seed")

    train_parser.set_defaults(func=command_train)

    live_parser = subparsers.add_parser("live", help="Record a live webcam clip and predict the move.")
    live_parser.add_argument("--model", type=Path, default=model_path, help="Checkpoint path to load the model.")
    live_parser.add_argument("--camera", type=int, default=default_camera_index, help="Camera index.")
    live_parser.add_argument("--countdown", type=int, default=countdown_seconds, help="Seconds to wait before live recording starts.")
    live_parser.add_argument("--min-frames", type=int, default=min_recording_frame, help="Minimum frames required for a live clip.")
    live_parser.add_argument("--device", choices=["auto","cpu","cuda"], default="auto", help="Inference device.")
    live_parser.set_defaults(func=command_live)

    check_parser = subparsers.add_parser("check", help="Run a quick check of the dataset and model.")
    check_parser.add_argument("--dataset", type=Path, default=dataset_dir, help="Dataset folder Path.")
    check_parser.add_argument("--model", type=Path, default=model_path, help="Checkpoint path to inspect.")
    check_parser.add_argument("--labels", type=parse_label_list, default=None, help="Optional comma-seperated label filter.")
    check_parser.set_defaults(func=command_check)

    return parser

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))

if __name__ == "__main__":
    main()

