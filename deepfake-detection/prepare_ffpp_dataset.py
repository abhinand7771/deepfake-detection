from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


DEFAULT_METHODS = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "FaceShifter",
    "NeuralTextures",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert downloaded FaceForensics++ videos into dataset/real and dataset/fake image folders."
    )
    parser.add_argument(
        "--ffpp-root",
        type=str,
        default="ffpp",
        help="Root folder that contains the official FaceForensics++ download structure.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="dataset",
        help="Output dataset root that will be filled with dataset/real and dataset/fake.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=DEFAULT_METHODS,
        help="Manipulated methods to convert to fake samples. Defaults to standard FF++ methods.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=10,
        help="Sample every Nth frame from each video while extracting faces.",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=40,
        help="Maximum number of face frames kept per video to avoid huge dataset skew.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing dataset/real and dataset/fake contents before organizing new data.",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=40,
        help="Minimum face pixel size. Smaller faces are skipped.",
    )
    return parser.parse_args()


def ensure_face_detector() -> cv2.CascadeClassifier:
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if classifier.empty():
        raise RuntimeError("OpenCV Haar cascade could not be loaded. Check your OpenCV installation.")
    return classifier


def list_ffpp_videos(ffpp_root: Path, methods: list[str]) -> tuple[list[Path], list[Path]]:
    real_videos: list[Path] = []
    fake_videos: list[Path] = []

    original_dir = ffpp_root / "original_sequences" / "youtube" / "c40" / "videos"
    if original_dir.exists():
        real_videos.extend(sorted(original_dir.glob("*.mp4")))

    for method in methods:
        method_dir = ffpp_root / "manipulated_sequences" / method / "c40" / "videos"
        if method_dir.exists():
            fake_videos.extend(sorted(method_dir.glob("*.mp4")))

    return real_videos, fake_videos


def crop_largest_face(frame: np.ndarray, detector: cv2.CascadeClassifier, min_face_size: int) -> np.ndarray | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face_size, min_face_size))
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    margin_x = int(0.20 * w)
    margin_y = int(0.20 * h)
    x0 = max(0, x - margin_x)
    y0 = max(0, y - margin_y)
    x1 = min(frame.shape[1], x + w + margin_x)
    y1 = min(frame.shape[0], y + h + margin_y)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return crop


def save_face_frames(video_path: Path, output_dir: Path, label: str, detector: cv2.CascadeClassifier, step: int, max_frames: int, min_face_size: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    saved = 0
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % step != 0:
            frame_index += 1
            continue
        face = crop_largest_face(frame, detector, min_face_size)
        if face is not None:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            save_path = output_dir / f"{video_path.stem}_{saved:04d}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            saved += 1
            if saved >= max_frames:
                break
        frame_index += 1

    cap.release()
    return saved


def prepare_dataset(ffpp_root: Path, dataset_root: Path, methods: list[str], step: int, max_frames: int, min_face_size: int, reset: bool = False) -> dict[str, int]:
    ffpp_root = ffpp_root.resolve()
    if not ffpp_root.exists():
        raise FileNotFoundError(
            f"FF++ root not found: {ffpp_root}. Download the official dataset first, or pass --ffpp-root to the downloaded folder."
        )

    real_dir = dataset_root / "real"
    fake_dir = dataset_root / "fake"

    if reset:
        for directory in (real_dir, fake_dir):
            if directory.exists():
                shutil.rmtree(directory)

    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    detector = ensure_face_detector()
    real_videos, fake_videos = list_ffpp_videos(ffpp_root, methods)

    if not real_videos and not fake_videos:
        raise FileNotFoundError(
            "No FaceForensics++ videos were found. Expected folders like "
            "original_sequences/youtube/c40/videos and manipulated_sequences/<method>/c40/videos."
        )

    real_count = 0
    fake_count = 0

    for video_path in real_videos:
        real_count += save_face_frames(video_path, real_dir, "REAL", detector, step, max_frames, min_face_size)

    for video_path in fake_videos:
        fake_count += save_face_frames(video_path, fake_dir, "FAKE", detector, step, max_frames, min_face_size)

    return {"REAL": real_count, "FAKE": fake_count}


def main() -> None:
    args = parse_args()
    ffpp_root = Path(args.ffpp_root)
    dataset_root = Path(args.dataset_root)

    try:
        counts = prepare_dataset(
            ffpp_root=ffpp_root,
            dataset_root=dataset_root,
            methods=args.methods,
            step=args.frame_step,
            max_frames=args.max_frames_per_video,
            min_face_size=args.min_face_size,
            reset=args.reset,
        )
        print("Dataset organization complete.")
        print(f"Saved real faces: {counts['REAL']}")
        print(f"Saved fake faces: {counts['FAKE']}")
        print(f"Output folders: {dataset_root / 'real'} and {dataset_root / 'fake'}")
    except Exception as exc:
        print(f"[ERROR] {exc}")
        print("\nTo prepare the dataset:")
        print("1. Download FaceForensics++ with the official script using --server EU2")
        print("2. Place the downloaded root folder at the path you pass to --ffpp-root")
        print("3. Run: python prepare_ffpp_dataset.py --ffpp-root ffpp --dataset-root dataset")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
