"""Find, read, and slice audio files."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from audiodeepfake.errors import AudioLoadError

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SEGMENT_SECONDS = 1.0

# wav, flac, and ogg are read by libsndfile. mp3, m4a, aac, and wma need ffmpeg.
AUDIO_EXTENSIONS = frozenset(
    {
        ".wav",
        ".flac",
        ".ogg",
        ".oga",
        ".mp3",
        ".m4a",
        ".aac",
        ".aiff",
        ".aif",
        ".wma",
        ".opus",
    }
)


def iter_audio_files(root: Path) -> list[Path]:
    """Return supported audio files under ``root``, skipping hidden names."""
    root = Path(root)
    if root.is_file():
        if root.name.startswith(".") or root.suffix.lower() not in AUDIO_EXTENSIONS:
            return []
        return [root]
    if not root.is_dir():
        return []

    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            path = Path(dirpath) / name
            if path.suffix.lower() in AUDIO_EXTENSIONS:
                found.append(path)
    return found


def collect_audio_files(path: Path) -> list[Path]:
    """Return audio files for a CLI or API input, or raise ``AudioLoadError``."""
    path = Path(path)
    if not path.exists():
        raise AudioLoadError(f"Path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            supported = ", ".join(sorted(AUDIO_EXTENSIONS))
            raise AudioLoadError(
                f"{path} is not a supported audio file. Supported extensions: {supported}."
            )
        return [path]
    files = iter_audio_files(path)
    if not files:
        raise AudioLoadError(f"No supported audio files under {path}")
    return files


def expand_inputs(source: str | Path | list | tuple) -> tuple[list[Path], bool]:
    """Expand a path, directory, or list of paths.

    The boolean is true when the caller should get a list of results back
    (a directory or an explicit list) instead of one result.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        files = collect_audio_files(path)
        return files, path.is_dir()
    if isinstance(source, (list, tuple)):
        if not source:
            raise AudioLoadError("No input paths were given.")
        files: list[Path] = []
        for item in source:
            files.extend(collect_audio_files(Path(item)))
        if not files:
            raise AudioLoadError("No supported audio files in the input paths.")
        return files, True
    raise TypeError("path must be a file path, a directory path, or a list of paths.")


def load_audio(path: Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Load one file as mono float32 audio at ``sample_rate``."""
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - dependency is installed with the package
        raise AudioLoadError("Reading audio requires librosa.") from exc
    try:
        waveform, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    except Exception as exc:
        raise AudioLoadError(
            f"Could not read {path}. wav, flac, and ogg use libsndfile. "
            "mp3, m4a, aac, and wma also need ffmpeg on PATH."
        ) from exc
    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise AudioLoadError(f"Audio file is empty: {path}")
    return audio


def segment_waveform(
    waveform: np.ndarray,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    mode: str = "all",
) -> list[np.ndarray]:
    """Cut audio into fixed windows.

    Full windows are kept from the start. A leftover shorter than one window
    is dropped, matching ``predict_audio`` in the original notebook. A clip
    that is itself shorter than one window is padded with zeros so it still
    gets a score. Training pads short clips the same way.
    """
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive.")
    if mode not in {"all", "first"}:
        raise ValueError("mode must be 'all' or 'first'.")
    window = int(round(segment_seconds * sample_rate))
    if window <= 0:
        raise ValueError("The segment window has no samples.")

    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise AudioLoadError("Audio is empty.")
    if audio.size < window:
        padded = np.zeros(window, dtype=np.float32)
        padded[: audio.size] = audio
        return [padded]

    segments: list[np.ndarray] = []
    for start in range(0, audio.size, window):
        chunk = audio[start : start + window]
        if chunk.size < window:
            break
        segments.append(np.ascontiguousarray(chunk))
        if mode == "first":
            break
    if not segments:
        raise AudioLoadError("No full audio window could be built.")
    return segments
