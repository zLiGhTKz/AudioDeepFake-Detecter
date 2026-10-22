"""Read public deepfake-dataset layouts. This module does not download data."""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path

from audiodeepfake.audio import iter_audio_files
from audiodeepfake.errors import DatasetLayoutError
from audiodeepfake.labels import FAKE, REAL, label_from_filename, normalize_label

FORMATS = (
    "auto",
    "directory",
    "filename",
    "notebook",
    "notebook-f",
    "asvspoof",
    "wavefake",
    "csv",
    "in-the-wild",
)

PATH_COLUMNS = ("path", "file", "filename", "audio", "filepath", "utt", "utterance")
LABEL_COLUMNS = ("label", "class", "target", "key")

CLASS_DIRS = {
    "fake": FAKE,
    "spoof": FAKE,
    "spoofs": FAKE,
    "synthetic": FAKE,
    "deepfake": FAKE,
    "deepfakes": FAKE,
    "generated": FAKE,
    "real": REAL,
    "reals": REAL,
    "bonafide": REAL,
    "bona-fide": REAL,
    "bona_fide": REAL,
    "genuine": REAL,
    "human": REAL,
}

WAVEFAKE_MARKERS = (
    "melgan",
    "waveglow",
    "hifigan",
    "wavegan",
    "parallel_wavegan",
)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


@dataclass(frozen=True)
class SplitSpec:
    root: Path | None = None
    format: str = "auto"
    protocol: Path | None = None
    audio_dir: Path | None = None
    csv_path: Path | None = None
    real_dir: Path | None = None
    audio_root: Path | None = None


@dataclass(frozen=True)
class SplitData:
    samples: list[Sample]
    format: str


def summarize(samples: list[Sample]) -> dict[str, int]:
    fake = sum(sample.label == FAKE for sample in samples)
    real = sum(sample.label == REAL for sample in samples)
    return {"total": len(samples), "fake": fake, "real": real}


def balance_undersample(samples: list[Sample], seed: int = 123) -> list[Sample]:
    """Shrink the larger class so fake and real counts match.

    The notebook undersampled real clips down to the fake count. This does the
    same job for whichever class is larger. ``seed`` keeps the draw repeatable.
    """
    grouped = {FAKE: [], REAL: []}
    for sample in samples:
        if sample.label not in grouped:
            raise DatasetLayoutError(f"Unexpected label {sample.label} on {sample.path}")
        grouped[sample.label].append(sample)
    if not grouped[FAKE] or not grouped[REAL]:
        counts = summarize(samples)
        raise DatasetLayoutError(
            "Both fake and real clips are required before undersampling. "
            f"Found {counts['fake']} fake and {counts['real']} real."
        )
    keep = min(len(grouped[FAKE]), len(grouped[REAL]))
    rng = random.Random(seed)
    fake = grouped[FAKE] if len(grouped[FAKE]) == keep else rng.sample(grouped[FAKE], keep)
    real = grouped[REAL] if len(grouped[REAL]) == keep else rng.sample(grouped[REAL], keep)
    return list(fake) + list(real)


def overlapping_paths(left: list[Sample], right: list[Sample]) -> list[Path]:
    right_paths = {sample.path.resolve() for sample in right}
    return [sample.path for sample in left if sample.path.resolve() in right_paths]


def load_split(spec: SplitSpec) -> SplitData:
    if spec.format not in FORMATS:
        choices = ", ".join(FORMATS)
        raise DatasetLayoutError(f"Unknown format {spec.format!r}. Choose one of: {choices}.")
    if spec.protocol is not None:
        samples = load_asvspoof(spec.protocol, audio_dir=spec.audio_dir or spec.root)
        return SplitData(samples, "asvspoof")
    if spec.csv_path is not None:
        samples = load_csv(spec.csv_path, audio_root=spec.audio_root or spec.root)
        return SplitData(samples, "csv")
    if spec.root is None:
        raise DatasetLayoutError("Pass a dataset directory, a protocol file, or a csv.")

    root = Path(spec.root)
    if not root.exists():
        raise DatasetLayoutError(f"Dataset path does not exist: {root}")

    chosen = detect_format(root) if spec.format == "auto" else spec.format
    if chosen == "in-the-wild":
        return SplitData(load_in_the_wild(root), chosen)
    if chosen == "csv":
        return SplitData(load_csv(_single_csv(root), audio_root=spec.audio_root or root), chosen)
    if chosen == "asvspoof":
        return SplitData(load_asvspoof_root(root, audio_dir=spec.audio_dir), chosen)
    if chosen == "wavefake":
        return SplitData(load_wavefake(root, spec.real_dir), chosen)
    if chosen == "directory":
        return SplitData(load_directory(root), chosen)
    if chosen in {"filename", "notebook", "notebook-f"}:
        return SplitData(load_by_filename(root, chosen), chosen)
    raise DatasetLayoutError(f"Unknown format {chosen!r}.")


def detect_format(root: Path) -> str:
    root = Path(root)
    if (root / "meta.csv").is_file() or (root / "release_in_the_wild" / "meta.csv").is_file():
        return "in-the-wild"
    csv_files = [path for path in _top_csv_files(root) if _csv_is_manifest(path)]
    if len(csv_files) == 1:
        return "csv"
    if find_protocol_files(root):
        return "asvspoof"
    # Vocoder folders win over a sibling real/ directory so a WaveFake
    # checkout is not read as a plain class-folder dataset.
    if _looks_like_wavefake(root):
        return "wavefake"
    if _has_class_dirs(root):
        return "directory"
    return "filename"


def load_directory(root: Path) -> list[Sample]:
    root = Path(root)
    samples: list[Sample] = []
    skipped: list[Path] = []
    for audio in iter_audio_files(root):
        label = _nearest_class_dir(audio, root)
        if label is None:
            skipped.append(audio)
            continue
        samples.append(Sample(audio, label))
    if not samples:
        raise DatasetLayoutError(
            f"No labeled audio under {root}. Put clips in real/ and fake/ folders. "
            "bonafide, genuine, spoof, and synthetic folder names also work."
        )
    if skipped:
        preview = ", ".join(str(path) for path in skipped[:5])
        raise DatasetLayoutError(
            f"{len(skipped)} audio files are outside a real/ or fake/ folder "
            f"(for example {preview})."
        )
    return samples


def load_by_filename(root: Path, scheme: str) -> list[Sample]:
    audios = iter_audio_files(root)
    if not audios:
        raise DatasetLayoutError(f"No audio files under {root}")
    samples: list[Sample] = []
    unlabeled: list[Path] = []
    for audio in audios:
        label = label_from_filename(audio.name, scheme)
        if label is None:
            unlabeled.append(audio)
            continue
        samples.append(Sample(audio, label))
    if unlabeled:
        preview = ", ".join(path.name for path in unlabeled[:5])
        raise DatasetLayoutError(
            f"{len(unlabeled)} files have no fake/real word in the file name "
            f"(for example {preview}). Use real/ and fake/ folders, a csv, or "
            "--format notebook if every other file should count as real."
        )
    return samples


def load_wavefake(generated_dir: Path, real_dir: Path | None) -> list[Sample]:
    """Label WaveFake generated audio as fake and a separate real folder as real.

    The WaveFake zip does not include the genuine LJSpeech or JSUT recordings.
    A ``real/`` folder inside or next to ``generated_dir`` is used when
    ``real_dir`` is omitted.
    """
    if real_dir is None:
        real_dir = _default_real_dir(Path(generated_dir))
    if real_dir is None:
        raise DatasetLayoutError(
            "WaveFake audio in generated_audio is synthetic only. Pass --real-dir "
            "with genuine clips, usually LJSpeech-1.1/wavs. The WaveFake download "
            "does not include that reference set."
        )
    generated_dir = Path(generated_dir)
    real_dir = Path(real_dir)
    if not generated_dir.exists():
        raise DatasetLayoutError(f"Generated audio path does not exist: {generated_dir}")
    if not real_dir.exists():
        raise DatasetLayoutError(f"Real audio path does not exist: {real_dir}")
    fakes = iter_audio_files(generated_dir)
    reals = iter_audio_files(real_dir)
    if not fakes:
        raise DatasetLayoutError(f"No generated audio under {generated_dir}")
    if not reals:
        raise DatasetLayoutError(f"No real audio under {real_dir}")
    real_paths = {path.resolve() for path in reals}
    samples = [Sample(path, FAKE) for path in fakes if path.resolve() not in real_paths]
    samples.extend(Sample(path, REAL) for path in reals)
    return samples


def load_in_the_wild(root: Path) -> list[Sample]:
    """Read the Müller et al. release (``meta.csv`` next to the wav files)."""
    root = Path(root)
    meta = root / "meta.csv"
    if not meta.is_file():
        nested = root / "release_in_the_wild" / "meta.csv"
        if nested.is_file():
            meta = nested
        else:
            raise DatasetLayoutError(
                "In-the-Wild layout was not found. Expected meta.csv with columns "
                "file, speaker, and label (spoof or bona-fide), usually inside "
                "release_in_the_wild/."
            )
    return load_csv(meta, audio_root=meta.parent)


def load_csv(csv_path: Path, audio_root: Path | None = None) -> list[Sample]:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise DatasetLayoutError(f"CSV does not exist: {csv_path}")
    root = Path(audio_root) if audio_root is not None else csv_path.parent
    samples: list[Sample] = []
    missing: list[str] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise DatasetLayoutError(f"{csv_path} has no header row.")
        fields = {name.strip().lower(): name for name in reader.fieldnames if name}
        path_key = next((fields[name] for name in PATH_COLUMNS if name in fields), None)
        label_key = next((fields[name] for name in LABEL_COLUMNS if name in fields), None)
        if path_key is None or label_key is None:
            found = ", ".join(reader.fieldnames)
            raise DatasetLayoutError(
                f"{csv_path} needs a path column ({', '.join(PATH_COLUMNS)}) and a "
                f"label column ({', '.join(LABEL_COLUMNS)}). Found columns: {found}."
            )
        for row in reader:
            raw_path = (row.get(path_key) or "").strip()
            raw_label = (row.get(label_key) or "").strip()
            if not raw_path:
                continue
            label = normalize_label(raw_label)
            if label is None:
                raise DatasetLayoutError(
                    f"Unknown label {raw_label!r} for {raw_path} in {csv_path}."
                )
            audio = _resolve_audio_path(raw_path, root)
            if audio is None:
                missing.append(raw_path)
                continue
            samples.append(Sample(audio, label))
    if missing:
        preview = ", ".join(missing[:5])
        raise DatasetLayoutError(
            f"{len(missing)} CSV rows have no audio file (for example {preview}). "
            f"Looked next to {csv_path} and under {root}."
        )
    if not samples:
        raise DatasetLayoutError(f"No labeled rows in {csv_path}.")
    return samples


def load_asvspoof(protocol: Path, audio_dir: Path | None) -> list[Sample]:
    """Join an ASVspoof CM protocol to audio files.

    2019 Logical Access rows look like ``SPEAKER UTTERANCE - SYSTEM KEY`` with
    KEY ``bonafide`` or ``spoof``. Shorter rows that end in the same key are
    accepted too. The utterance id is matched to the file stem.
    """
    protocol = Path(protocol)
    if audio_dir is None:
        raise DatasetLayoutError(
            "ASVspoof needs the audio folder. Pass --train-audio-dir "
            "(for example ASVspoof2019_LA_train/flac)."
        )
    audio_dir = Path(audio_dir)
    if not protocol.is_file():
        raise DatasetLayoutError(f"Protocol file does not exist: {protocol}")
    if not audio_dir.exists():
        raise DatasetLayoutError(f"Audio folder does not exist: {audio_dir}")
    index = _index_audio(audio_dir)
    samples: list[Sample] = []
    missing: list[str] = []
    for utterance, label in read_protocol(protocol):
        path = _lookup_audio(index, utterance)
        if path is None:
            missing.append(utterance)
            continue
        samples.append(Sample(path, label))
    if missing:
        preview = ", ".join(missing[:5])
        raise DatasetLayoutError(
            f"{len(missing)} protocol rows have no audio file (for example {preview}). "
            f"Indexed audio under {audio_dir}."
        )
    if not samples:
        raise DatasetLayoutError(f"No usable rows in {protocol}.")
    return samples


def load_asvspoof_root(root: Path, audio_dir: Path | None = None) -> list[Sample]:
    root = Path(root)
    protocols = find_protocol_files(root)
    if len(protocols) != 1:
        preview = ", ".join(path.name for path in protocols[:8]) or "none"
        raise DatasetLayoutError(
            f"Expected one ASVspoof CM protocol under {root}, found {len(protocols)} "
            f"({preview}). Pass --train-protocol and --train-audio-dir so train, "
            "dev, and eval are not mixed."
        )
    return load_asvspoof(protocols[0], audio_dir or root)


def read_protocol(protocol: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    text = Path(protocol).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        parsed = parse_protocol_line(line)
        if parsed is not None:
            rows.append(parsed)
    if not rows:
        raise DatasetLayoutError(
            f"{protocol} has no rows ending in bonafide or spoof. "
            "2019 CM columns are: SPEAKER UTTERANCE - SYSTEM KEY."
        )
    return rows


def parse_protocol_line(line: str) -> tuple[str, int] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    parts = stripped.split()
    if len(parts) < 2:
        return None
    label = normalize_label(parts[-1])
    if label is None:
        return None
    utterance = parts[1] if len(parts) >= 5 else parts[0]
    return utterance, label


def find_protocol_files(root: Path) -> list[Path]:
    found: list[Path] = []
    root = Path(root)
    if root.is_file():
        return [root] if _file_looks_like_protocol(root) else []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        for name in filenames:
            if name.startswith(".") or not name.lower().endswith(".txt"):
                continue
            path = Path(dirpath) / name
            if _file_looks_like_protocol(path):
                found.append(path)
    return sorted(found)


def _file_looks_like_protocol(path: Path) -> bool:
    name = path.name.lower()
    if ".asv." in name:
        return False
    named = any(token in name for token in (".cm.", "protocol", ".trl", ".trn"))
    count = _count_protocol_lines(path)
    if named:
        return count > 0
    return count >= 2


def _count_protocol_lines(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    return sum(parse_protocol_line(line) is not None for line in text.splitlines())


def _index_audio(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for audio in iter_audio_files(root):
        for key in (audio.name, audio.stem):
            previous = index.get(key)
            if previous is not None and previous.resolve() != audio.resolve():
                raise DatasetLayoutError(
                    f"Two audio files share the name {key}: {previous} and {audio}."
                )
            index[key] = audio
    if not index:
        raise DatasetLayoutError(f"No audio files under {root}")
    return index


def _lookup_audio(index: dict[str, Path], utterance: str) -> Path | None:
    raw = utterance.strip()
    path = Path(raw)
    for key in (raw, path.name, path.stem):
        if key in index:
            return index[key]
    return None


def _resolve_audio_path(raw_path: str, root: Path) -> Path | None:
    candidate = Path(raw_path)
    if candidate.is_file():
        return candidate
    rooted = root / raw_path
    if rooted.is_file():
        return rooted
    return None


def _nearest_class_dir(audio: Path, root: Path) -> int | None:
    current = audio.parent
    root = root.resolve()
    while True:
        label = CLASS_DIRS.get(current.name.lower())
        if label is not None:
            return label
        if current.resolve() == root or current.parent == current:
            return None
        current = current.parent


def _has_class_dirs(root: Path) -> bool:
    root = Path(root)
    if not root.is_dir():
        return False
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() in CLASS_DIRS:
            return True
    return False


def _default_real_dir(generated_dir: Path) -> Path | None:
    generated_dir = Path(generated_dir)
    names = ("real", "reals", "genuine", "bonafide", "bona-fide", "human")
    bases = [generated_dir]
    if generated_dir.parent != generated_dir:
        bases.append(generated_dir.parent)
    for base in bases:
        for name in names:
            for candidate in (base / name, base / name / "wavs"):
                if candidate.is_dir() and iter_audio_files(candidate):
                    return candidate
    return None


def _looks_like_wavefake(root: Path) -> bool:
    if not root.is_dir():
        return False
    for child in root.iterdir():
        if not child.is_dir():
            continue
        lowered = child.name.lower()
        if any(marker in lowered for marker in WAVEFAKE_MARKERS):
            return True
    return False


def _top_csv_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path for path in root.glob("*.csv") if path.is_file() and not path.name.startswith(".")
    )


def _single_csv(root: Path) -> Path:
    files = [path for path in _top_csv_files(root) if _csv_is_manifest(path)]
    if len(files) != 1:
        raise DatasetLayoutError(
            f"Expected one label CSV in {root}, found {len(files)}. Pass the file with --train-csv."
        )
    return files[0]


def _csv_is_manifest(path: Path) -> bool:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return False
            fields = {name.strip().lower() for name in reader.fieldnames if name}
    except OSError:
        return False
    return bool(fields & set(PATH_COLUMNS)) and bool(fields & set(LABEL_COLUMNS))
