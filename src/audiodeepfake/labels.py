"""Label ids shared by training and inference.

The original notebook used ``0`` for synthetic speech and ``1`` for real speech.
Keep that mapping so old checkpoints and new training runs agree.
"""

from __future__ import annotations

import re
from pathlib import Path

FAKE = 0
REAL = 1

ID2LABEL = {FAKE: "fake", REAL: "real"}
LABEL2ID = {"fake": FAKE, "real": REAL}

FAKE_NAMES = frozenset({"fake", "spoof", "spoofed", "synth", "synthetic", "deepfake", "generated"})
REAL_NAMES = frozenset({"real", "bonafide", "genuine", "human", "authentic"})

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def canon(value: str) -> str:
    """Lowercase a label and drop spaces, hyphens, and underscores."""
    return _NON_ALNUM.sub("", value.strip().lower())


def label_kind(value: str) -> str | None:
    """Return ``fake``, ``real``, or ``None`` for a free-text label."""
    name = canon(value)
    if not name:
        return None
    if name in FAKE_NAMES or "spoof" in name or name.startswith("fake"):
        return "fake"
    if name in REAL_NAMES or "bonafide" in name or name.startswith("real"):
        return "real"
    return None


def normalize_label(value: object) -> int | None:
    """Map a dataset label onto ``0`` (fake) or ``1`` (real)."""
    text = str(value).strip()
    if text in {"0", "1"}:
        return FAKE if text == "0" else REAL
    kind = label_kind(text)
    if kind == "fake":
        return FAKE
    if kind == "real":
        return REAL
    return None


def resolve_label_indices(id2label: dict) -> tuple[int, int, bool]:
    """Find the fake and real logit indexes on a model config.

    Returns ``(fake_index, real_index, explicit)``. ``explicit`` is false when
    the config still has placeholder names such as ``LABEL_0``. In that case
    index 0 is treated as fake and index 1 as real, matching the notebook.
    """
    normalized: dict[int, str] = {}
    for key, value in id2label.items():
        normalized[int(key)] = str(value)

    fake_index = None
    real_index = None
    for index, name in normalized.items():
        kind = label_kind(name)
        if kind == "fake" and fake_index is None:
            fake_index = index
        elif kind == "real" and real_index is None:
            real_index = index

    explicit = fake_index is not None and real_index is not None
    if not explicit:
        if fake_index is None:
            fake_index = FAKE
        if real_index is None:
            real_index = REAL
        if fake_index == real_index:
            real_index = REAL if fake_index != REAL else FAKE
    if fake_index == real_index:
        raise ValueError("Fake and real would use the same logit index.")
    return fake_index, real_index, explicit


def label_from_filename(filename: str, scheme: str) -> int | None:
    """Label one file name.

    ``notebook`` copies ``collect_audio_files_and_labels``: the lowercase
    substring ``fake`` means fake, and every other name means real.

    ``notebook-f`` copies ``collect_audio_files_and_labels0``: any capital
    ``F`` means fake. This is case-sensitive, so ``fake.wav`` is real and
    ``myFile.wav`` is fake. Prefer ``filename`` or class folders.

    ``filename`` looks for fake/real words and returns ``None`` when it finds none.
    """
    if scheme == "notebook":
        return FAKE if "fake" in filename else REAL
    if scheme == "notebook-f":
        return FAKE if "F" in filename else REAL
    if scheme != "filename":
        raise ValueError(f"Unknown filename scheme: {scheme}")

    stem = Path(filename).stem
    kinds: set[str] = set()
    whole = label_kind(stem)
    if whole:
        kinds.add(whole)
    for part in re.split(r"[^A-Za-z0-9]+", stem):
        kind = label_kind(part)
        if kind:
            kinds.add(kind)
    if "fake" in kinds:
        return FAKE
    if "real" in kinds:
        return REAL
    return None
