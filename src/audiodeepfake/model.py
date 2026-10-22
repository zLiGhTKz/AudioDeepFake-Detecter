"""Load a fine-tuned classifier, or a DistilHuBERT backbone for training."""

from __future__ import annotations

import os
from pathlib import Path

from audiodeepfake.errors import (
    MISSING_MODEL_HELP,
    AudioDeepfakeError,
    ModelNotConfigured,
    UntrainedModelError,
)
from audiodeepfake.labels import ID2LABEL, LABEL2ID

DEFAULT_BACKBONE = "ntu-spml/distilhubert"
ENV_MODEL_ID = "AUDIODEEPFAKE_MODEL"

IMPLICIT_LABEL_WARNING = (
    "This checkpoint has no fake/real label names. Scores assume index 0 is "
    "fake and index 1 is real, which is what audiodeepfake-train writes. "
    "If this model was trained with the opposite order, the label will be swapped."
)


def resolve_model_id(model_id: str | None) -> str:
    """Use the argument, then ``AUDIODEEPFAKE_MODEL``. Empty means unset."""
    if model_id is not None and str(model_id).strip():
        return str(model_id).strip()
    env_value = os.environ.get(ENV_MODEL_ID, "").strip()
    if env_value:
        return env_value
    raise ModelNotConfigured(MISSING_MODEL_HELP)


def is_base_backbone(model_id: str) -> bool:
    """True for the public DistilHuBERT checkpoint, which has no classifier."""
    name = model_id.strip().rstrip("/")
    for prefix in ("https://huggingface.co/", "http://huggingface.co/"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name.lower() == DEFAULT_BACKBONE


def ensure_finetuned_architecture(model_id: str, architectures: list[str] | None) -> None:
    """Reject the base model and any checkpoint that is not a classifier."""
    if is_base_backbone(model_id):
        raise UntrainedModelError(
            f"{model_id} is the DistilHuBERT backbone, not a deepfake detector. "
            "Train a classifier and pass the fine-tuned repo id. " + MISSING_MODEL_HELP
        )
    names = list(architectures or [])
    if not names or not any("Classification" in name for name in names):
        shown = ", ".join(names) if names else "none"
        raise UntrainedModelError(
            f"{model_id} is not an audio classification model (architectures: {shown}). "
            "Train with audiodeepfake-train and load the saved folder or the Hub repo "
            "you publish from it."
        )


def load_finetuned(model_id: str, device: str | None = None):
    """Download a fine-tuned classifier and its feature extractor.

    The config is checked before the weights download so the base DistilHuBERT
    checkpoint is not loaded with a random classification head.
    """
    from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForAudioClassification

    config = AutoConfig.from_pretrained(model_id)
    ensure_finetuned_architecture(model_id, list(getattr(config, "architectures", None) or []))
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    model.eval()
    if device is not None:
        model.to(device)
    return model, feature_extractor


def load_backbone(model_name: str = DEFAULT_BACKBONE):
    """Start a 2-class classifier from a speech backbone such as DistilHuBERT."""
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=dict(ID2LABEL),
        label2id=dict(LABEL2ID),
        ignore_mismatched_sizes=True,
    )
    return model, feature_extractor


def load_checkpoint(model, path: str | Path):
    """Load weights from a ``.bin`` / ``.pt`` file or a ``save_pretrained`` folder.

    Notebook checkpoints may be a raw ``state_dict`` or a dict with
    ``model_state_dict``. Optimizer state in those files is ignored here.
    """
    import torch

    checkpoint = Path(path)
    if not checkpoint.exists():
        raise AudioDeepfakeError(f"Checkpoint does not exist: {checkpoint}")

    if checkpoint.is_dir():
        from transformers import AutoModelForAudioClassification

        loaded = AutoModelForAudioClassification.from_pretrained(checkpoint)
        model.load_state_dict(loaded.state_dict())
        return None

    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file

        model.load_state_dict(load_file(checkpoint))
        return None

    try:
        blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception:
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)

    epoch = None
    if isinstance(blob, dict) and "model_state_dict" in blob:
        state = blob["model_state_dict"]
        epoch = blob.get("epoch")
    else:
        state = blob
    model.load_state_dict(state)
    return epoch
