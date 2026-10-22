"""Score speech files with a fine-tuned DistilHuBERT classifier."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch

from audiodeepfake.audio import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SEGMENT_SECONDS,
    expand_inputs,
    load_audio,
    segment_waveform,
)
from audiodeepfake.labels import resolve_label_indices
from audiodeepfake.model import IMPLICIT_LABEL_WARNING, load_finetuned, resolve_model_id
from audiodeepfake.results import DetectionResult, SegmentResult


class Detector:
    """Reusable deepfake scorer.

    Pass ``model`` and ``feature_extractor`` in tests. Otherwise ``model_id``
    (or ``AUDIODEEPFAKE_MODEL``) must name a fine-tuned Hugging Face repo or a
    local ``save_pretrained`` folder.
    """

    def __init__(
        self,
        model_id: str | None = None,
        *,
        model=None,
        feature_extractor=None,
        device: str | None = None,
        segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
        threshold: float = 0.5,
        sample_rate: int | None = None,
        batch_size: int = 8,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        if segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self.model_id = model_id
        if model is None or feature_extractor is None:
            if model is not None or feature_extractor is not None:
                raise ValueError("Pass both model and feature_extractor, or neither.")
            self.model_id = resolve_model_id(model_id)
            if device is None:
                device = default_device()
            model, feature_extractor = load_finetuned(self.model_id, device=device)

        self.model = model
        self.feature_extractor = feature_extractor
        self.segment_seconds = float(segment_seconds)
        self.threshold = float(threshold)
        self.batch_size = int(batch_size)
        self.sample_rate = int(
            sample_rate or getattr(feature_extractor, "sampling_rate", None) or DEFAULT_SAMPLE_RATE
        )
        self.window = int(round(self.segment_seconds * self.sample_rate))
        self.device = _select_device(model, device)
        self.model.to(self.device)
        self.model.eval()

        id2label = getattr(getattr(model, "config", None), "id2label", None) or {
            0: "fake",
            1: "real",
        }
        self.fake_index, self.real_index, explicit = resolve_label_indices(id2label)
        if not explicit:
            warnings.warn(IMPLICIT_LABEL_WARNING, stacklevel=2)

    def detect_file(self, path: str | Path, mode: str = "all") -> DetectionResult:
        audio_path = Path(path)
        waveform = load_audio(audio_path, self.sample_rate)
        return self.detect_waveform(waveform, path=str(audio_path), mode=mode)

    def detect_waveform(
        self,
        waveform: np.ndarray,
        path: str = "",
        mode: str = "all",
    ) -> DetectionResult:
        audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
        segments = segment_waveform(
            audio,
            sample_rate=self.sample_rate,
            segment_seconds=self.segment_seconds,
            mode=mode,
        )
        probabilities = self._probabilities(segments)
        fake_scores = probabilities[:, self.fake_index]
        real_scores = probabilities[:, self.real_index]
        fake_score = float(fake_scores.mean())
        real_score = float(real_scores.mean())
        label = "fake" if fake_score >= self.threshold else "real"
        duration = float(audio.shape[0]) / self.sample_rate
        segment_results = []
        for index, (fake_value, real_value) in enumerate(
            zip(fake_scores, real_scores, strict=True)
        ):
            start = index * self.segment_seconds
            end = min(duration, start + self.segment_seconds)
            segment_label = "fake" if float(fake_value) >= self.threshold else "real"
            segment_results.append(
                SegmentResult(
                    index=index,
                    start_seconds=start,
                    end_seconds=end,
                    label=segment_label,
                    fake_score=float(fake_value),
                    real_score=float(real_value),
                )
            )
        return DetectionResult(
            path=path,
            label=label,
            score=fake_score,
            fake_score=fake_score,
            real_score=real_score,
            threshold=self.threshold,
            num_segments=len(segment_results),
            segments=tuple(segment_results),
        )

    def _probabilities(self, segments: list[np.ndarray]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, len(segments), self.batch_size):
            batch = segments[start : start + self.batch_size]
            values, attention_mask = self._encode(batch)
            kwargs = {"input_values": values.to(self.device)}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask.to(self.device)
            with torch.no_grad():
                outputs = self.model(**kwargs)
            chunks.append(torch.softmax(outputs.logits, dim=-1).detach().cpu())
        return torch.cat(chunks, dim=0).numpy()

    def _encode(self, segments: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor | None]:
        arrays = [np.asarray(segment, dtype=np.float32) for segment in segments]
        inputs = self.feature_extractor(
            arrays,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=self.window,
            truncation=True,
        )
        values = _as_batch(inputs["input_values"])
        mask = None
        if "attention_mask" in inputs and inputs["attention_mask"] is not None:
            mask = _as_batch(inputs["attention_mask"])
        return values, mask


def detect(
    path: str | Path | list | tuple,
    *,
    model: str | None = None,
    threshold: float = 0.5,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    device: str | None = None,
    batch_size: int = 8,
    mode: str = "all",
) -> DetectionResult | list[DetectionResult]:
    """Score a file, a folder, or a list of paths.

    A single file returns one :class:`DetectionResult`. A directory or a list
    returns a list. ``result.label`` is ``fake`` or ``real``. ``result.score``
    is the mean probability of the fake class.
    """
    files, many = expand_inputs(path)
    detector = Detector(
        model_id=model,
        threshold=threshold,
        segment_seconds=segment_seconds,
        device=device,
        batch_size=batch_size,
    )
    results = [detector.detect_file(audio_path, mode=mode) for audio_path in files]
    if many:
        return results
    return results[0]


def default_device() -> str:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _select_device(model, device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _as_batch(values) -> torch.Tensor:
    tensor = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
    if tensor.dim() == 3 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2:
        raise ValueError(f"Expected features with shape (batch, time), got {tuple(tensor.shape)}.")
    return tensor
