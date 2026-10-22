"""Tiny stand-ins for DistilHuBERT so tests never download weights."""

from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def write_wav(path: Path, samples, sample_rate: int = 16000) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return path


class ConstantModel(torch.nn.Module):
    def __init__(self, logits=(2.0, -2.0), id2label=None):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor(list(logits), dtype=torch.float32))
        labels = {0: "fake", 1: "real"} if id2label is None else id2label
        self.config = SimpleNamespace(
            id2label=labels,
            label2id={str(value): key for key, value in labels.items()},
        )

    def forward(self, input_values=None, attention_mask=None, labels=None, **kwargs):
        batch = input_values.shape[0]
        logits = self.logits.to(device=input_values.device, dtype=input_values.dtype)
        return SimpleNamespace(logits=logits.view(1, -1).expand(batch, -1))


class FirstSampleModel(torch.nn.Module):
    """Fake when the first sample is positive, real when it is negative."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            id2label={0: "fake", 1: "real"},
            label2id={"fake": 0, "real": 1},
        )

    def forward(self, input_values=None, attention_mask=None, labels=None, **kwargs):
        first = input_values[:, 0]
        fake = first * 20
        real = -first * 20
        return SimpleNamespace(logits=torch.stack([fake, real], dim=-1))


class IdentityExtractor:
    sampling_rate = 16000

    def __call__(
        self,
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=None,
        max_length=None,
        truncation=None,
        **kwargs,
    ):
        if isinstance(audio, (list, tuple)):
            rows = [self._one(item, max_length) for item in audio]
            return {"input_values": torch.stack(rows, dim=0)}
        return {"input_values": self._one(audio, max_length).unsqueeze(0)}

    @staticmethod
    def _one(audio, max_length):
        array = np.asarray(audio, dtype=np.float32).reshape(-1)
        if max_length is not None:
            length = int(max_length)
            if array.size < length:
                array = np.pad(array, (0, length - array.size))
            elif array.size > length:
                array = array[:length]
        return torch.tensor(array, dtype=torch.float32)
