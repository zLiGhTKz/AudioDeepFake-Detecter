"""Model card written next to a training run. It never invents accuracy numbers."""

from __future__ import annotations

from pathlib import Path


def render_model_card(
    *,
    base_model: str,
    max_seconds: float,
    sample_rate: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    train_summary: dict[str, int],
    eval_summary: dict[str, int] | None,
) -> str:
    eval_line = "No evaluation split was passed to this run."
    if eval_summary is not None:
        eval_line = (
            f"Eval clips: {eval_summary['total']} "
            f"({eval_summary['fake']} fake / {eval_summary['real']} real)."
        )
    train_counts = (
        f"{train_summary['total']} ({train_summary['fake']} fake / {train_summary['real']} real)"
    )
    return f"""---
language:
- en
tags:
- audio-classification
- deepfake-detection
- speech
base_model: {base_model}
---

# DistilHuBERT audio deepfake detector

Classifier fine-tuned from `{base_model}`.

Label 0 is `fake` (AI-synthesized speech). Label 1 is `real` (genuine speech).
The fake score is the model's probability for label 0.

This card does not include accuracy numbers. Measure a held-out set with
`audiodeepfake-eval` and paste those results here yourself.

## Training setup

- Audio is resampled to {sample_rate} Hz.
- Each training clip uses the first {max_seconds:g} seconds. Shorter clips are padded with zeros.
- Epochs: {epochs}
- Batch size: {batch_size}
- Learning rate: {learning_rate:g}
- Train clips: {train_counts}.
- {eval_line}

## Use

```python
from audiodeepfake import detect

result = detect("speech.wav", model="your-name/your-model")
print(result.label, result.score)
```

## Limitations

A score is not proof that a speaker is real or synthetic. Unseen generators,
codecs, noise, languages, and recording setups can make the model wrong.
Check the license of `{base_model}` and of every dataset you trained on before
you share the weights.
"""


def write_model_card(directory: Path, **kwargs) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "README.md"
    destination.write_text(render_model_card(**kwargs), encoding="utf-8")
    return destination
