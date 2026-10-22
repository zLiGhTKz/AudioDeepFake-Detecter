import json
from pathlib import Path

import numpy as np
import pytest

from audiodeepfake import DetectionResult, detect
from audiodeepfake.detector import Detector
from audiodeepfake.model import load_checkpoint
from tests.fakes import ConstantModel, FirstSampleModel, IdentityExtractor, write_wav


def _detector(model=None, **kwargs) -> Detector:
    return Detector(
        model=model or ConstantModel(),
        feature_extractor=IdentityExtractor(),
        **kwargs,
    )


def test_constant_model_scores_a_file(tmp_path: Path):
    path = write_wav(tmp_path / "clip.wav", np.zeros(16000))
    result = _detector().detect_file(path)
    assert isinstance(result, DetectionResult)
    assert result.label == "fake"
    assert result.score > 0.8
    assert result.num_segments == 1
    assert result.to_dict()["label"] == "fake"
    json.dumps(result.to_dict())


def test_threshold_tie_is_fake(tmp_path: Path):
    path = write_wav(tmp_path / "clip.wav", np.zeros(16000))
    result = _detector(ConstantModel(logits=(0.0, 0.0)), threshold=0.5).detect_file(path)
    assert result.score == pytest.approx(0.5)
    assert result.label == "fake"
    real = _detector(ConstantModel(logits=(0.0, 0.0)), threshold=0.51).detect_file(path)
    assert real.label == "real"


def test_id2label_order_is_respected(tmp_path: Path):
    path = write_wav(tmp_path / "clip.wav", np.zeros(16000))
    model = ConstantModel(logits=(5.0, -5.0), id2label={0: "real", 1: "fake"})
    result = _detector(model).detect_file(path)
    assert result.label == "real"
    assert result.fake_score < 0.01


def test_placeholder_labels_warn(tmp_path: Path):
    path = write_wav(tmp_path / "clip.wav", np.zeros(8000))
    model = ConstantModel(id2label={0: "LABEL_0", 1: "LABEL_1"})
    with pytest.warns(UserWarning, match="index 0 is fake"):
        detector = _detector(model)
    result = detector.detect_file(path)
    assert result.num_segments == 1
    assert result.segments[0].end_seconds == pytest.approx(0.5)


def test_windows_are_averaged_and_tail_is_dropped(tmp_path: Path):
    samples = np.concatenate([np.full(16000, 0.8), np.full(16000, -0.8), np.full(50, 0.8)])
    path = write_wav(tmp_path / "long.wav", samples)
    result = _detector(FirstSampleModel()).detect_file(path)
    assert result.num_segments == 2
    assert result.segments[0].label == "fake"
    assert result.segments[1].label == "real"
    assert result.score == pytest.approx(0.5, abs=1e-3)
    assert result.label == "fake"

    first = _detector(FirstSampleModel()).detect_file(path, mode="first")
    assert first.num_segments == 1
    assert first.label == "fake"


def test_folder_and_public_detect_helper(monkeypatch, tmp_path: Path):
    folder = tmp_path / "clips"
    write_wav(folder / "a.wav", np.zeros(16000))
    write_wav(folder / "b.wav", np.ones(16000))

    def fake_load(model_id, device=None):
        assert model_id == "example/detector"
        return ConstantModel(logits=(-2.0, 2.0)), IdentityExtractor()

    monkeypatch.setattr("audiodeepfake.detector.load_finetuned", fake_load)
    one = detect(folder / "a.wav", model="example/detector")
    assert isinstance(one, DetectionResult)
    assert one.label == "real"

    many = detect(folder, model="example/detector")
    assert isinstance(many, list)
    assert len(many) == 2
    assert {item.label for item in many} == {"real"}


def test_checkpoint_roundtrip(tmp_path: Path):
    model = ConstantModel(logits=(0.0, 0.0))
    path = tmp_path / "notebook.bin"
    import torch

    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": {"skip": 1}}, path)
    with torch.no_grad():
        model.logits.fill_(4)
    load_checkpoint(model, path)
    assert torch.allclose(model.logits.detach().cpu(), torch.zeros(2))

    raw = tmp_path / "raw.pt"
    torch.save(model.state_dict(), raw)
    with torch.no_grad():
        model.logits.fill_(1)
    load_checkpoint(model, raw)
    assert torch.allclose(model.logits.detach().cpu(), torch.zeros(2))
