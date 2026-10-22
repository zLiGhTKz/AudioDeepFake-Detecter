import json
import logging
from pathlib import Path

import numpy as np
import pytest

from audiodeepfake.card import render_model_card
from audiodeepfake.data import Sample
from audiodeepfake.detector import Detector
from audiodeepfake.evaluate import score_samples
from audiodeepfake.training import AudioDataset, _training_arguments, build_parser
from audiodeepfake.training import main as train_main
from tests.fakes import ConstantModel, IdentityExtractor, write_wav


def test_parser_keeps_notebook_hyperparameters():
    args = build_parser().parse_args(["--train-dir", "data", "--dry-run"])
    assert args.epochs == 5
    assert args.batch_size == 4
    assert args.learning_rate == 5e-5
    assert args.weight_decay == 0.01
    assert args.warmup_steps == 500
    assert args.max_seconds == 1.0
    assert args.base_model == "ntu-spml/distilhubert"
    assert args.seed == 123


def test_dry_run_counts_and_stops(tmp_path: Path, capsys):
    root = tmp_path / "set"
    (root / "real" / "a.wav").parent.mkdir(parents=True)
    (root / "real" / "a.wav").write_bytes(b"")
    (root / "fake" / "b.wav").parent.mkdir(parents=True)
    (root / "fake" / "b.wav").write_bytes(b"")
    (root / "fake" / "c.wav").write_bytes(b"")
    code = train_main(["--train-dir", str(root), "--format", "directory", "--dry-run", "--balance"])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["format"] == "directory"
    assert payload["train"] == {"total": 2, "fake": 1, "real": 1}


def test_dry_run_rejects_one_class(tmp_path: Path, capsys):
    root = tmp_path / "onlyfake" / "fake"
    root.mkdir(parents=True)
    (root / "a.wav").write_bytes(b"")
    code = train_main(
        ["--train-dir", str(tmp_path / "onlyfake"), "--format", "directory", "--dry-run"]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "both real and fake" in captured.err
    assert json.loads(captured.out)["train"]["real"] == 0


def test_dataset_window_and_bad_file(tmp_path: Path, caplog):
    path = write_wav(tmp_path / "long.wav", np.ones(20000))
    dataset = AudioDataset([Sample(path, 0)], IdentityExtractor(), 16000, 16000)
    item = dataset[0]
    assert tuple(item["input_values"].shape) == (16000,)
    assert float(item["input_values"][0]) > 0
    assert item["labels"] == 0

    bad = tmp_path / "bad.wav"
    bad.write_text("nope", encoding="utf-8")
    broken = AudioDataset([Sample(bad, 1)], IdentityExtractor(), 16000, 8)
    with caplog.at_level(logging.WARNING):
        replacement = broken[0]
    assert tuple(replacement["input_values"].shape) == (8,)
    assert float(replacement["input_values"].abs().sum()) == 0
    assert "Could not read" in caplog.text


def test_training_argument_rename():
    class OldArguments:
        def __init__(self, **kwargs):
            if "eval_strategy" in kwargs:
                raise TypeError("old transformers")
            self.kwargs = kwargs

    args = _training_arguments(OldArguments, {"output_dir": "out", "eval_strategy": "epoch"})
    assert args.kwargs["evaluation_strategy"] == "epoch"

    class NewArguments:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    args = _training_arguments(NewArguments, {"eval_strategy": "no"})
    assert args.kwargs["eval_strategy"] == "no"


def test_model_card_has_no_accuracy_numbers(tmp_path: Path):
    text = render_model_card(
        base_model="ntu-spml/distilhubert",
        max_seconds=1,
        sample_rate=16000,
        epochs=5,
        learning_rate=5e-5,
        batch_size=4,
        train_summary={"total": 10, "fake": 4, "real": 6},
        eval_summary=None,
    )
    assert "does not include accuracy" in text
    assert "%" not in text
    assert "94.8" not in text
    assert "61.7" not in text


def test_score_samples_uses_the_detector(tmp_path: Path):
    fake = write_wav(tmp_path / "spoof.wav", np.ones(16000))
    real = write_wav(tmp_path / "real.wav", np.zeros(16000))
    detector = Detector(model=ConstantModel(), feature_extractor=IdentityExtractor())
    report = score_samples(detector, [Sample(fake, 0), Sample(real, 1)])
    assert report["metrics"]["accuracy"] == pytest.approx(0.5)
    assert report["metrics"]["recall_fake"] == pytest.approx(1.0)
    assert report["files"][0]["predicted_label"] == "fake"
    assert report["files"][1]["predicted_label"] == "fake"
