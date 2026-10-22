import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from audiodeepfake.cli import main
from audiodeepfake.detector import Detector
from tests.fakes import ConstantModel, IdentityExtractor, write_wav


def _install(monkeypatch):
    def build(args):
        return Detector(
            model=ConstantModel(),
            feature_extractor=IdentityExtractor(),
            threshold=args.threshold,
            segment_seconds=args.segment_seconds,
            batch_size=args.batch_size,
        )

    monkeypatch.setattr("audiodeepfake.cli.build_detector", build)


def test_cli_json_and_fail_on_fake(monkeypatch, tmp_path: Path, capsys):
    _install(monkeypatch)
    path = write_wav(tmp_path / "clip.wav", np.zeros(16000))
    code = main([str(path), "--json", "--fail-on-fake"])
    captured = capsys.readouterr()
    assert code == 3
    payload = json.loads(captured.out)
    assert payload["results"][0]["label"] == "fake"
    assert payload["errors"] == []


def test_cli_table_for_a_real_score(monkeypatch, tmp_path: Path, capsys):
    def build(args):
        return Detector(
            model=ConstantModel(logits=(-2.0, 2.0)),
            feature_extractor=IdentityExtractor(),
            threshold=args.threshold,
        )

    monkeypatch.setattr("audiodeepfake.cli.build_detector", build)
    path = write_wav(tmp_path / "clip.wav", np.zeros(16000))
    code = main([str(path)])
    captured = capsys.readouterr()
    assert code == 0
    assert "real" in captured.out
    assert "fake_score" in captured.out


def test_cli_reports_unreadable_audio(monkeypatch, tmp_path: Path, capsys):
    _install(monkeypatch)
    folder = tmp_path / "clips"
    write_wav(folder / "good.wav", np.zeros(16000))
    (folder / "bad.wav").write_text("not audio", encoding="utf-8")
    code = main([str(folder), "--json"])
    captured = capsys.readouterr()
    assert code == 1
    payload = json.loads(captured.out)
    assert len(payload["results"]) == 1
    assert payload["errors"]


def test_cli_uses_the_published_default(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.delenv("AUDIODEEPFAKE_MODEL", raising=False)
    loaded: dict[str, str] = {}

    def fake_load(model_id, device=None):
        loaded["id"] = model_id
        return ConstantModel(logits=(-2.0, 2.0)), IdentityExtractor()

    monkeypatch.setattr("audiodeepfake.detector.load_finetuned", fake_load)
    path = write_wav(tmp_path / "clip.wav", np.zeros(1600))
    code = main([str(path)])
    captured = capsys.readouterr()
    assert code == 0
    assert loaded["id"] == "lightk/audiodeepfake-detecter-laogao"
    assert "real" in captured.out

    loaded.clear()
    monkeypatch.setenv("AUDIODEEPFAKE_MODEL", "example/other")
    code = main([str(path), "--model", "example/flag"])
    assert code == 0
    assert loaded["id"] == "example/flag"


def test_unknown_extension(tmp_path: Path, capsys):
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")
    code = main([str(path)])
    captured = capsys.readouterr()
    assert code == 2
    assert "supported" in captured.err


def test_help_commands():
    # Do not resolve the venv interpreter. It is often a symlink into /usr/bin,
    # and the console scripts live next to the symlink.
    bindir = Path(sys.executable).parent
    commands = [
        [str(bindir / "audiodeepfake-detect"), "--help"],
        [str(bindir / "audiodeepfake-train"), "--help"],
        [str(bindir / "audiodeepfake-eval"), "--help"],
        [sys.executable, "-m", "audiodeepfake", "--help"],
    ]
    for command in commands:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        assert "usage" in completed.stdout.lower()
