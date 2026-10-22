from pathlib import Path

import pytest

from audiodeepfake.data import Sample, SplitSpec, balance_undersample, load_split, summarize
from audiodeepfake.errors import DatasetLayoutError
from audiodeepfake.labels import FAKE, REAL


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_directory_and_filename_layouts(tmp_path: Path):
    _touch(tmp_path / "real" / "a.wav")
    _touch(tmp_path / "Fake" / "b.flac")
    split = load_split(SplitSpec(root=tmp_path, format="auto"))
    assert split.format == "directory"
    counts = summarize(split.samples)
    assert counts == {"total": 2, "fake": 1, "real": 1}

    names = tmp_path / "names"
    _touch(names / "interview-spoof.wav")
    _touch(names / "bonafide-01.wav")
    named = load_split(SplitSpec(root=names, format="auto"))
    assert named.format == "filename"
    assert summarize(named.samples)["fake"] == 1


def test_notebook_f_quirk(tmp_path: Path):
    _touch(tmp_path / "fake.wav")
    _touch(tmp_path / "myFile.wav")
    split = load_split(SplitSpec(root=tmp_path, format="notebook-f"))
    labels = {sample.path.name: sample.label for sample in split.samples}
    assert labels["fake.wav"] == REAL
    assert labels["myFile.wav"] == FAKE


def test_unlabeled_filename_is_an_error(tmp_path: Path):
    _touch(tmp_path / "lecture.wav")
    with pytest.raises(DatasetLayoutError):
        load_split(SplitSpec(root=tmp_path, format="filename"))


def test_asvspoof_protocol(tmp_path: Path):
    audio = tmp_path / "flac"
    _touch(audio / "LA_T_1.flac")
    _touch(audio / "LA_T_2.flac")
    protocol = tmp_path / "ASVspoof2019.LA.cm.train.trn.txt"
    protocol.write_text(
        "LA_0079 LA_T_1 - - bonafide\nLA_0079 LA_T_2 - A01 spoof\n",
        encoding="utf-8",
    )
    split = load_split(SplitSpec(root=tmp_path, format="asvspoof"))
    labels = {sample.path.stem: sample.label for sample in split.samples}
    assert labels == {"LA_T_1": REAL, "LA_T_2": FAKE}

    short = tmp_path / "keys.txt"
    short.write_text("LA_T_1 spoof\nLA_T_2 bonafide\n", encoding="utf-8")
    # Two protocols in one tree must not be merged silently.
    with pytest.raises(DatasetLayoutError):
        load_split(SplitSpec(root=tmp_path, format="asvspoof"))

    explicit = load_split(SplitSpec(protocol=short, audio_dir=audio, format="asvspoof"))
    labels = {sample.path.stem: sample.label for sample in explicit.samples}
    assert labels["LA_T_1"] == FAKE
    assert labels["LA_T_2"] == REAL


def test_duplicate_audio_stems_are_rejected(tmp_path: Path):
    _touch(tmp_path / "a.wav")
    _touch(tmp_path / "nested" / "a.flac")
    protocol = tmp_path / "protocol.txt"
    protocol.write_text("a bonafide\n", encoding="utf-8")
    with pytest.raises(DatasetLayoutError):
        load_split(SplitSpec(protocol=protocol, audio_dir=tmp_path))


def test_in_the_wild_csv(tmp_path: Path):
    root = tmp_path / "release_in_the_wild"
    _touch(root / "0.wav")
    _touch(root / "1.wav")
    (root / "meta.csv").write_text(
        "file,speaker,label\n0.wav,Ada,spoof\n1.wav,Ada,bona-fide\n",
        encoding="utf-8",
    )
    split = load_split(SplitSpec(root=tmp_path, format="auto"))
    assert split.format == "in-the-wild"
    labels = {sample.path.name: sample.label for sample in split.samples}
    assert labels == {"0.wav": FAKE, "1.wav": REAL}


def test_wavefake_needs_a_real_directory(tmp_path: Path):
    generated = tmp_path / "ljspeech_melgan"
    _touch(generated / "utt.wav")
    with pytest.raises(DatasetLayoutError, match="LJSpeech"):
        load_split(SplitSpec(root=tmp_path, format="auto"))

    real = tmp_path / "real_wavs"
    _touch(real / "real.wav")
    # auto sees the vocoder folder and selects wavefake, but real_dir is separate.
    split = load_split(SplitSpec(root=tmp_path, format="wavefake", real_dir=real))
    assert summarize(split.samples) == {"total": 2, "fake": 1, "real": 1}


def test_wavefake_auto_uses_a_real_folder(tmp_path: Path):
    _touch(tmp_path / "ljspeech_hifiGAN" / "a.wav")
    _touch(tmp_path / "real" / "b.wav")
    split = load_split(SplitSpec(root=tmp_path, format="auto"))
    assert split.format == "wavefake"
    assert summarize(split.samples) == {"total": 2, "fake": 1, "real": 1}


def test_balance_is_repeatable():
    samples = [Sample(Path(f"f{i}.wav"), FAKE) for i in range(4)]
    samples += [Sample(Path(f"r{i}.wav"), REAL) for i in range(2)]
    first = balance_undersample(samples, seed=123)
    second = balance_undersample(samples, seed=123)
    assert first == second
    assert summarize(first) == {"total": 4, "fake": 2, "real": 2}
