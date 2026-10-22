from pathlib import Path

import numpy as np
import pytest

from audiodeepfake.audio import (
    collect_audio_files,
    expand_inputs,
    load_audio,
    segment_waveform,
)
from audiodeepfake.errors import AudioLoadError
from tests.fakes import write_wav


def test_segments_drop_a_short_tail_and_pad_a_short_clip():
    full = segment_waveform(np.ones(16000, dtype=np.float32))
    assert len(full) == 1
    assert full[0].shape == (16000,)

    tail = segment_waveform(np.ones(16000 + 100, dtype=np.float32))
    assert len(tail) == 1

    two = segment_waveform(np.ones(32000, dtype=np.float32))
    assert len(two) == 2

    first = segment_waveform(np.ones(32000, dtype=np.float32), mode="first")
    assert len(first) == 1

    short = segment_waveform(np.ones(8000, dtype=np.float32))
    assert len(short) == 1
    assert short[0].shape == (16000,)
    assert np.count_nonzero(short[0][8000:]) == 0


def test_collect_audio_files_skips_hidden_and_rejects_unknown_types(tmp_path: Path):
    (tmp_path / "keep.wav").write_bytes(b"")
    (tmp_path / "song.mp3").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    (tmp_path / ".secret.wav").write_bytes(b"")
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "nope.wav").write_bytes(b"")

    found = collect_audio_files(tmp_path)
    assert [path.name for path in found] == ["keep.wav", "song.mp3"]

    with pytest.raises(AudioLoadError):
        collect_audio_files(tmp_path / "notes.txt")
    with pytest.raises(AudioLoadError):
        collect_audio_files(tmp_path / "missing.wav")


def test_expand_inputs_distinguishes_file_and_folder(tmp_path: Path):
    audio = write_wav(tmp_path / "one.wav", np.zeros(1000))
    files, many = expand_inputs(audio)
    assert files == [audio]
    assert many is False

    folder = tmp_path / "folder"
    write_wav(folder / "a.wav", np.zeros(1000))
    listed, many = expand_inputs(folder)
    assert len(listed) == 1
    assert many is True

    batch, many = expand_inputs([audio])
    assert batch == [audio]
    assert many is True


def test_load_wav_resamples_to_16k(tmp_path: Path):
    path = write_wav(tmp_path / "tone.wav", np.linspace(-0.2, 0.2, 16000))
    audio = load_audio(path, 16000)
    assert audio.shape == (16000,)
    assert audio.dtype == np.float32


def test_load_rejects_garbage(tmp_path: Path):
    path = tmp_path / "bad.wav"
    path.write_text("this is not audio", encoding="utf-8")
    with pytest.raises(AudioLoadError):
        load_audio(path)
