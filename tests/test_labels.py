import pytest

from audiodeepfake.errors import UntrainedModelError
from audiodeepfake.labels import label_from_filename, normalize_label, resolve_label_indices
from audiodeepfake.model import (
    DEFAULT_MODEL_ID,
    ensure_finetuned_architecture,
    is_base_backbone,
    resolve_model_id,
)


def test_normalize_public_dataset_labels():
    assert normalize_label("spoof") == 0
    assert normalize_label("bona-fide") == 1
    assert normalize_label("bonafide") == 1
    assert normalize_label("0") == 0
    assert normalize_label("1") == 1
    assert normalize_label("target") is None


def test_notebook_filename_rules_are_case_sensitive():
    assert label_from_filename("clip_fake_01.wav", "notebook") == 0
    assert label_from_filename("FAKE.wav", "notebook") == 1
    assert label_from_filename("speaker.wav", "notebook") == 1
    assert label_from_filename("fake.wav", "notebook-f") == 1
    assert label_from_filename("myFile.wav", "notebook-f") == 0
    assert label_from_filename("clip.wav", "notebook-f") == 1


def test_filename_tokens():
    assert label_from_filename("interview-spoof.wav", "filename") == 0
    assert label_from_filename("bonafide_03.wav", "filename") == 1
    assert label_from_filename("lecture.wav", "filename") is None
    assert label_from_filename("real-spoof-mix.wav", "filename") == 0


def test_label_indexes_follow_config_names():
    fake_index, real_index, explicit = resolve_label_indices({0: "real", 1: "fake"})
    assert (fake_index, real_index, explicit) == (1, 0, True)
    fake_index, real_index, explicit = resolve_label_indices({"0": "LABEL_0", "1": "LABEL_1"})
    assert (fake_index, real_index, explicit) == (0, 1, False)


def test_default_model_and_backbone_rejection(monkeypatch):
    monkeypatch.delenv("AUDIODEEPFAKE_MODEL", raising=False)
    assert resolve_model_id(None) == DEFAULT_MODEL_ID
    monkeypatch.setenv("AUDIODEEPFAKE_MODEL", "   ")
    assert resolve_model_id(None) == DEFAULT_MODEL_ID
    monkeypatch.setenv("AUDIODEEPFAKE_MODEL", "example/detector")
    assert resolve_model_id(None) == "example/detector"
    assert resolve_model_id(" other/model ") == "other/model"
    assert is_base_backbone("https://huggingface.co/ntu-spml/distilhubert")
    with pytest.raises(UntrainedModelError):
        ensure_finetuned_architecture("ntu-spml/distilhubert", ["HubertForSequenceClassification"])
    with pytest.raises(UntrainedModelError):
        ensure_finetuned_architecture("example/encoder", ["HubertModel"])
    ensure_finetuned_architecture("example/detector", ["HubertForSequenceClassification"])
