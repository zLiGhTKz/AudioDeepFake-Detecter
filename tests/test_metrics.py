import pytest

from audiodeepfake.metrics import binary_metrics


def test_binary_metrics_for_the_fake_class():
    scores = binary_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert scores["accuracy"] == pytest.approx(0.75)
    assert scores["precision_fake"] == pytest.approx(1.0)
    assert scores["recall_fake"] == pytest.approx(0.5)
    assert scores["f1_fake"] == pytest.approx(2 / 3)
    assert scores["num_samples"] == 4


def test_no_predicted_fakes():
    scores = binary_metrics([0, 1], [1, 1])
    assert scores["precision_fake"] == 0
    assert scores["recall_fake"] == 0
    assert scores["f1_fake"] == 0


def test_empty_metrics_fail():
    with pytest.raises(ValueError):
        binary_metrics([], [])
