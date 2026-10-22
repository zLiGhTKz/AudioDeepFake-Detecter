"""Binary scores for the fake class (label 0), without a scikit-learn dependency."""

from __future__ import annotations

from audiodeepfake.labels import FAKE


def binary_metrics(y_true, y_pred) -> dict[str, float]:
    """Accuracy plus precision, recall, and F1 for the fake class."""
    true = [int(value) for value in y_true]
    pred = [int(value) for value in y_pred]
    if not true:
        raise ValueError("No samples to score.")
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must be the same length.")

    pairs = list(zip(true, pred, strict=True))
    correct = sum(left == right for left, right in pairs)
    true_positive = sum(left == FAKE and right == FAKE for left, right in pairs)
    false_positive = sum(left != FAKE and right == FAKE for left, right in pairs)
    false_negative = sum(left == FAKE and right != FAKE for left, right in pairs)
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "accuracy": correct / len(true),
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1,
        "num_samples": float(len(true)),
    }
