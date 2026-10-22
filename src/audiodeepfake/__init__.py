"""Detect AI-synthesized speech with a fine-tuned DistilHuBERT classifier."""

from audiodeepfake.detector import Detector, detect
from audiodeepfake.errors import ModelNotConfigured, UntrainedModelError
from audiodeepfake.results import DetectionResult, SegmentResult

__version__ = "0.1.0"

__all__ = [
    "DetectionResult",
    "Detector",
    "ModelNotConfigured",
    "SegmentResult",
    "UntrainedModelError",
    "__version__",
    "detect",
]
