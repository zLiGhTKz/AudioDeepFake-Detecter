"""Detection results returned by the public API."""

from __future__ import annotations

from dataclasses import asdict, dataclass


def _round(value: float) -> float:
    return round(float(value), 6)


@dataclass(frozen=True)
class SegmentResult:
    """Score for one window of a longer file."""

    index: int
    start_seconds: float
    end_seconds: float
    label: str
    fake_score: float
    real_score: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["start_seconds"] = _round(self.start_seconds)
        payload["end_seconds"] = _round(self.end_seconds)
        payload["fake_score"] = _round(self.fake_score)
        payload["real_score"] = _round(self.real_score)
        return payload


@dataclass(frozen=True)
class DetectionResult:
    """File-level deepfake score.

    ``label`` is ``fake`` or ``real``. ``score`` is the mean probability of the
    fake class across windows. Higher means the clip sounds more synthetic.
    A clip is labeled fake when ``score >= threshold``.
    """

    path: str
    label: str
    score: float
    fake_score: float
    real_score: float
    threshold: float
    num_segments: int
    segments: tuple[SegmentResult, ...]

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "label": self.label,
            "score": _round(self.score),
            "fake_score": _round(self.fake_score),
            "real_score": _round(self.real_score),
            "threshold": _round(self.threshold),
            "num_segments": self.num_segments,
            "segments": [segment.to_dict() for segment in self.segments],
        }
