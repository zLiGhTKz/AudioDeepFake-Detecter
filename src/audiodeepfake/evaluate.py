"""Score a labeled folder with the same windows the detector uses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from audiodeepfake.data import FORMATS, SplitSpec, load_split, summarize
from audiodeepfake.errors import AudioDeepfakeError
from audiodeepfake.labels import LABEL2ID
from audiodeepfake.metrics import binary_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiodeepfake-eval",
        description=(
            "Measure a fine-tuned model on a labeled set. "
            "Prints the scores from this run only. Nothing is filled in for you."
        ),
    )
    parser.add_argument("--model", "-m", required=True, help="Fine-tuned model id or local folder.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--format", default="auto", choices=FORMATS)
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--audio-dir", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument("--real-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--segment-seconds", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--window",
        action="store_true",
        help="Score only the first window, the same crop training uses.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def score_samples(detector, samples, mode: str = "all") -> dict:
    """Run ``detector`` on labeled samples and return metrics plus per-file rows."""
    y_true = []
    y_pred = []
    rows = []
    for sample in samples:
        result = detector.detect_file(sample.path, mode=mode)
        predicted = LABEL2ID[result.label]
        y_true.append(sample.label)
        y_pred.append(predicted)
        rows.append(
            {
                "path": str(sample.path),
                "true_label": "fake" if sample.label == 0 else "real",
                "predicted_label": result.label,
                "fake_score": result.score,
            }
        )
    return {
        "metrics": binary_metrics(y_true, y_pred),
        "counts": summarize(samples),
        "files": rows,
    }


def main(argv: list[str] | None = None) -> int:
    from audiodeepfake.detector import Detector

    args = build_parser().parse_args(argv)
    spec = SplitSpec(
        root=args.data_dir,
        format=args.format,
        protocol=args.protocol,
        audio_dir=args.audio_dir,
        csv_path=args.csv,
        real_dir=args.real_dir,
        audio_root=args.audio_root,
    )
    try:
        if spec.root is None and spec.protocol is None and spec.csv_path is None:
            raise AudioDeepfakeError("Pass --data-dir, --protocol, or --csv.")
        split = load_split(spec)
        detector = Detector(
            model_id=args.model,
            threshold=args.threshold,
            segment_seconds=args.segment_seconds,
            device=args.device,
            batch_size=args.batch_size,
        )
        mode = "first" if args.window else "all"
        report = score_samples(detector, split.samples, mode=mode)
    except AudioDeepfakeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report["format"] = split.format
    report["mode"] = "first" if args.window else "segments"
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"format: {report['format']}")
        print(f"mode: {report['mode']}")
        counts = report["counts"]
        print(f"clips: {counts['total']} ({counts['fake']} fake / {counts['real']} real)")
        for key, value in report["metrics"].items():
            if key == "num_samples":
                print(f"{key}: {int(value)}")
            else:
                print(f"{key}: {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
