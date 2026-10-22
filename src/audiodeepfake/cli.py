"""``audiodeepfake-detect`` command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from audiodeepfake import __version__
from audiodeepfake.audio import collect_audio_files
from audiodeepfake.detector import Detector
from audiodeepfake.errors import AudioDeepfakeError
from audiodeepfake.model import DEFAULT_MODEL_ID
from audiodeepfake.results import DetectionResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiodeepfake-detect",
        description=(
            "Score speech as real or AI-synthesized. Directories are searched recursively."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Audio files or folders. wav, flac, and ogg work directly. mp3 and m4a need ffmpeg.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help=(
            "Hugging Face model id or a local model folder. "
            f"Default: {DEFAULT_MODEL_ID}. "
            "Override with this flag or AUDIODEEPFAKE_MODEL."
        ),
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.5,
        help=(
            "Label a clip fake when its mean fake probability is at least this value "
            "(default: 0.5)."
        ),
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=1.0,
        help=(
            "Window length in seconds. Use the same value the model was trained with (default: 1)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Windows per forward pass.")
    parser.add_argument("--device", default=None, help="Torch device, for example cpu or cuda.")
    parser.add_argument("--json", action="store_true", help="Print one JSON object.")
    parser.add_argument(
        "--fail-on-fake",
        action="store_true",
        help="Exit with status 3 when any file is labeled fake.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def build_detector(args: argparse.Namespace) -> Detector:
    return Detector(
        model_id=args.model,
        threshold=args.threshold,
        segment_seconds=args.segment_seconds,
        device=args.device,
        batch_size=args.batch_size,
    )


def format_table(results: list[DetectionResult]) -> str:
    lines = ["path\tlabel\tfake_score\tsegments"]
    for result in results:
        lines.append(f"{result.path}\t{result.label}\t{result.score:.4f}\t{result.num_segments}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        files: list[Path] = []
        for raw in args.inputs:
            files.extend(collect_audio_files(Path(raw)))
        detector = build_detector(args)
        results: list[DetectionResult] = []
        failures: list[str] = []
        for audio_path in files:
            try:
                results.append(detector.detect_file(audio_path))
            except AudioDeepfakeError as exc:
                failures.append(f"{audio_path}: {exc}")
    except AudioDeepfakeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        payload = {
            "results": [result.to_dict() for result in results],
            "errors": failures,
        }
        print(json.dumps(payload, indent=2))
    else:
        if results:
            print(format_table(results))
        for failure in failures:
            print(f"error: {failure}", file=sys.stderr)

    if failures:
        return 1
    if args.fail_on_fake and any(result.label == "fake" for result in results):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
