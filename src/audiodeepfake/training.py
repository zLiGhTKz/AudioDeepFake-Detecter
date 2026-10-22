"""Fine-tune DistilHuBERT for real-versus-fake speech.

Hyperparameters follow the original notebook unless you override them:
AdamW, learning rate 5e-5, weight decay 0.01, warmup 500 steps, 5 epochs,
batch size 4, and a 1 second window (16,000 samples at 16 kHz). The notebook
file name said "5s", but the feature extractor ``max_length`` was 16000.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from audiodeepfake.audio import DEFAULT_SAMPLE_RATE, DEFAULT_SEGMENT_SECONDS, load_audio
from audiodeepfake.card import write_model_card
from audiodeepfake.data import (
    FORMATS,
    SplitSpec,
    balance_undersample,
    load_split,
    overlapping_paths,
    summarize,
)
from audiodeepfake.errors import AudioDeepfakeError, AudioLoadError
from audiodeepfake.metrics import binary_metrics
from audiodeepfake.model import DEFAULT_BACKBONE, load_backbone, load_checkpoint

logger = logging.getLogger(__name__)


class AudioDataset:
    """One padded or truncated window per file, matching the notebook dataset.

    This is a map-style dataset (``__len__`` and ``__getitem__``). It is created
    without importing torch until a sample is read, so ``--dry-run`` stays light.
    ``run_training`` wraps it for the Trainer.
    """

    def __init__(self, samples, feature_extractor, sample_rate: int, max_length: int):
        self.samples = list(samples)
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        import torch

        sample = self.samples[index]
        try:
            waveform = load_audio(sample.path, self.sample_rate)[: self.max_length]
            inputs = self.feature_extractor(
                waveform,
                sampling_rate=self.sample_rate,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            values = inputs["input_values"]
            while hasattr(values, "dim") and values.dim() > 1 and values.shape[0] == 1:
                values = values.squeeze(0)
        except AudioLoadError as exc:
            logger.warning("%s. Using silence so training can continue.", exc)
            values = torch.zeros(self.max_length, dtype=torch.float32)
        return {"input_values": values, "labels": int(sample.label)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiodeepfake-train",
        description=(
            "Fine-tune DistilHuBERT on a folder of real and fake speech. "
            "No dataset is downloaded. See the README for ASVspoof, WaveFake, "
            "and In-the-Wild layouts."
        ),
    )
    parser.add_argument("--train-dir", type=Path, default=None, help="Training audio root.")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Optional evaluation audio root.",
    )
    parser.add_argument("--format", default="auto", choices=FORMATS, help="How to read labels.")
    parser.add_argument("--train-protocol", type=Path, default=None, help="ASVspoof CM protocol.")
    parser.add_argument(
        "--train-audio-dir",
        type=Path,
        default=None,
        help="Audio for --train-protocol.",
    )
    parser.add_argument("--eval-protocol", type=Path, default=None)
    parser.add_argument("--eval-audio-dir", type=Path, default=None)
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help="CSV with path and label columns.",
    )
    parser.add_argument("--eval-csv", type=Path, default=None)
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="Folder that relative CSV paths are resolved against.",
    )
    parser.add_argument(
        "--real-dir",
        type=Path,
        default=None,
        help="Genuine audio used with --format wavefake (for example LJSpeech wavs).",
    )
    parser.add_argument("--eval-real-dir", type=Path, default=None)
    parser.add_argument("--base-model", default=DEFAULT_BACKBONE, help="Backbone to fine-tune.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/distilhubert-deepfake"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_SEGMENT_SECONDS,
        help="Seconds kept from the start of each training clip (default: 1).",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Undersample the larger class so fake and real counts match.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Notebook .bin checkpoint, a save_pretrained folder, or a Trainer checkpoint.",
    )
    parser.add_argument(
        "--push-to-hub",
        default=None,
        help="Hugging Face repo id to upload after training. Requires huggingface-cli login.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", help="Train in fp16. Needs a CUDA GPU.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print how many fake and real clips were found, then exit.",
    )
    return parser


def _spec_for(args: argparse.Namespace, split: str) -> SplitSpec | None:
    if split == "train":
        root = args.train_dir
        protocol = args.train_protocol
        audio_dir = args.train_audio_dir
        csv_path = args.train_csv
        real_dir = args.real_dir
    else:
        root = args.eval_dir
        protocol = args.eval_protocol
        audio_dir = args.eval_audio_dir
        csv_path = args.eval_csv
        real_dir = args.eval_real_dir or args.real_dir
        if root is None and protocol is None and csv_path is None:
            return None
    return SplitSpec(
        root=root,
        format=args.format,
        protocol=protocol,
        audio_dir=audio_dir,
        csv_path=csv_path,
        real_dir=real_dir,
        audio_root=args.audio_root,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        train_spec = _spec_for(args, "train")
        if train_spec is None or (
            train_spec.root is None and train_spec.protocol is None and train_spec.csv_path is None
        ):
            raise AudioDeepfakeError("Pass --train-dir, --train-protocol, or --train-csv.")
        train_data = load_split(train_spec)
        eval_spec = _spec_for(args, "eval")
        eval_data = load_split(eval_spec) if eval_spec is not None else None
    except AudioDeepfakeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    train_samples = train_data.samples
    if args.balance:
        try:
            train_samples = balance_undersample(train_samples, seed=args.seed)
        except AudioDeepfakeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    train_summary = summarize(train_samples)
    eval_summary = summarize(eval_data.samples) if eval_data is not None else None
    payload = {"format": train_data.format, "train": train_summary}
    if eval_data is not None and eval_summary is not None:
        payload["eval_format"] = eval_data.format
        payload["eval"] = eval_summary
        shared = overlapping_paths(train_samples, eval_data.samples)
        if shared:
            print(
                f"warning: {len(shared)} clips are in both train and eval. "
                "Hold out evaluation files so the score is not memorization.",
                file=sys.stderr,
            )
    print(json.dumps(payload, indent=2))

    if train_summary["fake"] == 0 or train_summary["real"] == 0:
        print(
            "error: training needs both real and fake clips. "
            f"Found {train_summary['real']} real and {train_summary['fake']} fake.",
            file=sys.stderr,
        )
        return 2
    if args.dry_run:
        return 0

    try:
        run_training(
            train_samples,
            None if eval_data is None else eval_data.samples,
            base_model=args.base_model,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            max_seconds=args.max_seconds,
            seed=args.seed,
            resume_from=args.resume_from,
            push_to_hub=args.push_to_hub,
            num_workers=args.num_workers,
            fp16=args.fp16,
            train_summary=train_summary,
            eval_summary=eval_summary,
        )
    except AudioDeepfakeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def run_training(
    train_samples,
    eval_samples,
    *,
    base_model: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    max_seconds: float,
    seed: int,
    resume_from: Path | None,
    push_to_hub: str | None,
    num_workers: int,
    fp16: bool,
    train_summary: dict[str, int],
    eval_summary: dict[str, int] | None,
) -> None:
    import torch

    try:
        from transformers import Trainer, TrainingArguments
    except ImportError as exc:
        raise AudioDeepfakeError(
            'Training needs the train extra: pip install "audiodeepfake-detect[train]". '
            f"Original error: {exc}"
        ) from exc

    _set_seed(seed)
    model, feature_extractor = load_backbone(base_model)
    resume_checkpoint = None
    if resume_from is not None:
        if resume_from.is_dir() and (resume_from / "trainer_state.json").is_file():
            resume_checkpoint = str(resume_from)
        else:
            load_checkpoint(model, resume_from)

    sample_rate = int(getattr(feature_extractor, "sampling_rate", DEFAULT_SAMPLE_RATE))
    max_length = int(round(max_seconds * sample_rate))
    train_dataset = AudioDataset(train_samples, feature_extractor, sample_rate, max_length)
    eval_dataset = None
    if eval_samples is not None:
        eval_dataset = AudioDataset(eval_samples, feature_extractor, sample_rate, max_length)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    argument_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "save_strategy": "epoch",
        "logging_steps": 50,
        "report_to": "none",
        "seed": seed,
        "fp16": fp16,
        "dataloader_num_workers": num_workers,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "save_total_limit": 2,
        "remove_unused_columns": False,
        "eval_strategy": "epoch" if eval_dataset is not None else "no",
        "load_best_model_at_end": eval_dataset is not None,
    }
    if eval_dataset is not None:
        argument_kwargs["metric_for_best_model"] = "f1_fake"
        argument_kwargs["greater_is_better"] = True

    trainer = Trainer(
        model=model,
        args=_training_arguments(TrainingArguments, argument_kwargs),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=_compute_metrics if eval_dataset is not None else None,
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(str(output_dir))
    feature_extractor.save_pretrained(str(output_dir))
    write_model_card(
        output_dir,
        base_model=base_model,
        max_seconds=max_seconds,
        sample_rate=sample_rate,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        train_summary=train_summary,
        eval_summary=eval_summary,
    )
    if push_to_hub:
        _push_to_hub(model, feature_extractor, output_dir, push_to_hub)
    print(f"Saved model to {output_dir}")


def _training_arguments(argument_cls, kwargs: dict):
    try:
        return argument_cls(**kwargs)
    except TypeError:
        if "eval_strategy" not in kwargs:
            raise
        renamed = dict(kwargs)
        renamed["evaluation_strategy"] = renamed.pop("eval_strategy")
        return argument_cls(**renamed)


def _compute_metrics(prediction) -> dict[str, float]:
    import numpy as np

    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    labels = prediction.label_ids
    predicted = np.asarray(logits).argmax(axis=-1)
    scores = binary_metrics(labels, predicted)
    return {key: float(value) for key, value in scores.items()}


def _set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _push_to_hub(model, feature_extractor, output_dir: Path, repo_id: str) -> None:
    model.push_to_hub(repo_id)
    feature_extractor.push_to_hub(repo_id)
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=str(output_dir / "README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add model card",
    )
    print(f"Pushed model to {repo_id}")


if __name__ == "__main__":
    raise SystemExit(main())
