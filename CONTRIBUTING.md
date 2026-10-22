# Contributing

Thanks for helping improve this project. The goal is a small, honest detector:
real code, real tests, and no made-up accuracy numbers.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,train,demo]"
```

## Checks

```bash
ruff check .
ruff format --check .
pytest
```

`pytest` uses a fake model. It must not download DistilHuBERT weights. CI sets
`HF_HUB_OFFLINE=1` for the same reason.

## Pull requests

- Describe what changed and how you tested it.
- Keep the public API stable: `detect(path)` returns a label and a fake score.
- Label `0` is fake and label `1` is real. Say so in the changelog if that ever changes.
- Do not commit datasets, checkpoints, or audio of real people.
- Do not add accuracy, precision, or F1 numbers unless the PR includes the
  command that produced them and the data split is public and described.

## Training data

Document new dataset readers in the README. A reader should accept a path the
user already downloaded. It should not download data inside the library.

## Reporting issues

Use the bug and feature templates. Do not attach private speech. Security
reports can go through a GitHub security advisory, or through a public issue
if you are comfortable with that.
