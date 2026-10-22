# Changelog

## 0.1.0 - 2026-10-22

First packaged release. The Colab notebook is still in
`notebooks/realdofinetuneV2.ipynb`.

- Added the `audiodeepfake` package, the `detect` API, and the
  `audiodeepfake-detect`, `audiodeepfake-train`, and `audiodeepfake-eval`
  commands.
- Training reads ASVspoof CM protocols, WaveFake plus a real-audio folder,
  the In-the-Wild `meta.csv` layout, class folders, CSVs, and the notebook's
  file-name rules.
- Inference loads a fine-tuned model id from the Hugging Face Hub or a local
  folder. The base checkpoint `ntu-spml/distilhubert` is rejected for detection
  because it has no trained classifier.
- The CLI and the Gradio app default to
  `lightk/audiodeepfake-detecter-laogao` (override with `--model` or
  `AUDIODEEPFAKE_MODEL`). That checkpoint was trained on ASVspoof and then
  fine-tuned on one speaker (`laogao`). It has not been evaluated on other
  speakers. This changelog does not claim an accuracy number.
- Added a Gradio app, pytest coverage with a fake model, Ruff, and GitHub
  Actions.

Behavior that differs from the notebook on purpose:

- Clips shorter than one window are padded and scored. The notebook returned
  no prediction for those clips.
- A leftover tail shorter than one window is still dropped.
- The capital-`F` file-name rule is opt-in (`--format notebook-f`). It is easy
  to mislabel files with it.
- Checkpoints are saved with `save_pretrained` so they can be uploaded to the
  Hub. A notebook `.bin` file can still be loaded with `--resume-from`.
