# Audio deepfake detector

[![CI](https://github.com/zLiGhTKz/AudioDeepFake-Detecter/actions/workflows/ci.yml/badge.svg)](https://github.com/zLiGhTKz/AudioDeepFake-Detecter/actions/workflows/ci.yml)

Fine-tune DistilHuBERT to tell **real speech** from **AI-synthesized speech**, then score new clips from Python, the command line, or a Gradio page.

**Status:** the code is ready to train. This repository does **not** ship fine-tuned weights, and it does **not** claim an accuracy number. `ntu-spml/distilhubert` is only the starting checkpoint. Detection refuses to load it, because that checkpoint has no deepfake classifier.

The original Colab notebook is kept at [`notebooks/realdofinetuneV2.ipynb`](notebooks/realdofinetuneV2.ipynb). Use the commands below for anything you want to keep.

## What a score means

`result.label` is `fake` or `real`.

`result.score` is the mean probability of the **fake** class. `0` means the model leans real. `1` means the model leans synthetic. A clip is labeled fake when `score >= threshold` (default `0.5`).

Label ids match the notebook: **0 = fake**, **1 = real**.

A score is a model opinion about the audio it was trained on. It is not proof that a person is real or fake. See [Limitations](#limitations) and [Ethics](#ethics).

## Install

Python 3.10 or newer.

```bash
pip install -e ".[dev,train,demo]"
```

From Git, once this branch is merged:

```bash
pip install "audiodeepfake-detect @ git+https://github.com/zLiGhTKz/AudioDeepFake-Detecter.git"
```

The package is not on PyPI yet. Inference needs PyTorch, Transformers, librosa, and soundfile. Training also needs the `train` extra (`accelerate`). The demo needs the `demo` extra (`gradio`).

wav, flac, and ogg are read with libsndfile. mp3, m4a, aac, and wma also need `ffmpeg` on your `PATH`.

## Detect

```bash
audiodeepfake-detect speech.wav --model your-name/distilhubert-deepfake
audiodeepfake-detect clips/ --model your-name/distilhubert-deepfake --json
```

Folders are searched recursively. The table columns are path, label, fake score, and number of one-second windows.

```bash
export AUDIODEEPFAKE_MODEL=your-name/distilhubert-deepfake
audiodeepfake-detect speech.wav
```

Exit codes: `0` finished, `1` at least one file could not be read, `2` missing model or bad arguments, `3` with `--fail-on-fake` when any file is labeled fake.

Python:

```python
from audiodeepfake import detect

result = detect("speech.wav", model="your-name/distilhubert-deepfake")
print(result.label, round(result.score, 3))

for item in detect("clips/", model="your-name/distilhubert-deepfake"):
    print(item.path, item.label, round(item.score, 3))
```

`detect("file.wav")` returns one result. `detect("folder/")` and `detect(["a.wav", "b.wav"])` return a list.

Audio is resampled to 16 kHz. Each full second is scored on its own. The file score is the mean of those window scores. A tail shorter than one second is dropped, same as the notebook. A whole file shorter than one second is padded with silence so it still gets one score.

Use the same `--segment-seconds` you trained with. The default is `1`, because the notebook set `max_length=16000` samples. The notebook file name says `5s`. The code did not.

If you have not trained a model yet, the command stops and tells you how to train. It does not download the backbone and invent a score.

## Train

No dataset is downloaded or stored in this repo. Put files on disk yourself, and follow each dataset's license.

Defaults follow the notebook: AdamW, learning rate `5e-5`, weight decay `0.01`, warmup `500`, `5` epochs, batch size `4`, and a **1 second** crop from the start of each file. The Trainer builds that optimizer. The notebook also built its own scheduler with `num_training_steps` set to the number of files, which is not one step per file. This script uses the Trainer schedule instead.

Check the layout before a long run:

```bash
audiodeepfake-train --train-dir /data/myset --dry-run
```

Train:

```bash
audiodeepfake-train \
  --train-dir /data/myset/train \
  --eval-dir /data/myset/dev \
  --output-dir outputs/distilhubert-deepfake \
  --balance
```

`--balance` undersamples the larger class so the fake count and the real count match. The draw is repeatable (`--seed`, default `123`). The notebook only undersampled real clips. This command undersamples whichever class is larger.

`--dry-run` prints counts and exits. It does not download the backbone.

The saved folder is a normal Transformers classifier (`config.json`, weights, feature extractor, and a model card). The card lists the setup and the clip counts. It leaves accuracy blank on purpose. Fill that in only after you measure it.

Resume from a notebook `.bin` (weights only) or keep training from a Trainer checkpoint directory:

```bash
audiodeepfake-train --train-dir /data/myset/train --resume-from path/to/model.bin
```

### Folder layout (`--format directory` or `auto`)

```text
train/
  real/
    clip.wav
  fake/
    clip.wav
```

`bonafide`, `genuine`, `spoof`, and `synthetic` folder names work too. The closest labeled folder wins.

### File names (`--format filename`)

`spoof`, `fake`, `bonafide`, or `real` in the file name. Files with none of those words are an error, so they are not silently called real.

Two notebook rules are still available, because the Colab code used them:

| Format | Rule |
| --- | --- |
| `notebook` | Lowercase `fake` anywhere in the name means fake. Every other name means real. `FAKE.wav` does **not** match. |
| `notebook-f` | Any capital `F` means fake. `fake.wav` is real. `myFile.wav` is fake. |

Do not use `notebook-f` unless you are reproducing that notebook. It is a very wide match.

### CSV and In-the-Wild (`--format csv` or `in-the-wild`)

A header row needs a path column (`file`, `path`, `audio`, ...) and a label column (`label`, `class`, ...).

[In-the-Wild](https://arxiv.org/abs/2203.16263) (Müller et al.) is usually a folder like this:

```text
release_in_the_wild/
  meta.csv      # columns: file, speaker, label
  0.wav
  1.wav
```

`label` is `spoof` or `bona-fide`. People usually **test** on this set after training somewhere else, because it is recorded outside the lab. One public copy of the author's release is [mueller91/In-The-Wild](https://huggingface.co/datasets/mueller91/In-The-Wild). Read that dataset's terms before you use it. This repo does not redistribute the audio.

```bash
audiodeepfake-eval \
  --model outputs/distilhubert-deepfake \
  --data-dir /data/release_in_the_wild \
  --format in-the-wild
```

### ASVspoof (`--format asvspoof`)

[ASVspoof 2019 Logical Access](https://datashare.ed.ac.uk/handle/10283/3336) CM protocols look like this:

```text
SPEAKER UTTERANCE - SYSTEM KEY
LA_0079 LA_T_1138215 - - bonafide
LA_0079 LA_T_1271820 - A01 spoof
```

Audio files are flac. The utterance id is the file stem. Later ASVspoof CM lists that end in `bonafide` or `spoof` use the same reader. ASV trial lists (keys like `target` / `nontarget`) are ignored.

Pass one protocol per split so train, dev, and eval stay apart. Pointing `--format asvspoof` at a folder that contains several protocols is an error.

```bash
audiodeepfake-train \
  --format asvspoof \
  --train-protocol /data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
  --train-audio-dir /data/LA/ASVspoof2019_LA_train/flac \
  --eval-protocol /data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
  --eval-audio-dir /data/LA/ASVspoof2019_LA_dev/flac \
  --output-dir outputs/asvspoof2019 \
  --balance
```

ASVspoof has its own DataShare terms. Read them. The 2019 evaluation protocol is easy to overfit if you train on it. Keep it for a final test.

The project site is <https://www.asvspoof.org/>.

### WaveFake (`--format wavefake`)

[WaveFake](https://zenodo.org/records/5642694) (Frank and Schönherr) ships generated clips only. Reference speech is separate. LJSpeech is the usual real set for the English vocoders: <https://keithito.com/LJ-Speech-Dataset/>. The generator code and folder names are documented at <https://github.com/RUB-SysSec/WaveFake>.

```text
data/
  ljspeech_melgan/
  ljspeech_hifiGAN/
  real/                # or pass --real-dir /path/to/LJSpeech-1.1/wavs
```

Every file under the generated tree is fake. Files under `real/` (or `--real-dir`) are real. `auto` chooses this layout when it sees vocoder folder names such as `melgan` or `hifigan`.

Split the genuine files yourself. Do not put the same real clip in train and eval. The trainer prints a warning when paths overlap. Hold out at least one vocoder if you care about unseen generators.

```bash
audiodeepfake-train \
  --format wavefake \
  --train-dir /data/wavefake/ljspeech_melgan \
  --real-dir /data/LJSpeech-1.1/wavs_train \
  --eval-dir /data/wavefake/ljspeech_hifiGAN \
  --eval-real-dir /data/LJSpeech-1.1/wavs_dev \
  --output-dir outputs/wavefake
```

WaveFake's generated audio is CC BY-SA 4.0. LJSpeech has its own license. Neither set is included here.

### `auto` order

1. `meta.csv` or `release_in_the_wild/meta.csv` → In-the-Wild
2. One CSV with path and label columns
3. One ASVspoof-style protocol
4. Vocoder folders → WaveFake (and a `real/` folder if it is there)
5. `real/` and `fake/` folders
6. Words in the file name

## Publish weights

1. Train and read `outputs/<run>/README.md`. Add the scores you actually measured. Delete nothing you cannot explain.
2. Log in: `huggingface-cli login`
3. Either train with `--push-to-hub your-name/distilhubert-deepfake`, or upload the output folder in the Hub UI.
4. The model config should say label 0 is `fake` and label 1 is `real`. `audiodeepfake-train` writes that.
5. Check the license of `ntu-spml/distilhubert` and of every dataset in the run before the repo is public.
6. Score a file:

```bash
audiodeepfake-detect speech.wav --model your-name/distilhubert-deepfake
```

`audiodeepfake-eval` prints accuracy and fake-class precision, recall, and F1 for **this** run only. Add `--window` to score the same first second that training saw. The default scores every full second, which is what the CLI does.

```bash
audiodeepfake-eval \
  --model outputs/distilhubert-deepfake \
  --data-dir /data/myset/test \
  --json
```

## Gradio demo

[`app.py`](app.py) is a small Space app. It loads the model on the first click.

```bash
pip install -e ".[demo]"
export AUDIODEEPFAKE_MODEL=your-name/distilhubert-deepfake
python app.py
```

On a new Hugging Face Space (Gradio SDK), this repo's [`requirements.txt`](requirements.txt) installs the package and Gradio. Set the Space variable `AUDIODEEPFAKE_MODEL` to the model id. Until that variable is set, the page explains how to train. It does not pretend the base model can detect anything.

## Development

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

Tests build a tiny fake classifier in memory. CI also sets `HF_HUB_OFFLINE=1`.

```text
src/audiodeepfake/    library, CLI, training, evaluation
app.py                Gradio Space
notebooks/            original Colab notebook
tests/                pytest, no weight download
```

## Limitations

- There is no published detector checkpoint in this release. You must train or supply one.
- The default crop is one second. A model trained at a different length will not match these scores.
- Windows do not overlap. A short tail is ignored. Padding can change the score of a very short clip.
- A model trained on one set of vocoders often fails on a new vocoder, a new language, telephone audio, or heavy compression. In-the-Wild exists because lab models often do not travel.
- File-name labels and the notebook's capital-`F` rule will mislabel careless names. Prefer protocols or `real/` and `fake/` folders.
- The notebook drew a plot from a private Google Drive run. Those figures are not a benchmark for this package, and they are not repeated here.
- CPU inference works. It is slow on long files. `--batch-size` controls how many windows run at once.

## Ethics

Use this for research, dataset work, and your own audio. Do not treat a score as evidence in a dispute, a newsroom, or a workplace investigation. Tell people when you run their voice through a model. Datasets of public figures and of spoofed speech still have licenses and privacy limits. Read them.

A public detector can also be studied by people who build synthesizers. The code is here so the method can be checked and improved, not so a score can be used to accuse someone.

## 繁體中文簡介

這個專案把 DistilHuBERT 微調成一個分類器，用來判斷一段語音比較像真人，還是像 AI 合成的語音（深偽語音）。

目前**沒有**附上已經訓練好的權重，也**沒有**公布準確率。`ntu-spml/distilhubert` 只是訓練的起點，不能直接拿來偵測。

安裝（在這個資料夾裡）：

```bash
pip install -e ".[dev,train,demo]"
```

訓練後偵測：

```bash
audiodeepfake-detect speech.wav --model 你的帳號/你的模型
```

`score` 是「像合成語音」的機率，愈高愈像假的。標籤 `fake` 是合成，`real` 是真人。程式預設每 1 秒看一次，因為筆記本裡的 `max_length` 是 16000 個取樣點，不是檔名寫的 5 秒。

資料要自己下載，不能放進這個倉庫。支援的公開資料包含 ASVspoof（2019 LA 的 CM 清單）、WaveFake（另外準備 LJSpeech 當真人語音）、以及 In-the-Wild（`meta.csv`，標籤是 `spoof` 或 `bona-fide`）。

這個分數不是證據。沒看過的合成器、壓縮、雜訊、語言，都可能讓判斷失準。請不要只用它指控別人。

## License

[MIT](LICENSE). The base model and any dataset you train on have their own licenses. Those licenses still apply to weights you publish.
