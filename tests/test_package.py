import json
from pathlib import Path

import audiodeepfake

ROOT = Path(__file__).resolve().parents[1]


def test_public_api():
    assert audiodeepfake.__version__ == "0.1.0"
    assert callable(audiodeepfake.detect)
    assert "Detector" in audiodeepfake.__all__


def test_original_notebook_is_kept():
    notebook = ROOT / "notebooks" / "realdofinetuneV2.ipynb"
    data = json.loads(notebook.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in data["cells"])
    assert "ntu-spml/distilhubert" in text
    assert "def predict_audio" in text
    assert "def collect_audio_files_and_labels0" in text


def test_readme_does_not_copy_private_plot_numbers():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for number in ("94.8", "98.2", "98.7", "99.2", "61.7", "66.7", "64.0", "65.6"):
        assert number not in readme
    assert "Traditional Chinese" in readme or "繁體中文" in readme
    assert "ASVspoof" in readme
    assert "WaveFake" in readme
    assert "In-the-Wild" in readme
    assert "lightk/audiodeepfake-detecter-laogao" in readme
    assert "laogao" in readme
    assert "other speakers" in readme
    assert "預訓練模型" in readme
