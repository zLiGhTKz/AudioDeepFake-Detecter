"""Gradio demo for a Hugging Face Space.

The default model is ``lightk/audiodeepfake-detecter-laogao``. Set
``AUDIODEEPFAKE_MODEL`` to use a different fine-tuned repo. The base
checkpoint ``ntu-spml/distilhubert`` is not a detector and is rejected.
"""

from __future__ import annotations

from audiodeepfake.errors import AudioDeepfakeError
from audiodeepfake.model import DEFAULT_MODEL_ID, ENV_MODEL_ID, resolve_model_id

_detector = None
_loaded_model_id = None


def _configured_model_id() -> str:
    return resolve_model_id(None)


def _status_text(model_id: str) -> str:
    text = f"Model: `{model_id}`. The fake score is the mean probability of synthetic speech."
    if model_id == DEFAULT_MODEL_ID:
        text += (
            " This checkpoint was trained on ASVspoof and then fine-tuned on one "
            "speaker (laogao). It is specialised to that speaker and has not been "
            "evaluated on other speakers. This page does not claim an accuracy number."
        )
    text += f" Set `{ENV_MODEL_ID}` to load a different model."
    return text


def predict(audio_path: str | None):
    """Score one uploaded file. The model is loaded on the first successful call."""
    if audio_path is None:
        return "No file yet", None, {"error": "Upload a speech clip or record one."}

    model_id = _configured_model_id()
    global _detector, _loaded_model_id
    try:
        if _detector is None or _loaded_model_id != model_id:
            from audiodeepfake import Detector

            _detector = Detector(model_id=model_id)
            _loaded_model_id = model_id
        result = _detector.detect_file(audio_path)
    except AudioDeepfakeError as exc:
        return "Error", None, {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - show Space startup failures in the UI
        return "Error", None, {"error": str(exc)}

    return result.label, result.score, result.to_dict()


def create_demo():
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit(
            'The demo needs Gradio. Install it with: pip install "audiodeepfake-detect[demo]"'
        ) from exc

    model_id = _configured_model_id()
    with gr.Blocks(title="Audio deepfake detector") as demo:
        gr.Markdown(
            "# Audio deepfake detector\n\n"
            "This demo scores a speech clip as **real** or **fake** (AI-synthesized). "
            "A score is not proof. Unseen generators, codecs, noise, and languages "
            "can make it wrong. Do not use it as the only reason to accuse someone.\n\n"
            f"{_status_text(model_id)}"
        )
        audio = gr.Audio(type="filepath", label="Speech clip")
        button = gr.Button("Check audio")
        label = gr.Textbox(label="Label")
        score = gr.Number(label="Fake score (higher means more likely synthetic)")
        details = gr.JSON(label="Details")
        button.click(predict, inputs=audio, outputs=[label, score, details])
    return demo


demo = create_demo()

if __name__ == "__main__":
    demo.launch()
