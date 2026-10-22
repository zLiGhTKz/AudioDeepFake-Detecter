"""Gradio demo for a Hugging Face Space.

Set AUDIODEEPFAKE_MODEL to a fine-tuned model id before you expect real scores.
With that variable unset, the page explains how to train and publish weights.
It does not download ntu-spml/distilhubert and pretend those weights can detect deepfakes.
"""

from __future__ import annotations

import os

from audiodeepfake.errors import MISSING_MODEL_HELP, AudioDeepfakeError
from audiodeepfake.model import ENV_MODEL_ID

_detector = None


def _configured_model_id() -> str:
    return os.environ.get(ENV_MODEL_ID, "").strip()


def predict(audio_path: str | None):
    """Score one uploaded file. The model is loaded on the first successful call."""
    if audio_path is None:
        return "No file yet", None, {"error": "Upload a speech clip or record one."}

    model_id = _configured_model_id()
    if not model_id:
        return (
            "Model not configured",
            None,
            {
                "error": MISSING_MODEL_HELP,
                "env": ENV_MODEL_ID,
            },
        )

    global _detector
    try:
        if _detector is None:
            from audiodeepfake import Detector

            _detector = Detector(model_id=model_id)
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
    if model_id:
        status = f"Model: `{model_id}`. The fake score is the mean probability of synthetic speech."
    else:
        status = (
            "No fine-tuned model is configured. Set the Space variable "
            f"`{ENV_MODEL_ID}` to a Hugging Face model id after you train and upload weights. "
            "Until then this page cannot score audio."
        )

    with gr.Blocks(title="Audio deepfake detector") as demo:
        gr.Markdown(
            "# Audio deepfake detector\n\n"
            "This demo scores a speech clip as **real** or **fake** (AI-synthesized). "
            "A score is not proof. Unseen generators, codecs, noise, and languages "
            "can make it wrong. Do not use it as the only reason to accuse someone.\n\n"
            f"{status}"
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
