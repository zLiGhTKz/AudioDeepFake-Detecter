"""Expected failures raised by the library and the command line tools."""


class AudioDeepfakeError(Exception):
    """Base class for errors the user can fix."""


class ModelNotConfigured(AudioDeepfakeError):
    """Raised when no fine-tuned Hugging Face model id is available."""


class UntrainedModelError(AudioDeepfakeError):
    """Raised when the chosen repo is a backbone, not a classifier."""


class AudioLoadError(AudioDeepfakeError):
    """Raised when an input path cannot be read as audio."""


class DatasetLayoutError(AudioDeepfakeError):
    """Raised when a dataset folder does not match the selected layout."""


MISSING_MODEL_HELP = (
    "No fine-tuned model id was given. This package does not ship "
    "deepfake-detection weights. Train with audiodeepfake-train, upload the "
    "output folder to the Hugging Face Hub, then pass that repo id with "
    "--model or the AUDIODEEPFAKE_MODEL environment variable. "
    "ntu-spml/distilhubert is only the backbone used for training. "
    "It cannot detect deepfakes on its own."
)
