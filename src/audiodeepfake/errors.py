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
    "ntu-spml/distilhubert is only the backbone used for training. "
    "It cannot detect deepfakes. The default detector is "
    "lightk/audiodeepfake-detecter-laogao. Pass that id, or another fine-tuned "
    "repo, with --model or the AUDIODEEPFAKE_MODEL environment variable."
)
