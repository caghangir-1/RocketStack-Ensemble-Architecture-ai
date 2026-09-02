"""Public package interface for RocketStack Ensemble AI."""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from rocketstack_functions_with_model_api_original import RocketStack

try:
    __version__ = version("rocketstack_ensemble_ai")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.1.0"


def make_binary_classifier(**kwargs: Any) -> Any:
    """Create a trainable binary RocketStack model.

    Keyword arguments are passed to
    :meth:`RocketStack.AscentTheRocket_binary_model`.
    """

    return RocketStack().AscentTheRocket_binary_model(**kwargs)


def make_multiclass_classifier(**kwargs: Any) -> Any:
    """Create a trainable multiclass RocketStack model.

    Keyword arguments are passed to
    :meth:`RocketStack.AscentTheRocket_multiclass_model`.
    """

    return RocketStack().AscentTheRocket_multiclass_model(**kwargs)


__all__ = [
    "RocketStack",
    "__version__",
    "make_binary_classifier",
    "make_multiclass_classifier",
]

