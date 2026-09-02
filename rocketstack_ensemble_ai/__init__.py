"""Public package interface for RocketStack Ensemble AI."""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from rocketstack_functions_with_model_api_original import RocketStack as _RocketStack

try:
    __version__ = version("rocketstack_ensemble_ai")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.1.0"


class RocketStack(_RocketStack):
    """RocketStack API with branded lowercase model-factory aliases."""

    ascentTheRocket_binary_model = _RocketStack.AscentTheRocket_binary_model
    ascentTheRocket_multiclass_model = _RocketStack.AscentTheRocket_multiclass_model


def ascentTheRocket_binary_model(**kwargs: Any) -> Any:
    """Create a trainable binary RocketStack model."""

    return RocketStack().ascentTheRocket_binary_model(**kwargs)


def ascentTheRocket_multiclass_model(**kwargs: Any) -> Any:
    """Create a trainable multiclass RocketStack model."""

    return RocketStack().ascentTheRocket_multiclass_model(**kwargs)


__all__ = [
    "RocketStack",
    "__version__",
    "ascentTheRocket_binary_model",
    "ascentTheRocket_multiclass_model",
]
