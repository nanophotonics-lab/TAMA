"""Meep-native TAMA API; numerical modules are imported on demand."""

from importlib import import_module as _import_module
from ... import __version__, _EXPORTS

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = "...nyquist" if _EXPORTS[name] == "nyquist" else f".{_EXPORTS[name]}"
    value = getattr(_import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
