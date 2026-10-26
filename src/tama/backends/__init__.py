"""Lazy selection of the Meep backend."""

from importlib import import_module


def get_backend(name):
    """Return a backend module without changing global state or falling back."""
    if not isinstance(name, str) or name != "meep":
        raise ValueError("backend must be 'meep'")
    try:
        import_module(name)
        return import_module(f".{name}", __name__)
    except ModuleNotFoundError as exc:
        if exc.name != "meep":
            raise
        raise ModuleNotFoundError(
            "The Meep backend requires pymeep in the active environment.",
            name=exc.name,
        ) from exc


__all__ = ["get_backend"]
