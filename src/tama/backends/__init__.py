"""Explicit backend selection without importing an unused engine."""

from importlib import import_module


def get_backend(name):
    """Return a backend module without changing global state or falling back."""
    if not isinstance(name, str) or name not in ("meep", "fdtdx"):
        raise ValueError("backend must be 'meep' or 'fdtdx'")
    try:
        import_module(name)
        return import_module(f".{name}", __name__)
    except ModuleNotFoundError as exc:
        if name == "meep" and exc.name == "meep":
            message = "The Meep backend requires pymeep in the active environment."
        elif name == "fdtdx" and exc.name in ("fdtdx", "jax", "jaxlib", "equinox"):
            message = (
                "The FDTDX backend requires its optional dependencies. "
                "Install tama[fdtdx] in a supported Python environment."
            )
        else:
            raise
        raise ModuleNotFoundError(message, name=exc.name) from exc


__all__ = ["get_backend"]
