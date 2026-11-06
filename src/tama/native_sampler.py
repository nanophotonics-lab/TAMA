"""Load the separate Meep extension without changing its native API."""

from importlib import import_module as _import_module
import sys as _sys

try:
    _native = _import_module("tama_meep_native.native_sampler")
except ModuleNotFoundError as exc:
    if exc.name not in ("tama_meep_native", "tama_meep_native.native_sampler"):
        raise
    raise ImportError(
        "TAMA native sampler extension is not built; install tama-meep-native "
        "from native/meep in the active Meep/MPI conda environment."
    ) from exc

_sys.modules[__name__] = _native
