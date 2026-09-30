"""Compatibility alias for :mod:`tama.backends.meep.sampling_grid`."""

from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module(".backends.meep.sampling_grid", "tama")
