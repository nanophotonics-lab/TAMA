"""Compatibility imports for the previous sampling-grid module name."""

from .sampling_grid import FastFieldGrid, FastGradientGrid, native_sampler_available

__all__ = ["FastFieldGrid", "FastGradientGrid", "native_sampler_available"]
