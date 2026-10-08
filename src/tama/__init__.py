"""TAMA with explicit engine selection and lazy legacy Meep exports.

Existing root-level classes remain the Meep API. Use ``get_backend('fdtdx')``
for FDTDX's native scene, design and target types.
"""

from importlib import import_module as _import_module

from .backends import get_backend

__version__ = "0.6.0a1"

_EXPORTS = {
    "regularize_source_size_and_amplitude": "chunking",
    "AdaptiveAdjointChunkBalancer": "adaptive_chunking",
    "AdaptiveChunkBalanceReport": "adaptive_chunking",
    "AdaptiveSourceBoundaryDecision": "adaptive_chunking",
    "centered_grid_coords": "coords",
    "periodic_conic_filter": "fabrication",
    "periodic_conic_filter_vjp": "fabrication",
    "tanh_projection": "fabrication",
    "tanh_projection_vjp": "fabrication",
    "FastFieldGrid": "sampling_grid",
    "FastGradientGrid": "sampling_grid",
    "native_sampler_available": "sampling_grid",
    "DesignGrid": "specs",
    "EigenmodeCoefficientTarget": "specs",
    "FieldRegionTarget": "specs",
    "FluxTarget": "specs",
    "Near2FarTarget": "specs",
    "PointTarget": "specs",
    "SimulationSpec": "specs",
    "TDAObjective": "tda_objective",
    "MultiTDAObjective": "multi_tda_objective",
    "auto_pixel_chunk": "multi_tda_objective",
    "nyquist_sampling_interval": "nyquist",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = (
        ".nyquist"
        if _EXPORTS[name] == "nyquist"
        else f".backends.meep.{_EXPORTS[name]}"
    )
    try:
        value = getattr(_import_module(module, __name__), name)
    except ModuleNotFoundError as exc:
        if exc.name != "meep":
            raise
        raise ModuleNotFoundError(
            f"tama.{name} is part of the Meep backend. Install pymeep in the "
            "active environment, or select tama.get_backend('fdtdx').",
            name="meep",
        ) from exc
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "__version__",
    "get_backend",
    "regularize_source_size_and_amplitude",
    "AdaptiveAdjointChunkBalancer",
    "AdaptiveChunkBalanceReport",
    "AdaptiveSourceBoundaryDecision",
    "centered_grid_coords",
    "periodic_conic_filter",
    "periodic_conic_filter_vjp",
    "tanh_projection",
    "tanh_projection_vjp",
    "FastFieldGrid",
    "FastGradientGrid",
    "DesignGrid",
    "EigenmodeCoefficientTarget",
    "FieldRegionTarget",
    "FluxTarget",
    "Near2FarTarget",
    "PointTarget",
    "SimulationSpec",
    "TDAObjective",
    "MultiTDAObjective",
    "auto_pixel_chunk",
    "native_sampler_available",
    "nyquist_sampling_interval",
]
