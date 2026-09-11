"""TAMA: Time-domain Adjoint with Memory-efficient Acceleration.

TAMA intentionally keeps Meep objects visible: users still define sources,
geometry, materials, and simulations with Meep. TAMA evaluates point,
regional-field, signed-flux, and fixed-eigenmode objectives and computes their
time-domain adjoint gradients.
"""

__version__ = "0.5.0a3"

from .chunking import regularize_source_size_and_amplitude
from .adaptive_chunking import (
    AdaptiveAdjointChunkBalancer,
    AdaptiveChunkBalanceReport,
    AdaptiveSourceBoundaryDecision,
)
from .coords import centered_grid_coords
from .fabrication import (
    periodic_conic_filter,
    periodic_conic_filter_vjp,
    tanh_projection,
    tanh_projection_vjp,
)
from .sampling_grid import FastFieldGrid, FastGradientGrid, native_sampler_available
from .specs import (
    DesignGrid,
    EigenmodeCoefficientTarget,
    FieldRegionTarget,
    FluxTarget,
    PointTarget,
    SimulationSpec,
)
from .tda_objective import TDAObjective
from .multi_tda_objective import MultiTDAObjective, auto_pixel_chunk
from .nyquist import nyquist_sampling_interval

__all__ = [
    "__version__",
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
    "PointTarget",
    "SimulationSpec",
    "TDAObjective",
    "MultiTDAObjective",
    "auto_pixel_chunk",
    "native_sampler_available",
    "nyquist_sampling_interval",
]
