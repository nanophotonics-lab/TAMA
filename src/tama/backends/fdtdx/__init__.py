"""TAMA evaluation interface over the FDTDX engine.

Scene, target, band and callback definitions retain their FDTDX conventions.
Only the design-vector shape and evaluation result follow the common TAMA API.
"""

import numpy as np

from .specs import DesignGrid, SimulationSpec
from .targets import (Band, EigenmodeCoefficientTarget, FieldRegionTarget,
                      FluxTarget, PointTarget)
from .filters import (bandpass_kernel, periodic_conic_filter,
                      power_complementary_kernels, tanh_projection)
from .backend import (AccuracyError, verify_fdtdx_compatibility, verify_fdtdx_sources,
                      MultiTDAObjective as _NativeMultiTDAObjective,
                      TDAObjective as _NativeTDAObjective)

__all__ = [
    "SimulationSpec", "DesignGrid", "PointTarget", "FieldRegionTarget",
    "FluxTarget", "EigenmodeCoefficientTarget", "Band", "TDAObjective",
    "MultiTDAObjective", "AccuracyError", "verify_fdtdx_compatibility", "verify_fdtdx_sources",
    "bandpass_kernel", "power_complementary_kernels",
    "periodic_conic_filter", "tanh_projection",
]


class MultiTDAObjective:
    """Native FDTDX multi-band objective accepting a flat design vector.

    ``sampling_interval`` is the native history stride. Values greater than one
    reconstruct electric histories from sparse samples.
    ``reconstruction_half_width`` is the native sinc reconstruction half-width.
    Other settings keep their native meanings; Meep geometry, wavelength bands,
    component constants, callback signatures and normalization are not converted.

    Configuration is available through the read-only ``native_objective``.
    CPU offload stores design histories on the host; it requires a CUDA device.
    """

    __slots__ = ("_native_objective",)

    def __init__(self, *, simulation, design, targets, bands=None,
                 scalarization_fn=None, sampling_interval=1, block_steps=64,
                 reconstruction_half_width=64, cpu_offload=False):
        self._native_objective = _NativeMultiTDAObjective(
            simulation, design, targets, bands=bands,
            scalarization_fn=scalarization_fn, stride=sampling_interval,
            block_steps=block_steps, half_width=reconstruction_half_width,
            cpu_offload=cpu_offload,
        )

    @property
    def native_objective(self):
        """The validated backend objective, with immutable configuration."""
        return self._native_objective

    @property
    def last_info(self):
        """Latest gradient diagnostics; None after value-only/helper calls."""
        return self._native_objective.last_info

    def _rho(self, x):
        self._native_objective.last_info = None
        values = np.asarray(x)
        shape = tuple(self._native_objective.design.parameter_shape)
        if values.shape == (int(np.prod(shape)),):
            return values.reshape(shape)
        if values.shape != shape:
            raise ValueError(f"x must be flat with {np.prod(shape)} entries or have shape {shape}; got {values.shape}")
        # The backend checks dtype, finiteness, bounds and transformed material.
        return values

    @staticmethod
    def _result(value, gradient=None):
        value = float(value)
        gradient = None if gradient is None else np.asarray(gradient).ravel()
        if not np.isfinite(value) or (gradient is not None and not np.isfinite(gradient).all()):
            raise FloatingPointError("Nonfinite objective/gradient; check the simulation and objective callback")
        return value, gradient

    def evaluate(self, x, need_gradient=True, *, cpu_offload=None):
        """Return ``(float, flat NumPy gradient)`` or ``(float, None)``.

        Value-only evaluation calls the native forward value path, without an
        adjoint or stored design histories. A valid offload flag has no effect
        on that path. Native input validation remains active; no alternate
        solver is selected when an evaluation fails.
        """
        self._native_objective.last_info = None
        if type(need_gradient) is not bool:
            raise TypeError("need_gradient must be a Python bool")
        if cpu_offload is not None and type(cpu_offload) is not bool:
            raise TypeError("cpu_offload must be a Python bool or None")
        rho = self._rho(x)
        if not need_gradient:
            return self._result(self._native_objective.value(rho))
        return self._result(*self._native_objective.evaluate(rho, cpu_offload=cpu_offload))

    def __call__(self, x, need_gradient=True, *, cpu_offload=None):
        return self.evaluate(x, need_gradient=need_gradient, cpu_offload=cpu_offload)

    def fom(self, x):
        """Return the forward-only scalar objective."""
        return self.evaluate(x, need_gradient=False)[0]

    def fom_and_grad(self, x):
        """Return the scalar objective and flat design gradient."""
        return self.evaluate(x, need_gradient=True)

    def scipy_fun(self, x):
        """Callable for scipy.optimize.minimize(..., jac=True, bounds=...)."""
        return self.fom_and_grad(x)

    def reference_value_and_grad(self, x):
        """Full-state native AD reference with the same flat output contract."""
        return self._result(*self._native_objective.reference_value_and_grad(self._rho(x)))

    def validate(self, x, *, directions=3, step=None):
        """Check this design using full-state AD and centered finite differences."""
        return self._native_objective.validate(self._rho(x), directions=directions, step=step)


class TDAObjective(MultiTDAObjective):
    """Single-target FDTDX objective with the common TAMA evaluation interface."""

    __slots__ = ()

    def __init__(self, *, simulation, design, target, kernel=None, fom_fn=None,
                 scalarization_fn=None, sampling_interval=1, block_steps=64,
                 reconstruction_half_width=64, cpu_offload=False):
        self._native_objective = _NativeTDAObjective(
            simulation, design, target, kernel=kernel, fom_fn=fom_fn,
            scalarization_fn=scalarization_fn,
            stride=sampling_interval, block_steps=block_steps,
            half_width=reconstruction_half_width, cpu_offload=cpu_offload,
        )
