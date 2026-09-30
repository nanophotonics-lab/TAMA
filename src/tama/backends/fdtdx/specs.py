"""Validated native FDTDX scenes and an explicit scalar-permittivity design map.

The extension varies inverse permittivity only. Native object geometry, source
profiles, permeability and boundary coefficients remain fixed after placement.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


def _triple(value, name, *, positive):
    value = tuple(value)
    if len(value) != 3 or any(isinstance(n, bool) or not isinstance(n, Integral)
                              or n < (1 if positive else 0) for n in value):
        raise ValueError(f"{name} must contain three {'positive' if positive else 'nonnegative'} integers")
    return tuple(int(n) for n in value)


def _overlaps(start, shape, slices):
    return all(a < sl.stop and sl.start < a + n for a, n, sl in zip(start, shape, slices))


def _positive_array(value, name):
    value = np.asarray(value)
    if np.iscomplexobj(value) or not np.all(np.isfinite(value)) or np.any(value <= 0):
        raise ValueError(f"{name} must be finite, real and strictly positive")
    return value


@dataclass(frozen=True)
class SimulationSpec:
    """Native arrays, placed objects and config for a fixed-duration simulation.

    Construct after ``fdtdx.place_objects`` and ``fdtdx.apply_params`` so that
    native source caches have been initialized. Supported scenes use real scalar
    nondispersive lossless materials, fixed isotropic permeability, and native
    Cartesian PML, PEC/PMC or Bloch-periodic boundaries. Custom source subclasses
    are rejected because their update need not be affine in the field state.
    The object sequence is snapshotted. Each objective evaluation starts from
    zero E/H and PML fields; supplied dynamic fields are not initial conditions.
    """

    arrays: object
    objects: object
    config: object
    key: object = None

    def __post_init__(self):
        from fdtdx.config import SimulationConfig
        from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
        from fdtdx.objects.boundaries.bloch import BlochBoundary
        from fdtdx.objects.boundaries.pec import PerfectElectricConductor
        from fdtdx.objects.boundaries.pmc import PerfectMagneticConductor
        from fdtdx.objects.boundaries.perfectly_matched_layer import PerfectlyMatchedLayer
        from fdtdx.objects.sources.dipole import PointDipoleSource
        from fdtdx.objects.sources.linear_polarization import GaussianPlaneSource, UniformPlaneSource
        from fdtdx.objects.sources.profile import CustomTimeSignalProfile, GaussianPulseProfile, SingleFrequencyProfile

        if not isinstance(self.arrays, ArrayContainer) or not isinstance(self.objects, ObjectContainer):
            raise TypeError("arrays and objects must be native placed FDTDX containers")
        if not isinstance(self.config, SimulationConfig):
            raise TypeError("config must be a native FDTDX SimulationConfig")
        object.__setattr__(self, "objects", ObjectContainer(
            object_list=tuple(self.objects.object_list), volume_idx=self.objects.volume_idx))
        if self.key is None:
            object.__setattr__(self, "key", jax.random.PRNGKey(0))
        try:
            jax.random.key_data(self.key)
        except (TypeError, ValueError) as error:
            raise ValueError("key must be a valid native JAX PRNG key") from error
        if not np.isfinite(self.dt) or self.dt <= 0 or self.time_steps < 1:
            raise ValueError("simulation requires a positive time step and at least one time step")
        dtype = np.dtype(self.config.dtype)
        if dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("only real float32 and float64 simulations are supported")
        if dtype == np.dtype("float64") and not jax.config.x64_enabled:
            raise ValueError("enable jax_enable_x64 before constructing a float64 simulation")
        if self.arrays.fields.E.shape != (3, *self.shape) or self.arrays.fields.H.shape != (3, *self.shape):
            raise ValueError("native E/H field shapes must match the Cartesian simulation volume")
        field_dtype = np.dtype(self.arrays.fields.E.dtype)
        complex_dtype = np.dtype("complex64" if dtype == np.dtype("float32") else "complex128")
        if field_dtype not in (dtype, complex_dtype) or np.dtype(self.arrays.fields.H.dtype) != field_dtype:
            raise ValueError("E/H must share the configured float precision, real or complex")
        rebuild = "Rebuild the native scene with matching precision; for float32, set jax_enable_x64=False before placement."
        for name in ("psi_E", "psi_H"):
            if np.dtype(getattr(self.arrays.fields, name).dtype) != field_dtype:
                raise ValueError(f"{name} dtype must match E/H. {rebuild}")
        for name in ("alpha", "kappa", "sigma", "inv_permittivities"):
            if np.dtype(getattr(self.arrays, name).dtype) != dtype:
                raise ValueError(f"{name} dtype must match config.dtype. {rebuild}")
        if np.dtype(jnp.result_type(self.arrays.inv_permittivities,
                                   self.config.courant_number * self.config.resolution)) != dtype:
            raise ValueError(f"native Courant/resolution scalars promote config.dtype; use Python floats. {rebuild}")
        device_set = self.arrays.fields.E.devices()
        if len(device_set) != 1:
            raise ValueError("the extension supports one execution device, not multi-device sharded fields")
        for array in jax.tree.leaves((self.arrays.fields, self.arrays.alpha, self.arrays.kappa,
                                     self.arrays.sigma, self.arrays.inv_permittivities,
                                     self.arrays.inv_permeabilities)):
            if isinstance(array, jax.Array) and array.devices() != device_set:
                raise ValueError("all FDTD field and material arrays must share one execution device")
        q = _positive_array(self.arrays.inv_permittivities, "inverse permittivity")
        if q.shape != (1, *self.shape):
            raise ValueError("only one-component isotropic inverse permittivity is supported")
        mu = _positive_array(self.arrays.inv_permeabilities, "inverse permeability")
        if mu.ndim and mu.shape != (1, *self.shape):
            raise ValueError("permeability must be a fixed scalar or one-component isotropic array")
        if np.dtype(jnp.result_type(self.arrays.inv_permittivities, self.arrays.inv_permeabilities)) != dtype:
            raise ValueError(f"inverse permeability promotes config.dtype. {rebuild}")
        for name in ("electric_conductivity", "magnetic_conductivity", "dispersive_P_curr",
                     "dispersive_P_prev", "dispersive_c1", "dispersive_c2", "dispersive_c3",
                     "dispersive_inv_c2"):
            if getattr(self.arrays, name) is not None:
                raise ValueError(f"{name} is unsupported by the E-history adjoint")
        if not (self.objects.all_objects_non_dispersive
                and self.objects.all_objects_non_electrically_conductive
                and self.objects.all_objects_non_magnetically_conductive
                and self.objects.all_objects_isotropic_permittivity
                and self.objects.all_objects_isotropic_permeability):
            raise ValueError("all native object materials must be isotropic, lossless and nondispersive")
        self._validate_courant(float(q.max()), float(mu.max()))
        faces = {}
        for boundary in self.objects.boundary_objects:
            if type(boundary) not in (PerfectlyMatchedLayer, BlochBoundary, PerfectElectricConductor,
                                      PerfectMagneticConductor):
                raise ValueError("only native PML, PEC/PMC and Bloch-periodic boundaries are supported")
            if isinstance(boundary, BlochBoundary):
                if not np.isfinite(boundary.bloch_vector).all():
                    raise ValueError("Bloch vectors must be finite")
                if boundary.needs_complex_fields and not np.issubdtype(field_dtype, np.complexfloating):
                    raise ValueError("nonzero Bloch phase requires native complex field arrays")
            face = (boundary.axis, boundary.direction)
            if face in faces:
                raise ValueError("multiple boundaries on the same face are unsupported")
            faces[face] = boundary
        for axis in range(3):
            if (axis, "-") not in faces or (axis, "+") not in faces:
                raise ValueError("each Cartesian axis requires two explicit boundary faces")
            minus, plus = faces[axis, "-"], faces[axis, "+"]
            if (isinstance(minus, BlochBoundary) != isinstance(plus, BlochBoundary)):
                raise ValueError("periodic boundaries must occur on both faces of an axis")
            if isinstance(minus, BlochBoundary) and minus.bloch_vector[axis] != plus.bloch_vector[axis]:
                raise ValueError("paired Bloch boundaries must have the same wave vector")
        for source in self.objects.sources:
            if type(source) not in (UniformPlaneSource, GaussianPlaneSource, PointDipoleSource):
                raise ValueError("only native uniform plane, Gaussian plane and point dipole sources are validated")
            if type(source.temporal_profile) not in (CustomTimeSignalProfile, GaussianPulseProfile, SingleFrequencyProfile):
                raise ValueError("custom temporal-profile subclasses are not validated")
            required = (("_inv_eps_oriented", "_inv_mu_oriented") if type(source) is PointDipoleSource
                        else ("_E", "_H", "_time_offset_E", "_time_offset_H"))
            for name in required:
                value = getattr(source, name, None)
                if not isinstance(value, (jax.Array, np.ndarray, float, int)):
                    raise ValueError("source caches are uninitialized; call fdtdx.apply_params before SimulationSpec")
                if not np.all(np.isfinite(np.asarray(value))):
                    raise ValueError("source caches must contain finite fields")
                if np.iscomplexobj(np.asarray(value)) and not np.issubdtype(field_dtype, np.complexfloating):
                    raise ValueError("complex source caches require complex simulation fields")
                if np.asarray(value).real.dtype != dtype:
                    raise ValueError(f"source {name} cache precision differs from config.dtype. {rebuild}")
            if source._config.time_steps_total != self.time_steps or source._config.time_step_duration != self.dt:
                raise ValueError("source placement config differs from the simulation config")

    @property
    def time_steps(self):
        return self.config.time_steps_total

    @property
    def dt(self):
        return self.config.time_step_duration

    @property
    def shape(self):
        return tuple(self.objects.volume.grid_shape)

    def _validate_courant(self, max_inv_epsilon, max_inv_mu):
        # A conservative 3-D CFL bound also applies to thin periodic domains.
        factor = self.config.courant_factor * np.sqrt(max_inv_epsilon * max_inv_mu)
        if not np.isfinite(factor) or factor <= 0 or factor > 1 + 1e-12:
            raise ValueError("material wave speed exceeds the conservative native Cartesian CFL bound")


@dataclass(frozen=True)
class DesignGrid:
    """One rectangular scalar-epsilon design region on the native Yee grid.

    ``rho`` has exactly ``parameter_shape``. An optional differentiable JAX
    transform operates on this coarse array, then nearest-voxel repetition maps
    it to the simulation grid. Epsilon is interpolated linearly, not inverse
    epsilon. Native Device parameter transformations are not invoked implicitly.
    """

    start: tuple[int, int, int]
    shape: tuple[int, int, int]
    parameter_shape: tuple[int, int, int]
    epsilon_low: float = 1.0
    epsilon_high: float = 4.0
    transform: Callable | None = None

    def __post_init__(self):
        for name in ("start", "shape", "parameter_shape"):
            object.__setattr__(self, name, _triple(getattr(self, name), name, positive=name != "start"))
        if any(n % p for n, p in zip(self.shape, self.parameter_shape)):
            raise ValueError("each design grid size must be divisible by its parameter count")
        if not (np.isfinite(self.epsilon_low) and np.isfinite(self.epsilon_high)
                and 0 < self.epsilon_low < self.epsilon_high):
            raise ValueError("epsilon endpoints must be finite with 0 < epsilon_low < epsilon_high")
        if self.transform is not None and not callable(self.transform):
            raise TypeError("transform must be a differentiable callable or None")

    @property
    def region(self):
        return (slice(None), *(slice(s, s + n) for s, n in zip(self.start, self.shape)))

    def inverse_permittivity(self, rho):
        rho = jnp.asarray(rho)
        if rho.shape != self.parameter_shape:
            raise ValueError(f"rho must have shape {self.parameter_shape}, got {rho.shape}")
        if not jnp.issubdtype(rho.dtype, jnp.floating):
            raise ValueError("rho must be a real floating-point array")
        values = rho if self.transform is None else jnp.asarray(self.transform(rho))
        if values.shape != self.parameter_shape or not jnp.issubdtype(values.dtype, jnp.floating):
            raise ValueError("transform must preserve the real coarse parameter array shape")
        for axis, (n, p) in enumerate(zip(self.shape, self.parameter_shape)):
            values = jnp.repeat(values, n // p, axis=axis)
        epsilon = self.epsilon_low + (self.epsilon_high - self.epsilon_low) * values
        return jnp.reciprocal(epsilon)[None, ...]

    @classmethod
    def from_device(cls, placed_device, epsilon_low=1.0, epsilon_high=4.0, transform=None):
        """Reuse a placed native Device's extent and voxel sizes only.

        Endpoint materials are explicit; native inverse-epsilon interpolation is
        replaced by this extension's linear-epsilon map. Native transforms must
        be absent; pass an explicit JAX transform to this adapter instead.
        """
        from fdtdx.objects.device.device import Device

        if type(placed_device) is not Device:
            raise TypeError("from_device requires a placed native Device, not a custom subclass")
        if placed_device.param_transforms:
            raise ValueError("native Device transforms are not imported; use DesignGrid(transform=...) explicitly")
        permeabilities = [np.asarray(m.permeability) for m in placed_device.materials.values()]
        if any(not np.array_equal(permeabilities[0], mu) for mu in permeabilities[1:]):
            raise ValueError("Device endpoint permeability must be fixed across materials")
        return cls(tuple(s.start for s in placed_device.grid_slice), tuple(placed_device.grid_shape),
                   tuple(placed_device.matrix_voxel_grid_shape), epsilon_low, epsilon_high, transform)

    def validate(self, spec):
        """Reject geometry where the local E-only material contraction is invalid."""
        if any(s + n > full for s, n, full in zip(self.start, self.shape, spec.shape)):
            raise ValueError("design region lies outside the native simulation volume")
        for boundary in (*spec.objects.pml_objects, *spec.objects.pec_objects, *spec.objects.pmc_objects):
            if _overlaps(self.start, self.shape, boundary.grid_slice):
                raise ValueError("design region must not overlap PML or a PEC/PMC clamped boundary")
        for source in spec.objects.sources:
            if _overlaps(self.start, self.shape, source.grid_slice):
                raise ValueError("design region must not overlap a source or its cached local material")
        spec._validate_courant(max(float(np.asarray(spec.arrays.inv_permittivities).max()), 1 / self.epsilon_low),
                              float(np.asarray(spec.arrays.inv_permeabilities).max()))
        return self
