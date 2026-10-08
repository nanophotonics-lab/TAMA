"""Cartesian objectives on native FDTDX co-located, time-centered fields.

Indices refer to native grid cells, not SI positions or Meep coordinates.
The backend supplies E and normalized H (not SI amperes/metre), already
spatially interpolated and temporally centered. Flux is in solver units; no
watts, source normalization, or automatic surface quadrature is implied.
"""

from dataclasses import dataclass
from numbers import Integral, Real

import jax
import jax.numpy as jnp
import numpy as np

_COMPONENTS = {
    name: index for index, name in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))
}


def _real_scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(
            f"{name} must be finite" + (" and positive" if positive else "")
        )
    return value


def _indices(values):
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.shape[0] == 0:
        raise ValueError("indices must have nonempty shape (N, 3)")
    if any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in raw.flat
    ):
        raise TypeError("indices must contain integers, not booleans or coordinates")
    indices = np.asarray(raw, dtype=np.int64)
    if np.any(indices < 0):
        raise ValueError("indices must be nonnegative")
    indices.setflags(write=False)
    return indices


def _weights(values, count):
    raw = np.asarray(values)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "iuf":
        raise TypeError("weights must be real")
    weights = np.array(raw, dtype=float, copy=True)
    if weights.shape != (count,) or not np.all(np.isfinite(weights)):
        raise ValueError("weights must have finite shape (N,)")
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("weights must be nonnegative with at least one positive entry")
    weights.setflags(write=False)
    return weights


class _Target:
    """Immutable configuration: construct a new target when its settings change."""

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError(
                "target configuration is immutable; construct a new target"
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if getattr(self, "_frozen", False):
            raise AttributeError(
                "target configuration is immutable; construct a new target"
            )
        object.__delattr__(self, name)

    def validate_shape(self, grid_shape):
        raw = np.asarray(grid_shape, dtype=object)
        if raw.shape != (3,) or any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 1
            for v in raw
        ):
            raise ValueError(
                "grid_shape must contain three positive integer dimensions"
            )
        if np.any(self.indices >= np.asarray(raw, dtype=int)):
            raise ValueError("target indices lie outside the native grid")

    def sample(self, E, H):
        """Sample six native co-located components in Ex,Ey,Ez,Hx,Hy,Hz order."""
        if E.shape != H.shape or E.ndim != 4 or E.shape[0] != 3:
            raise ValueError("E and H must both have shape (3, Nx, Ny, Nz)")
        if not jnp.issubdtype(E.dtype, jnp.inexact) or not jnp.issubdtype(
            H.dtype, jnp.inexact
        ):
            raise TypeError("E and H must contain floating-point fields")
        self.validate_shape(E.shape[1:])
        indices = tuple(self.indices[:, axis] for axis in range(3))
        return jnp.concatenate(
            (E[(slice(None), *indices)].T, H[(slice(None), *indices)].T), axis=-1
        )

    def _history(self, history):
        history = jnp.asarray(history)
        if (
            history.ndim != 3
            or history.shape[0] == 0
            or history.shape[1:] != (len(self.indices), 6)
        ):
            raise ValueError("history must have shape (Nt, N, 6), with Nt > 0")
        if not jnp.issubdtype(history.dtype, jnp.inexact):
            raise TypeError(
                "history must contain real or complex floating-point fields"
            )
        return history


class FieldRegionTarget(_Target):
    """Weighted integral of one field component, dt/2 sum(w * abs(field)**2).

    Weights are explicit sample measures; they do not automatically include
    cell volume. A Band.fom_fn receives all six unweighted field components.
    """

    def __init__(self, indices, component="Ez", weights=None):
        self.indices = _indices(indices)
        if not isinstance(component, str) or component not in _COMPONENTS:
            raise ValueError("component must be Ex, Ey, Ez, Hx, Hy, or Hz")
        self.component = component
        self.weights = _weights(
            np.ones(len(self.indices)) if weights is None else weights,
            len(self.indices),
        )
        self._frozen = True

    def functional(self, history, dt):
        history = self._history(history)
        dt = _real_scalar(dt, "dt", positive=True)
        component = history[..., _COMPONENTS[self.component]]
        return (
            dt
            / 2
            * jnp.sum(
                jnp.asarray(self.weights, dtype=history.real.dtype)
                * jnp.abs(component) ** 2
            )
        )


class PointTarget(FieldRegionTarget):
    def __init__(self, index, component="Ez"):
        super().__init__([index], component=component)


class FluxTarget(_Target):
    """Signed dt sum(w * Re(E cross conj(H)).normal) in native solver units.

    Supply explicit dA quadrature weights (or a 2D line measure per unit
    invariant-axis length). All indices must lie in one axis-normal plane.
    The caller must not center H a second time. For complex histories this is
    a modal Poynting functional, not an automatically normalized physical power.
    """

    def __init__(self, indices, normal, weights):
        self.indices = _indices(indices)
        self.weights = _weights(weights, len(self.indices))
        normal = np.asarray(normal)
        if (
            normal.shape != (3,)
            or np.iscomplexobj(normal)
            or normal.dtype.kind not in "iuf"
        ):
            raise ValueError("normal must be a signed Cartesian axis unit vector")
        if np.count_nonzero(normal) != 1 or not np.isin(normal, [-1, 0, 1]).all():
            raise ValueError("normal must be a signed Cartesian axis unit vector")
        self.normal = np.array(normal, dtype=float, copy=True)
        self.normal.setflags(write=False)
        axis = int(np.flatnonzero(self.normal)[0])
        if np.any(self.indices[:, axis] != self.indices[0, axis]):
            raise ValueError("surface indices must lie in one plane normal to normal")
        self._frozen = True

    def power_trace(self, history):
        history = self._history(history)
        poynting = jnp.real(jnp.cross(history[..., :3], jnp.conj(history[..., 3:])))
        return jnp.sum(
            poynting
            * jnp.asarray(self.normal, dtype=history.real.dtype)
            * jnp.asarray(self.weights, dtype=history.real.dtype)[None, :, None],
            axis=(1, 2),
        )

    def functional(self, history, dt):
        return _real_scalar(dt, "dt", positive=True) * jnp.sum(
            self.power_trace(history)
        )


class EigenmodeCoefficientTarget(FluxTarget):
    """Fixed-profile overlap amplitude; this class does not solve eigenmodes.

    a = integral((E cross conj(Hm) + conj(Em) cross H).n)/(4P),
    P = integral(Re(Em cross conj(Hm)).n)/2. Reference fields must use native
    normalized-H units and the same co-located sampling convention. A supplied
    self-mode has amplitude one. Default objective is dt sum(abs(a)**2), NOT
    SI power and NOT Meep TAMA's power-normalized coefficient. Profiles are
    fixed and may be complex. Complex simulation fields require separate
    validation of the backend; these functionals do not establish that support.
    """

    def __init__(self, indices, normal, weights, reference_E, reference_H):
        super().__init__(indices, normal, weights)
        profiles = []
        for name, values in (
            ("reference_E", reference_E),
            ("reference_H", reference_H),
        ):
            values = np.asarray(values)
            if (
                values.shape != (len(self.indices), 3)
                or values.dtype.kind not in "iufc"
                or not np.all(np.isfinite(values))
            ):
                raise ValueError(f"{name} must have finite numeric shape (N, 3)")
            profile = np.array(values, copy=True)
            profile.setflags(write=False)
            profiles.append(profile)
        object.__setattr__(self, "reference_E", profiles[0])
        object.__setattr__(self, "reference_H", profiles[1])
        object.__setattr__(
            self,
            "reference_power",
            float(
                0.5
                * np.sum(
                    self.weights[:, None]
                    * self.normal
                    * np.real(
                        np.cross(self.reference_E, np.conjugate(self.reference_H))
                    )
                )
            ),
        )
        if not np.isfinite(self.reference_power) or self.reference_power <= 0:
            raise ValueError(
                "reference mode must carry positive finite power along normal"
            )

    def coefficient(self, history):
        history = self._history(history)
        dtype = jnp.result_type(history.dtype, jnp.complex64)
        E_mode = jnp.asarray(self.reference_E, dtype=dtype)
        H_mode = jnp.asarray(self.reference_H, dtype=dtype)
        overlap = jnp.cross(history[..., :3], jnp.conj(H_mode)) + jnp.cross(
            jnp.conj(E_mode), history[..., 3:]
        )
        return jnp.sum(
            overlap
            * jnp.asarray(self.normal, dtype=history.dtype)
            * jnp.asarray(self.weights, dtype=history.dtype)[None, :, None],
            axis=(1, 2),
        ) / (4 * self.reference_power)

    def functional(self, history, dt):
        return _real_scalar(dt, "dt", positive=True) * jnp.sum(
            jnp.abs(self.coefficient(history)) ** 2
        )


@dataclass(frozen=True, eq=False)
class Band:
    """One objective term; weight is objective priority, not filter amplitude.

    A kernel has continuous-time convolution units: output = dt * convolve
    (history, kernel, full). None means identity, with no extra dt factor.
    fom_fn, when given, receives (filtered_history[Nt+K-1,N,6], dt), must be
    JAX-differentiable, and must return a real scalar. Callback closures must
    remain fixed after objective construction, as required by JAX JIT capture.
    """

    target: int
    kernel: object = None
    weight: float = 1.0
    fom_fn: object = None

    def __post_init__(self):
        if (
            isinstance(self.target, (bool, np.bool_))
            or not isinstance(self.target, Integral)
            or self.target < 0
        ):
            raise ValueError("target must be a nonnegative integer target index")
        object.__setattr__(self, "weight", _real_scalar(self.weight, "weight"))
        if self.kernel is not None:
            kernel = np.asarray(self.kernel)
            if (
                kernel.ndim != 1
                or kernel.size == 0
                or kernel.dtype.kind not in "iufc"
                or not np.all(np.isfinite(kernel))
            ):
                raise ValueError("kernel must be a nonempty finite numeric vector")
            kernel = np.array(
                kernel, dtype=complex if np.iscomplexobj(kernel) else float, copy=True
            )
            kernel.setflags(write=False)
            object.__setattr__(self, "kernel", kernel)
        if self.fom_fn is not None and not callable(self.fom_fn):
            raise TypeError("fom_fn must be callable")


def make_loss(targets, bands, dt, scalarization_fn=None):
    """Build a differentiable real scalar objective of a tuple of histories.

    Each Band independently filters all constituent E/H histories before its
    functional. All N+K-1 convolution samples contribute. Scalarization sees
    the vector of weighted band functionals; the default is their sum. Thus
    unequal priorities never change the spectral response of the FIR bank.
    Scalarization closures must remain fixed after objective construction.
    """
    targets, bands = tuple(targets), tuple(bands)
    dt = _real_scalar(dt, "dt", positive=True)
    allowed_types = (
        PointTarget,
        FieldRegionTarget,
        FluxTarget,
        EigenmodeCoefficientTarget,
    )
    if not targets or not all(type(t) in allowed_types for t in targets):
        raise ValueError(
            "targets must use the four built-in target types; custom sampling subclasses are unsupported"
        )
    if not bands or not all(
        isinstance(b, Band) and b.target < len(targets) for b in bands
    ):
        raise ValueError("bands must contain Band objects indexing existing targets")
    if scalarization_fn is not None and not callable(scalarization_fn):
        raise TypeError("scalarization_fn must be callable")

    def loss(traces):
        if len(traces) != len(targets):
            raise ValueError("one history is required per target")
        histories = tuple(
            target._history(trace) for target, trace in zip(targets, traces)
        )
        if len({h.shape[0] for h in histories}) != 1:
            raise ValueError("target histories must share one time grid")
        values = []
        for band in bands:
            history = histories[band.target]
            if band.kernel is not None:
                # Scale before casting, avoiding overflow for optical-frequency FIR coefficients.
                dtype = (
                    jnp.result_type(history.dtype, jnp.complex64)
                    if np.iscomplexobj(band.kernel)
                    else history.dtype
                )
                kernel = jnp.asarray(dt * band.kernel, dtype=dtype)
                flat = history.reshape(history.shape[0], -1)
                filtered = jax.vmap(
                    lambda signal: jnp.convolve(signal, kernel, mode="full"), 1, 1
                )(flat)
                history = filtered.reshape(filtered.shape[0], *history.shape[1:])
            functional = band.fom_fn or targets[band.target].functional
            value = jnp.asarray(functional(history, dt))
            if value.shape != () or not jnp.issubdtype(value.dtype, jnp.floating):
                raise ValueError(
                    "each band functional must return a real floating-point scalar"
                )
            values.append(band.weight * value)
        vector = jnp.stack(values)
        result = jnp.asarray(
            jnp.sum(vector) if scalarization_fn is None else scalarization_fn(vector)
        )
        if result.shape != () or not jnp.issubdtype(result.dtype, jnp.floating):
            raise ValueError("scalarization must return a real floating-point scalar")
        return result

    return loss
