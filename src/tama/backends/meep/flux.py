from collections.abc import Mapping
from typing import Dict, Optional, Sequence, Tuple

import meep as mp
import numpy as np


_AXIS_COMPONENT_PAIRS = (
    ((mp.Ey, mp.Hz, 1.0), (mp.Ez, mp.Hy, -1.0)),
    ((mp.Ez, mp.Hx, 1.0), (mp.Ex, mp.Hz, -1.0)),
    ((mp.Ex, mp.Hy, 1.0), (mp.Ey, mp.Hx, -1.0)),
)

_CYLINDRICAL_AXIS_COMPONENT_PAIRS = {
    0: ((mp.Ep, mp.Hz, 1.0), (mp.Ez, mp.Hp, -1.0)),
    2: ((mp.Er, mp.Hp, 1.0), (mp.Ep, mp.Hr, -1.0)),
}


def _normalize_axis_normal(normal: mp.Vector3) -> Tuple[int, float]:
    if not all(hasattr(normal, axis) for axis in ("x", "y", "z")):
        raise TypeError("normal must be a Meep Vector3")
    raw_coordinates = np.asarray((normal.x, normal.y, normal.z))
    if np.iscomplexobj(raw_coordinates):
        raise TypeError("normal must have real coordinates")
    try:
        coordinates = np.asarray(raw_coordinates, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("normal must have numeric coordinates") from exc
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("normal must have finite coordinates")

    nonzero = np.flatnonzero(coordinates != 0.0)
    if nonzero.size != 1 or abs(coordinates[nonzero[0]]) != 1.0:
        raise ValueError(
            "normal must be a signed coordinate-axis unit vector"
        )
    axis = int(nonzero[0])
    return axis, float(coordinates[axis])


def flux_component_pairs(
    normal: mp.Vector3,
    *,
    cylindrical: bool = False,
) -> Tuple[Tuple[int, int, float], ...]:
    """Return the E/H component pairs in `n dot (conj(E) cross H)`."""
    axis, orientation = _normalize_axis_normal(normal)
    if cylindrical:
        if axis not in _CYLINDRICAL_AXIS_COMPONENT_PAIRS:
            raise ValueError(
                "cylindrical flux normals must be radial or axial"
            )
        component_pairs = _CYLINDRICAL_AXIS_COMPONENT_PAIRS[axis]
    else:
        component_pairs = _AXIS_COMPONENT_PAIRS[axis]
    return tuple(
        (electric, magnetic, orientation * sign)
        for electric, magnetic, sign in component_pairs
    )


def _validate_cylindrical_surface(
    positions,
    normal: mp.Vector3,
    spatial_weights: Sequence[float],
) -> None:
    """Validate an axis-aligned physical surface in the r-z half-plane."""
    axis, _ = _normalize_axis_normal(normal)
    flux_component_pairs(normal, cylindrical=True)
    coordinates = np.asarray(
        [(position.x, position.y, position.z) for position in positions],
        dtype=float,
    )
    weights = np.asarray(spatial_weights, dtype=float).reshape(-1)
    tolerance = 1.0e-12
    if np.any(coordinates[:, 0] < -tolerance):
        raise ValueError(
            "cylindrical surface positions must have nonnegative radius"
        )
    if np.any(np.abs(coordinates[:, 1]) > tolerance):
        raise ValueError(
            "cylindrical surface positions must lie in the r-z plane"
        )
    if axis == 0:
        if np.any(coordinates[:, 0] <= tolerance):
            raise ValueError(
                "radial cylindrical surfaces require positive radius"
            )
        if not np.allclose(
            coordinates[:, 0],
            coordinates[0, 0],
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError(
                "radial cylindrical surfaces require constant radius"
            )
    else:
        if not np.allclose(
            coordinates[:, 2],
            coordinates[0, 2],
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError(
                "axial cylindrical surfaces require constant z"
            )
        if np.any(
            (np.abs(coordinates[:, 0]) <= tolerance)
            & (weights != 0.0)
        ):
            raise ValueError(
                "axial cylindrical surface quadrature requires zero weight "
                "at r=0"
            )


def _normalized_histories(
    histories: Mapping[int, np.ndarray],
    normal: mp.Vector3,
    spatial_weights: Optional[Sequence[float]],
    *,
    cylindrical: bool = False,
):
    if not isinstance(histories, Mapping):
        raise TypeError("histories must map Meep components to arrays")

    active_pairs = []
    active_components = []
    for electric, magnetic, sign in flux_component_pairs(
        normal,
        cylindrical=cylindrical,
    ):
        has_electric = electric in histories
        has_magnetic = magnetic in histories
        if has_electric != has_magnetic:
            raise ValueError(
                "histories must contain both components of each E/H pair"
            )
        if has_electric:
            active_pairs.append((electric, magnetic, sign))
            for component in (electric, magnetic):
                if component not in active_components:
                    active_components.append(component)
    if not active_pairs:
        raise ValueError(
            "histories must contain at least one complete flux component pair"
        )

    arrays = {
        component: np.asarray(histories[component])
        for component in active_components
    }
    first = next(iter(arrays.values()))
    if first.ndim not in (1, 2):
        raise ValueError("field histories must be one- or two-dimensional")
    history_shape = first.shape
    if history_shape[0] < 2:
        raise ValueError("field histories must contain at least two samples")
    for history in arrays.values():
        if history.ndim not in (1, 2):
            raise ValueError(
                "field histories must be one- or two-dimensional"
            )
        if history.shape != history_shape:
            raise ValueError(
                "all field histories must have the same shape"
            )

    n_points = 1 if first.ndim == 1 else first.shape[1]
    if n_points == 0:
        raise ValueError("field histories must contain at least one point")
    matrices = {
        component: (
            history.reshape(history.shape[0], 1)
            if history.ndim == 1
            else history
        )
        for component, history in arrays.items()
    }
    if spatial_weights is None:
        weights = np.ones(n_points, dtype=float)
    else:
        raw_weights = np.asarray(spatial_weights)
        if np.iscomplexobj(raw_weights):
            raise TypeError("spatial_weights must be real")
        weights = np.asarray(raw_weights, dtype=float).reshape(-1)
        if weights.size != n_points:
            raise ValueError(
                "spatial_weights must match the history point count"
            )
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or not np.any(weights > 0.0)
        ):
            raise ValueError(
                "spatial_weights must be finite, nonnegative, "
                "and contain a positive value"
            )
    return active_pairs, matrices, weights, first.ndim == 1


def power_history(
    histories: Mapping[int, np.ndarray],
    normal: mp.Vector3,
    spatial_weights: Optional[Sequence[float]] = None,
    *,
    cylindrical: bool = False,
) -> np.ndarray:
    """Compute the signed time-domain Poynting functional at sample points.

    Real Cartesian histories represent sampled instantaneous power. Complex or
    cylindrical histories represent the Hermitian modal functional defined by
    the E/H pairing and supplied quadrature weights. Electric histories are
    evaluated at samples `0 .. N-2`. Magnetic
    histories are centered onto those samples as
    `0.5 * (H[:-1] + H[1:])`.
    """
    active_pairs, matrices, weights, _ = _normalized_histories(
        histories,
        normal,
        spatial_weights,
        cylindrical=cylindrical,
    )
    n_samples = next(iter(matrices.values())).shape[0]
    power = np.zeros(n_samples - 1, dtype=float)
    for electric, magnetic, sign in active_pairs:
        electric_history = matrices[electric][:-1]
        magnetic_history = 0.5 * (
            matrices[magnetic][:-1] + matrices[magnetic][1:]
        )
        power += sign * np.sum(
            weights
            * np.real(np.conjugate(electric_history) * magnetic_history),
            axis=1,
        )
    return power


def power_history_pullback(
    histories: Mapping[int, np.ndarray],
    normal: mp.Vector3,
    spatial_weights: Optional[Sequence[float]],
    power_covector: np.ndarray,
    *,
    cylindrical: bool = False,
) -> Dict[int, np.ndarray]:
    """Pull a power-trace covector back to raw E/H history covectors.

    Returned arrays use TAMA's bilinear complex-covector convention:
    `dJ = Re(sum(component_covector * d_component))`.
    """
    active_pairs, matrices, weights, was_one_dimensional = (
        _normalized_histories(
            histories,
            normal,
            spatial_weights,
            cylindrical=cylindrical,
        )
    )
    n_samples, n_points = next(iter(matrices.values())).shape
    raw_covector = np.asarray(power_covector)
    if raw_covector.shape != (n_samples - 1,):
        raise ValueError(
            "power_covector must match the power history shape"
        )
    if np.iscomplexobj(raw_covector) and np.any(
        np.imag(raw_covector) != 0.0
    ):
        raise ValueError("power_covector must be real")
    time_weights = np.asarray(np.real(raw_covector), dtype=float).reshape(
        n_samples - 1,
        1,
    )

    dtype = np.result_type(
        *(history.dtype for history in matrices.values()),
        time_weights.dtype,
    )
    covectors = {
        component: np.zeros((n_samples, n_points), dtype=dtype)
        for component in matrices
    }
    point_weights = weights.reshape(1, n_points)
    for electric, magnetic, sign in active_pairs:
        electric_history = matrices[electric][:-1]
        magnetic_history = 0.5 * (
            matrices[magnetic][:-1] + matrices[magnetic][1:]
        )
        electric_covector = (
            sign
            * time_weights
            * point_weights
            * np.conjugate(magnetic_history)
        )
        centered_magnetic_covector = (
            sign
            * time_weights
            * point_weights
            * np.conjugate(electric_history)
        )
        covectors[electric][:-1] += electric_covector
        covectors[magnetic][:-1] += 0.5 * centered_magnetic_covector
        covectors[magnetic][1:] += 0.5 * centered_magnetic_covector

    if was_one_dimensional:
        return {
            component: covector[:, 0]
            for component, covector in covectors.items()
        }
    return covectors
