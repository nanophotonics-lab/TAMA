from collections.abc import Mapping
from typing import Dict, Optional, Sequence

import meep as mp
import numpy as np

from .flux import (
    _normalize_axis_normal,
    flux_component_pairs,
)
from .objectives import (
    _is_cartesian_2d_simulation,
    _is_cylindrical_simulation,
)


_AXIS_DIRECTIONS = (mp.X, mp.Y, mp.Z)


def _spatial_weights(
    spatial_weights: Optional[Sequence[float]],
    n_points: int,
) -> np.ndarray:
    if spatial_weights is None:
        return np.ones(n_points, dtype=float)

    raw_weights = np.asarray(spatial_weights)
    if np.iscomplexobj(raw_weights):
        raise TypeError("spatial_weights must be real")
    weights = np.asarray(raw_weights, dtype=float).reshape(-1)
    if weights.size != n_points:
        raise ValueError("spatial_weights must match the mode point count")
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or not np.any(weights > 0.0)
    ):
        raise ValueError(
            "spatial_weights must be finite, nonnegative, "
            "and contain a positive value"
        )
    return weights


def _normalized_mode_fields(
    mode_fields: Mapping[int, np.ndarray],
    normal: mp.Vector3,
    spatial_weights: Optional[Sequence[float]],
    *,
    cylindrical: bool = False,
):
    if not isinstance(mode_fields, Mapping):
        raise TypeError("mode_fields must map Meep components to arrays")

    active_pairs = []
    active_components = []
    for electric, magnetic, sign in flux_component_pairs(
        normal,
        cylindrical=cylindrical,
    ):
        has_electric = electric in mode_fields
        has_magnetic = magnetic in mode_fields
        if has_electric != has_magnetic:
            raise ValueError(
                "mode_fields must contain both components of each E/H pair"
            )
        if has_electric:
            active_pairs.append((electric, magnetic, sign))
            for component in (electric, magnetic):
                if component not in active_components:
                    active_components.append(component)
    if not active_pairs:
        raise ValueError(
            "mode_fields must contain at least one complete component pair"
        )

    arrays = {
        component: np.asarray(mode_fields[component])
        for component in active_components
    }
    first = next(iter(arrays.values()))
    if first.ndim != 1 or first.size == 0:
        raise ValueError(
            "mode field arrays must be nonempty and one-dimensional"
        )
    for mode_field in arrays.values():
        if mode_field.ndim != 1:
            raise ValueError("mode field arrays must be one-dimensional")
        if mode_field.shape != first.shape:
            raise ValueError("all mode field arrays must have the same shape")

    weights = _spatial_weights(spatial_weights, first.size)
    return active_pairs, arrays, weights


def _mode_power(
    active_pairs,
    mode_fields: Mapping[int, np.ndarray],
    weights: np.ndarray,
) -> float:
    signed_power = 0.0
    absolute_power = 0.0
    for electric, magnetic, sign in active_pairs:
        contributions = (
            sign
            * weights
            * np.conjugate(mode_fields[electric])
            * mode_fields[magnetic]
        )
        signed_power += float(np.sum(np.real(contributions)))
        absolute_power += float(np.sum(np.abs(contributions)))

    if not np.isfinite(signed_power) or not np.isfinite(absolute_power):
        raise ValueError("mode fields must produce finite power")
    tolerance = 64.0 * np.finfo(float).eps * absolute_power
    if absolute_power == 0.0 or abs(signed_power) <= tolerance:
        raise ValueError("mode fields must carry nonzero signed power")
    return signed_power


def mode_power(
    mode_fields: Mapping[int, np.ndarray],
    normal: mp.Vector3,
    spatial_weights: Optional[Sequence[float]] = None,
    *,
    cylindrical: bool = False,
) -> float:
    """Return signed power of a fixed mode on the supplied quadrature."""
    active_pairs, arrays, weights = _normalized_mode_fields(
        mode_fields,
        normal,
        spatial_weights,
        cylindrical=cylindrical,
    )
    return _mode_power(active_pairs, arrays, weights)


def _sample_mode_fields(mode, positions, normal, *, cylindrical=False):
    components = tuple(
        dict.fromkeys(
            component
            for electric, magnetic, _ in flux_component_pairs(
                normal,
                cylindrical=cylindrical,
            )
            for component in (electric, magnetic)
        )
    )
    fields = {
        component: np.asarray(
            [
                mode.amplitude(position, component)
                for position in positions
            ],
            dtype=np.complex128,
        )
        for component in components
    }

    gauge_component, gauge_index = max(
        (
            (component, int(np.argmax(np.abs(values))))
            for component, values in fields.items()
        ),
        key=lambda item: abs(fields[item[0]][item[1]]),
    )
    gauge_value = fields[gauge_component][gauge_index]
    if gauge_value == 0.0:
        raise ValueError("eigenmode fields vanish at every target position")
    gauge = np.exp(-1j * np.angle(gauge_value))
    return {
        component: np.asarray(gauge * values, dtype=np.complex128)
        for component, values in fields.items()
    }


def sample_reference_mode(sim: mp.Simulation, target) -> Dict[int, np.ndarray]:
    """Solve and sample one fixed mode on every target quadrature point."""
    is_cylindrical = _is_cylindrical_simulation(sim)
    supplied_fields = getattr(target, "reference_mode_fields", None)
    if supplied_fields is not None:
        if is_cylindrical:
            reference_m = getattr(target, "reference_m", None)
            if reference_m is None:
                raise ValueError(
                    "cylindrical EigenmodeCoefficientTarget reference_m is "
                    "required"
                )
            simulation_m = getattr(sim, "m", 0)
            simulation_m = 0 if simulation_m is None else int(simulation_m)
            if int(reference_m) != simulation_m:
                raise ValueError(
                    "cylindrical EigenmodeCoefficientTarget reference_m must "
                    "match the forward simulation m"
                )
        mode_fields = {
            int(component): np.asarray(values, dtype=np.complex128)
            for component, values in supplied_fields.items()
        }
        required_components = {
            component
            for electric, magnetic, _ in flux_component_pairs(
                target.normal,
                cylindrical=is_cylindrical,
            )
            for component in (electric, magnetic)
        }
        if set(mode_fields) != required_components:
            raise ValueError(
                "EigenmodeCoefficientTarget reference_mode_fields must "
                "contain both complete tangential E/H pairs"
            )
        power = mode_power(
            mode_fields,
            target.normal,
            target.spatial_weights,
            cylindrical=is_cylindrical,
        )
        if power <= 0.0:
            raise ValueError(
                "reference mode does not carry positive power along normal"
            )
        return mode_fields
    if is_cylindrical:
        raise ValueError(
            "cylindrical EigenmodeCoefficientTarget requires "
            "reference_mode_fields because Meep's eigenmode solver does not "
            "support cylindrical simulations"
        )

    axis, orientation = _normalize_axis_normal(target.normal)
    region_size = np.asarray(
        (
            target.mode_region.size.x,
            target.mode_region.size.y,
            target.mode_region.size.z,
        ),
        dtype=float,
    )
    tangential_axes = tuple(index for index in range(3) if index != axis)
    if _is_cartesian_2d_simulation(sim):
        in_plane_tangent = 1 - axis
        valid_region = (
            axis in (0, 1)
            and region_size[in_plane_tangent] > 1.0e-12
            and region_size[2] <= 1.0e-12
        )
        dimension = "in-plane line"
    else:
        valid_region = all(
            region_size[index] > 1.0e-12
            for index in tangential_axes
        )
        dimension = "plane"
    if not valid_region:
        raise ValueError(
            "EigenmodeCoefficientTarget mode_region must be a "
            f"{dimension} in this simulation"
        )

    if getattr(sim, "fields", None) is None:
        sim.init_sim()
    mode = sim.get_eigenmode(
        target.frequency,
        _AXIS_DIRECTIONS[axis],
        target.mode_region,
        target.mode,
        mp.Vector3(),
        parity=target.eig_parity,
        resolution=target.eig_resolution,
        eigensolver_tol=target.eig_tolerance,
    )
    if orientation * float(mode.group_velocity) <= 0.0:
        kpoint = mode.k
        mode = sim.get_eigenmode(
            target.frequency,
            _AXIS_DIRECTIONS[axis],
            target.mode_region,
            target.mode,
            mp.Vector3(-kpoint.x, -kpoint.y, -kpoint.z),
            match_frequency=False,
            parity=target.eig_parity,
            resolution=target.eig_resolution,
            eigensolver_tol=target.eig_tolerance,
        )

    frequency_scale = max(1.0, abs(target.frequency))
    frequency_tolerance = max(
        1.0e-10,
        100.0 * target.eig_tolerance,
    ) * frequency_scale
    selected_frequency = float(mode.freq)
    if (
        not np.isfinite(selected_frequency)
        or abs(selected_frequency - target.frequency) > frequency_tolerance
    ):
        raise ValueError(
            "selected eigenmode frequency does not match target frequency; "
            "negative-normal targets require a reciprocal port medium"
        )
    selected_group_velocity = float(mode.group_velocity)
    if (
        not np.isfinite(selected_group_velocity)
        or orientation * selected_group_velocity <= 0.0
    ):
        raise ValueError(
            "selected eigenmode group velocity does not point along normal"
        )

    mode_fields = _sample_mode_fields(
        mode,
        target.positions,
        target.normal,
        cylindrical=is_cylindrical,
    )
    power = mode_power(
        mode_fields,
        target.normal,
        target.spatial_weights,
        cylindrical=is_cylindrical,
    )
    if power <= 0.0:
        raise ValueError(
            "selected eigenmode does not carry positive power along normal"
        )
    return mode_fields


def _normalized_inputs(
    histories: Mapping[int, np.ndarray],
    mode_fields: Mapping[int, np.ndarray],
    normal: mp.Vector3,
    spatial_weights: Optional[Sequence[float]],
    *,
    cylindrical: bool = False,
):
    active_pairs, modes, weights = _normalized_mode_fields(
        mode_fields,
        normal,
        spatial_weights,
        cylindrical=cylindrical,
    )
    if not isinstance(histories, Mapping):
        raise TypeError("histories must map Meep components to arrays")

    active_components = tuple(modes)
    expected = set(active_components)
    for electric, magnetic, _ in flux_component_pairs(
        normal,
        cylindrical=cylindrical,
    ):
        has_electric = electric in histories
        has_magnetic = magnetic in histories
        if has_electric != has_magnetic:
            raise ValueError(
                "histories must contain both components of each E/H pair"
            )
        if has_electric and {electric, magnetic}.isdisjoint(expected):
            raise ValueError(
                "histories and mode_fields must contain the same component pairs"
            )
    if any(component not in histories for component in active_components):
        raise ValueError(
            "histories and mode_fields must contain the same component pairs"
        )

    arrays = {
        component: np.asarray(histories[component])
        for component in active_components
    }
    first = next(iter(arrays.values()))
    if first.ndim not in (1, 2):
        raise ValueError("field histories must be one- or two-dimensional")
    if first.shape[0] < 2:
        raise ValueError("field histories must contain at least two samples")
    for history in arrays.values():
        if history.ndim not in (1, 2):
            raise ValueError(
                "field histories must be one- or two-dimensional"
            )
        if history.shape != first.shape:
            raise ValueError("all field histories must have the same shape")

    n_points = 1 if first.ndim == 1 else first.shape[1]
    if n_points != weights.size:
        raise ValueError(
            "field history point count must match the mode point count"
        )
    matrices = {
        component: (
            history.reshape(history.shape[0], 1)
            if history.ndim == 1
            else history
        )
        for component, history in arrays.items()
    }
    signed_power = _mode_power(active_pairs, modes, weights)
    normalization = 1.0 / np.sqrt(abs(signed_power))
    return (
        active_pairs,
        matrices,
        modes,
        weights,
        normalization,
        first.ndim == 1,
    )


def _projection_weights_from_normalized(
    active_pairs,
    modes: Mapping[int, np.ndarray],
    weights: np.ndarray,
    normalization: float,
):
    electric_weights = {}
    magnetic_weights = {}
    for electric, magnetic, sign in active_pairs:
        common = 0.5 * normalization * sign * weights
        electric_weights[electric] = (
            common * np.conjugate(modes[magnetic])
        )
        magnetic_weights[magnetic] = (
            common * np.conjugate(modes[electric])
        )
    return electric_weights, magnetic_weights


def _coefficient_projection_weights(
    mode_fields: Mapping[int, np.ndarray],
    normal: mp.Vector3,
    spatial_weights: Optional[Sequence[float]] = None,
    *,
    cylindrical: bool = False,
):
    """Return fixed spatial weights for electric and magnetic overlaps."""
    active_pairs, modes, weights = _normalized_mode_fields(
        mode_fields,
        normal,
        spatial_weights,
        cylindrical=cylindrical,
    )
    normalization = 1.0 / np.sqrt(
        abs(_mode_power(active_pairs, modes, weights))
    )
    return _projection_weights_from_normalized(
        active_pairs,
        modes,
        weights,
        normalization,
    )


def _coefficient_history_from_overlaps(
    overlap_history: np.ndarray,
) -> np.ndarray:
    """Center the magnetic overlap onto the electric time grid."""
    overlaps = np.asarray(overlap_history)
    if overlaps.ndim != 2 or overlaps.shape[1] != 2:
        raise ValueError("overlap_history must have shape (n_samples, 2)")
    if overlaps.shape[0] < 2:
        raise ValueError("overlap_history must contain at least two samples")
    return overlaps[:-1, 0] + 0.5 * (
        overlaps[:-1, 1] + overlaps[1:, 1]
    )


def _coefficient_overlap_pullback(
    coefficient_covector: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    """Pull a coefficient covector back to electric/magnetic overlaps."""
    if (
        isinstance(sample_count, (bool, np.bool_))
        or not isinstance(sample_count, (int, np.integer))
        or sample_count < 2
    ):
        raise ValueError("sample_count must be an integer of at least two")
    sample_count = int(sample_count)
    covector = np.asarray(coefficient_covector)
    if covector.shape != (sample_count - 1,):
        raise ValueError(
            "coefficient_covector must have shape (sample_count - 1,)"
        )
    pulled_back = np.zeros(
        (sample_count, 2),
        dtype=np.result_type(covector.dtype, np.complex128),
    )
    pulled_back[:-1, 0] = covector
    pulled_back[:-1, 1] += 0.5 * covector
    pulled_back[1:, 1] += 0.5 * covector
    return pulled_back


def coefficient_history(
    histories: Mapping[int, np.ndarray],
    mode_fields: Mapping[int, np.ndarray],
    normal: mp.Vector3,
    spatial_weights: Optional[Sequence[float]] = None,
    *,
    cylindrical: bool = False,
) -> np.ndarray:
    """Project time-domain fields onto a fixed unit-power mode.

    Electric histories are evaluated at samples `0 .. N-2`. Magnetic
    histories are centered onto those samples as
    `0.5 * (H[:-1] + H[1:])`.
    """
    (
        active_pairs,
        matrices,
        modes,
        weights,
        normalization,
        _,
    ) = _normalized_inputs(
        histories,
        mode_fields,
        normal,
        spatial_weights,
        cylindrical=cylindrical,
    )
    n_samples = next(iter(matrices.values())).shape[0]
    dtype = np.result_type(
        *(history.dtype for history in matrices.values()),
        *(mode.dtype for mode in modes.values()),
        np.complex128,
    )
    electric_weights, magnetic_weights = _projection_weights_from_normalized(
        active_pairs,
        modes,
        weights,
        normalization,
    )
    overlaps = np.zeros((n_samples, 2), dtype=dtype)
    for component, component_weights in electric_weights.items():
        overlaps[:, 0] += np.sum(
            matrices[component] * component_weights,
            axis=1,
        )
    for component, component_weights in magnetic_weights.items():
        overlaps[:, 1] += np.sum(
            matrices[component] * component_weights,
            axis=1,
        )
    return _coefficient_history_from_overlaps(overlaps)


def coefficient_history_pullback(
    histories: Mapping[int, np.ndarray],
    mode_fields: Mapping[int, np.ndarray],
    normal: mp.Vector3,
    spatial_weights: Optional[Sequence[float]],
    coefficient_covector: np.ndarray,
    *,
    cylindrical: bool = False,
) -> Dict[int, np.ndarray]:
    """Pull a coefficient-history covector back to raw E/H histories.

    The fixed mode is not differentiated. Returned arrays use TAMA's
    bilinear complex-covector convention:
    `dJ = Re(sum(component_covector * d_component))`.
    """
    (
        active_pairs,
        matrices,
        modes,
        weights,
        normalization,
        was_one_dimensional,
    ) = _normalized_inputs(
        histories,
        mode_fields,
        normal,
        spatial_weights,
        cylindrical=cylindrical,
    )
    n_samples, n_points = next(iter(matrices.values())).shape
    covector = np.asarray(coefficient_covector)
    if covector.shape != (n_samples - 1,):
        raise ValueError(
            "coefficient_covector must match the coefficient history shape"
        )

    dtype = np.result_type(
        *(history.dtype for history in matrices.values()),
        *(mode.dtype for mode in modes.values()),
        covector.dtype,
        np.complex128,
    )
    overlap_covectors = _coefficient_overlap_pullback(covector, n_samples)
    electric_weights, magnetic_weights = _projection_weights_from_normalized(
        active_pairs,
        modes,
        weights,
        normalization,
    )
    pulled_back = {
        component: np.zeros((n_samples, n_points), dtype=dtype)
        for component in matrices
    }
    for component, component_weights in electric_weights.items():
        pulled_back[component] = (
            overlap_covectors[:, 0, np.newaxis]
            * component_weights[np.newaxis, :]
        )
    for component, component_weights in magnetic_weights.items():
        pulled_back[component] = (
            overlap_covectors[:, 1, np.newaxis]
            * component_weights[np.newaxis, :]
        )

    if was_one_dimensional:
        return {
            component: component_covector[:, 0]
            for component, component_covector in pulled_back.items()
        }
    return pulled_back
