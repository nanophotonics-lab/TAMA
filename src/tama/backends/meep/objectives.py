from typing import Optional

import meep as mp
import numpy as np
from numpy.typing import NDArray
import scipy.interpolate as spi

from .sampling_grid import (
    FastFieldGrid,
    make_native_tabulated_cubic,
    make_native_tabulated_bspline_bank,
    make_native_tabulated_real_bspline_bank,
    make_native_tabulated_real_cubic_bank,
    merge_indexed_sourcedata,
    shift_native_tabulated_bspline,
)


def _tabulated_cubic_source(times: np.ndarray, values: np.ndarray):
    """Return a native cubic callable."""
    if np.asarray(times).size >= 4:
        spline = spi.make_interp_spline(times, values, k=3)
        coefficients = np.asarray(spline.c)
        if coefficients.ndim == 1:
            coefficients = coefficients[:, np.newaxis]
        return make_native_tabulated_bspline_bank(
            spline.t,
            coefficients,
        )[0]
    spline = spi.CubicSpline(times, values, extrapolate=False)
    return make_native_tabulated_cubic(spline.x, spline.c)


def _tabulated_cubic_sources(times: np.ndarray, values: np.ndarray):
    """Return channel callables sharing one compact coefficient bank."""
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("cubic source values must have shape (time, channel)")
    if values.shape[0] != np.asarray(times).size:
        raise ValueError("cubic source times must match the value rows")
    if values.shape[1] == 0:
        return ()
    if values.shape[0] < 4:
        if np.iscomplexobj(values) and np.any(values.imag != 0.0):
            return tuple(
                _tabulated_cubic_source(times, values[:, channel])
                for channel in range(values.shape[1])
            )
        spline = spi.CubicSpline(
            times,
            np.asarray(values.real, dtype=np.float64),
            axis=0,
            extrapolate=False,
        )
        return make_native_tabulated_real_cubic_bank(spline.x, spline.c)
    complex_values = bool(
        np.iscomplexobj(values) and np.any(values.imag != 0.0)
    )
    spline = spi.make_interp_spline(
        times,
        values if complex_values else np.asarray(values.real, dtype=np.float64),
        k=3,
        axis=0,
    )
    if complex_values:
        return make_native_tabulated_bspline_bank(
            spline.t,
            spline.c,
        )
    return make_native_tabulated_real_bspline_bank(
        spline.t,
        spline.c,
    )


def _shift_tabulated_cubic_source(source, time_shift: float):
    """Return a shifted view of a compact native cubic B-spline."""
    return shift_native_tabulated_bspline(source, time_shift)


def _time_step_from_simulation(sim: mp.Simulation) -> Optional[float]:
    fields = getattr(sim, "fields", None)
    dt = getattr(fields, "dt", None)
    if dt is not None:
        return float(dt)

    init_sim = getattr(sim, "init_sim", None)
    if callable(init_sim):
        init_sim()
        fields = getattr(sim, "fields", None)
        dt = getattr(fields, "dt", None)
        if dt is not None:
            return float(dt)

    return None


def _simulation_time_step(
    sim: mp.Simulation,
    explicit_dt: Optional[float] = None,
) -> Optional[float]:
    """Return Meep's time step and validate a legacy explicit value."""
    sim_dt = _time_step_from_simulation(sim)
    if sim_dt is None:
        return None if explicit_dt is None else float(explicit_dt)
    if explicit_dt is not None and not np.isclose(
        sim_dt,
        float(explicit_dt),
        rtol=1.0e-12,
        atol=0.0,
    ):
        raise ValueError(
            f"explicit dt={float(explicit_dt):.17g} does not match "
            f"Meep fields.dt={sim_dt:.17g}"
        )
    return sim_dt


def _sample_time_from_simulation(sim: mp.Simulation, sample_index: int, sample_dt: float) -> float:
    meep_time = getattr(sim, "meep_time", None)
    if callable(meep_time):
        return float(meep_time())
    return float(sample_index) * sample_dt


def _epsilon_from_medium(medium: mp.Medium) -> float:
    epsilon_diag = getattr(medium, "epsilon_diag", None)
    if epsilon_diag is None:
        raise ValueError("material_factor is required for materials without epsilon_diag")
    return float(epsilon_diag.x)


_CARTESIAN_ELECTRIC_COMPONENTS = (mp.Ex, mp.Ey, mp.Ez)
_CARTESIAN_MAGNETIC_COMPONENTS = (mp.Hx, mp.Hy, mp.Hz)
_CYLINDRICAL_ELECTRIC_COMPONENTS = (mp.Er, mp.Ep, mp.Ez)
_CYLINDRICAL_MAGNETIC_COMPONENTS = (mp.Hr, mp.Hp, mp.Hz)


def _is_magnetic_component(component: int) -> bool:
    """Return whether `component` is a supported magnetic-field component."""
    return component in (
        _CARTESIAN_MAGNETIC_COMPONENTS + _CYLINDRICAL_MAGNETIC_COMPONENTS
    )


def _is_cylindrical_simulation(simulation) -> bool:
    """Return whether a Meep simulation or SimulationSpec is cylindrical."""
    return bool(
        getattr(simulation, "is_cylindrical", False)
        or getattr(simulation, "dimensions", None) == mp.CYLINDRICAL
    )


def _is_cartesian_2d_simulation(simulation) -> bool:
    """Return whether a simulation has a zero-width Cartesian z axis."""
    if _is_cylindrical_simulation(simulation):
        return False
    if getattr(simulation, "dimensions", None) == 2:
        return True
    cell_size = getattr(simulation, "cell_size", None)
    return bool(
        cell_size is not None
        and hasattr(cell_size, "z")
        and float(cell_size.z) == 0.0
    )


def _validate_adjoint_source_amplitudes(simulation, amplitudes) -> None:
    """Reject complex source amplitudes that real adjoint fields cannot use."""
    fields = getattr(simulation, "fields", None)
    if fields is None:
        init_sim = getattr(simulation, "init_sim", None)
        if callable(init_sim):
            init_sim()
            fields = getattr(simulation, "fields", None)
    locally_incompatible = bool(
        fields is not None
        and hasattr(fields, "is_real")
        and bool(fields.is_real)
        and any(complex(amplitude).imag != 0.0 for amplitude in amplitudes)
    )
    incompatible_ranks = int(locally_incompatible)
    if mp.count_processors() > 1:
        incompatible_ranks = int(mp.sum_to_all(incompatible_ranks))
    if incompatible_ranks:
        raise ValueError(
            "gradient evaluation with real Meep fields requires "
            "adjoint_source_amplitude to be real; use a real amplitude or "
            "initialize complex Meep fields"
        )


def _validate_logical_adjoint_source_amplitudes(amplitudes) -> None:
    """Require source scaling that preserves the objective's derivative."""
    locally_invalid = any(complex(amplitude) != 1.0 for amplitude in amplitudes)
    invalid_ranks = int(locally_invalid)
    if mp.count_processors() > 1:
        invalid_ranks = int(mp.sum_to_all(invalid_ranks))
    if invalid_ranks:
        raise ValueError(
            "gradient evaluation requires adjoint_source_amplitude=1; "
            "finite-source density normalization is applied internally"
        )


def _validate_indexed_monitor_support(
    simulation,
    components,
    offsets,
) -> None:
    """Reject unsupported points whose field component is allocated.

    A globally unallocated component is a valid decoupled zero-field channel,
    so it is not treated as an invalid spatial location.
    """
    components = tuple(components)
    offsets = np.asarray(offsets, dtype=np.intp)
    if offsets.shape != (len(components) + 1,):
        raise RuntimeError("indexed monitor offsets do not match monitor points")
    local_support = (np.diff(offsets) > 0).astype(np.float64)
    fields = getattr(simulation, "fields", None)
    have_component = getattr(fields, "have_component", None)
    if callable(have_component):
        local_allocated = np.asarray(
            [bool(have_component(component)) for component in components],
            dtype=np.float64,
        )
    else:
        # Conservative for simulation doubles and unsupported Meep wrappers.
        local_allocated = np.ones(len(components), dtype=np.float64)
    support_state = np.vstack((local_support, local_allocated))
    if mp.count_processors() > 1:
        support_state = FastFieldGrid.reduce_real_inplace(support_state)
    unsupported = np.flatnonzero(
        (support_state[0] == 0.0)
        & (support_state[1] > 0.0)
    )
    if unsupported.size:
        indices = ", ".join(str(int(index)) for index in unsupported)
        raise ValueError(
            "monitor positions have no interpolation support at indices: "
            f"{indices}"
        )


def _validate_cylindrical_target_position(position) -> None:
    """Require an `(r, 0, z)` point in the cylindrical half-plane."""
    tolerance = 1.0e-12
    if float(position.x) < -tolerance:
        raise ValueError("cylindrical target positions require nonnegative radius")
    if abs(float(position.y)) > tolerance:
        raise ValueError("cylindrical target positions must lie in the r-z plane")


def _validate_cylindrical_axis_target(
    component: int,
    m: int,
    position: mp.Vector3,
) -> None:
    """Reject modal components constrained to zero on the cylindrical axis."""
    if abs(float(position.x)) > 1e-12:
        return
    if m == 0:
        valid_components = (mp.Ez, mp.Hz)
    elif abs(m) == 1:
        valid_components = (mp.Er, mp.Ep, mp.Hr, mp.Hp)
    else:
        return
    if component not in valid_components:
        raise ValueError(
            f"cylindrical component {component} vanishes at r=0 for m={m}"
        )


def _electric_components(
    component: int,
    dimensions: int,
    *,
    cylindrical: bool = False,
    m: int = 0,
) -> tuple[int, ...]:
    """Return electric components required by an isotropic design gradient."""
    if cylindrical:
        valid_components = (
            _CYLINDRICAL_ELECTRIC_COMPONENTS
            + _CYLINDRICAL_MAGNETIC_COMPONENTS
        )
        if component not in valid_components:
            raise ValueError(
                "exact cylindrical design gradients require an "
                "Er/Ep/Ez/Hr/Hp/Hz target"
            )
        if dimensions != 2:
            raise ValueError(
                "exact cylindrical design gradients require a 2D r-z design"
            )
        if m == 0:
            if component in (mp.Ep, mp.Hr, mp.Hz):
                return (mp.Ep,)
            return (mp.Er, mp.Ez)
        return _CYLINDRICAL_ELECTRIC_COMPONENTS

    valid_components = (
        _CARTESIAN_ELECTRIC_COMPONENTS + _CARTESIAN_MAGNETIC_COMPONENTS
    )
    if component not in valid_components:
        raise ValueError(
            "exact design gradients require an Ex/Ey/Ez/Hx/Hy/Hz target"
        )
    if dimensions == 3:
        return (mp.Ex, mp.Ey, mp.Ez)
    if dimensions != 2:
        raise ValueError("exact design gradients require a 2D or 3D Cartesian design")
    if component in (mp.Ez, mp.Hx, mp.Hy):
        return (mp.Ez,)
    if component in (mp.Hz, mp.Ex, mp.Ey):
        return (mp.Ex, mp.Ey)
    raise AssertionError("unreachable")


def _adjoint_point_sources(
    time_source,
    component: int,
    center: mp.Vector3,
    size: mp.Vector3,
    amplitude: complex,
    indexed_stencil=None,
) -> list:
    """Create one adjoint source or an exact indexed point transpose."""
    if indexed_stencil is None:
        return [
            mp.Source(
                time_source,
                component=component,
                center=center,
                size=size,
                amplitude=amplitude,
            )
        ]

    if not np.allclose(
        (size.x, size.y, size.z),
        (0.0, 0.0, 0.0),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "exact indexed point adjoints require a zero-size source"
        )
    source_data, weights = indexed_stencil
    weights = np.asarray(weights, dtype=np.complex128)
    if (
        weights.ndim != 1
        or len(source_data) != weights.size
    ):
        raise ValueError(
            "indexed point-adjoint stencil must contain one "
            "source-data object per weight"
        )
    grouped_indices = {}
    for index, source_datum in enumerate(source_data):
        try:
            key = (
                int(source_datum.near_fd_comp),
                int(source_datum.fc_idx),
            )
        except (AttributeError, TypeError, ValueError):
            key = ("unbatchable", index)
        grouped_indices.setdefault(key, []).append(index)

    sources = []
    for indices in grouped_indices.values():
        grouped_source_data = [source_data[index] for index in indices]
        source_datum = (
            grouped_source_data[0]
            if len(grouped_source_data) == 1
            else merge_indexed_sourcedata(grouped_source_data)
        )
        sources.append(
            mp.IndexedSource(
                time_source,
                source_datum,
                np.ascontiguousarray(
                    amplitude * weights[indices],
                    dtype=np.complex128,
                ),
            )
        )
    return sources


def _adjoint_indexed_profile_sources(
    time_source,
    point_amplitudes: np.ndarray,
    indexed_stencil,
) -> list:
    """Create batched exact sources for one separable spatial profile."""
    amplitudes = np.asarray(point_amplitudes, dtype=np.complex128)
    offsets, source_data, interpolation_weights = indexed_stencil
    offsets = np.asarray(offsets, dtype=np.intp)
    interpolation_weights = np.asarray(
        interpolation_weights,
        dtype=np.complex128,
    )
    if (
        amplitudes.ndim != 1
        or offsets.shape != (amplitudes.size + 1,)
        or offsets[0] != 0
        or offsets[-1] != interpolation_weights.size
        or len(source_data) != interpolation_weights.size
    ):
        raise ValueError(
            "indexed profile stencil must contain one CSR row per amplitude"
        )

    grouped_entries = {}
    for point_index, point_amplitude in enumerate(amplitudes):
        for entry_index in range(
            int(offsets[point_index]),
            int(offsets[point_index + 1]),
        ):
            source_datum = source_data[entry_index]
            try:
                key = (
                    int(source_datum.near_fd_comp),
                    int(source_datum.fc_idx),
                )
            except (AttributeError, TypeError, ValueError):
                key = ("unbatchable", entry_index)
            grouped_entries.setdefault(key, []).append(
                (
                    source_datum,
                    point_amplitude * interpolation_weights[entry_index],
                )
            )

    sources = []
    for entries in grouped_entries.values():
        grouped_source_data = [entry[0] for entry in entries]
        source_datum = (
            grouped_source_data[0]
            if len(grouped_source_data) == 1
            else merge_indexed_sourcedata(grouped_source_data)
        )
        sources.append(
            mp.IndexedSource(
                time_source,
                source_datum,
                np.ascontiguousarray(
                    [entry[1] for entry in entries],
                    dtype=np.complex128,
                ),
            )
        )
    return sources


def _is_zero_source_size(size: mp.Vector3) -> bool:
    return bool(
        np.array_equal(
            np.asarray((size.x, size.y, size.z), dtype=float),
            np.zeros(3),
        )
    )


class _PointTarget:
    """Point-monitor target adapter for TDAObjective.

    This is the time-domain counterpart of Meep adjoint's objective-quantity
    objects: it knows how to record the forward monitor signal, place the
    corresponding adjoint point source, and supply its source settings. The
    scalar FoM itself is supplied to TDAObjective.
    """

    def __init__(
        self,
        monitor_position: mp.Vector3,
        component: int,
        adjoint_source_size: Optional[mp.Vector3] = None,
        adjoint_source_amplitude: complex = 1.0,
    ) -> None:
        """Define a point monitor and its matching time-domain adjoint source.

        Args:
            monitor_position: Physical point at which the forward field is
                sampled and the adjoint source is placed.
            component: Meep field component, e.g. `mp.Ez`.
            adjoint_source_size: Meep source size for the adjoint injection.
                The default zero size enables exact indexed injection when the
                objective uses direct `SimulationSpec.make`. Explicit
                nonzero Cartesian sizes use ordinary `mp.Source` injection.
            adjoint_source_amplitude: Logical source amplitude. Gradient
                evaluation requires `1.0`; effective finite-source density
                normalization is applied separately.
        """
        self.monitor_position = monitor_position
        self.component = component
        self.adjoint_source_size = adjoint_source_size if adjoint_source_size is not None else mp.Vector3()
        self.adjoint_source_amplitude = adjoint_source_amplitude

    @property
    def positions(self) -> tuple[mp.Vector3, ...]:
        return (self.monitor_position,)

    @property
    def sample_shape(self) -> tuple:
        return ()

    @property
    def spatial_weights(self) -> NDArray[np.float64]:
        return np.ones(1, dtype=float)

    def record_monitor(self, sim: mp.Simulation) -> complex:
        """Sample the monitored field component during the forward run.

        Args:
            sim: Active forward Meep simulation.

        Returns:
            Complex field value at `monitor_position`.
        """
        return sim.get_field_point(self.component, self.monitor_position)

    def adjoint_sources(
        self,
        adjoint_signal: NDArray[np.complex128],
        sample_times: NDArray[np.float64],
        actual_time: float,
        *,
        source_size: Optional[mp.Vector3] = None,
        source_amplitude: Optional[complex] = None,
        indexed_stencil=None,
    ) -> list:
        """Create the time-reversed source from a sampled bilinear covector.

        Args:
            adjoint_signal: Continuous-time adjoint signal sampled on the
                forward monitor time grid.
            sample_times: Forward monitor sample times.
            actual_time: Final time reached by the forward simulation.
            source_size: Optional effective source size for this run.
            source_amplitude: Optional effective source amplitude for this run.
            indexed_stencil: Optional exact rank-local monitor-transpose source
                data and weights.

        Returns:
            List of Meep sources to pass back into the simulation factory.
        """
        source_parity = -1.0 if _is_magnetic_component(self.component) else 1.0
        adj_sig = source_parity * adjoint_signal[::-1].copy()
        sample_times = np.asarray(sample_times, dtype=float)
        if sample_times.shape[0] != adj_sig.shape[0]:
            raise ValueError("sample_times must match adjoint_signal length")
        t_adj = float(actual_time) - sample_times[::-1]
        adjoint_source = _tabulated_cubic_source(t_adj, adj_sig)
        if source_size is None:
            source_size = self.adjoint_source_size
        if source_amplitude is None:
            source_amplitude = self.adjoint_source_amplitude
        return _adjoint_point_sources(
            mp.CustomSource(src_func=adjoint_source),
            self.component,
            self.monitor_position,
            source_size,
            source_amplitude,
            indexed_stencil,
        )


class _FieldRegionTarget:
    """Exact multi-point field target adapter for TDAObjective."""

    def __init__(
        self,
        positions,
        component: int,
        sample_shape,
        spatial_weights,
        adjoint_source_amplitude: complex = 1.0,
    ) -> None:
        self.positions = tuple(positions)
        self.component = component
        self.sample_shape = tuple(sample_shape)
        self.spatial_weights = np.asarray(spatial_weights, dtype=float).reshape(
            self.sample_shape
        )
        self.adjoint_source_amplitude = complex(adjoint_source_amplitude)

    def reshape_history(
        self,
        flat_history: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        flat_history = np.asarray(flat_history)
        return flat_history.reshape(
            (flat_history.shape[0],) + self.sample_shape
        )

    def adjoint_sources(
        self,
        adjoint_signal: NDArray[np.complex128],
        sample_times: NDArray[np.float64],
        actual_time: float,
        *,
        source_amplitudes,
        indexed_stencils,
    ) -> list:
        """Create exact point-specific sources from a regional covector."""
        flat_signal = np.asarray(adjoint_signal).reshape(
            (np.asarray(adjoint_signal).shape[0], len(self.positions))
        )
        sample_times = np.asarray(sample_times, dtype=float)
        if sample_times.shape[0] != flat_signal.shape[0]:
            raise ValueError("sample_times must match adjoint_signal length")
        if len(source_amplitudes) != len(self.positions):
            raise ValueError("source_amplitudes must match regional positions")
        if (
            indexed_stencils is None
            or len(indexed_stencils) != len(self.positions)
            or any(stencil is None for stencil in indexed_stencils)
        ):
            raise ValueError(
                "regional adjoint sources require one exact indexed stencil "
                "per position"
            )

        source_parity = -1.0 if _is_magnetic_component(self.component) else 1.0
        flat_signal = source_parity * flat_signal[::-1].copy()
        t_adj = float(actual_time) - sample_times[::-1]
        active_indices = [
            point_index
            for point_index, indexed_stencil in enumerate(indexed_stencils)
            if len(indexed_stencil[0]) != 0
        ]
        time_sources = _tabulated_cubic_sources(
            t_adj,
            flat_signal[:, active_indices],
        )
        sources = []
        for point_index, time_source in zip(active_indices, time_sources):
            position = self.positions[point_index]
            indexed_stencil = indexed_stencils[point_index]
            sources.extend(
                _adjoint_point_sources(
                    mp.CustomSource(src_func=time_source),
                    self.component,
                    position,
                    mp.Vector3(),
                    source_amplitudes[point_index],
                    indexed_stencil,
                )
            )
        return sources
