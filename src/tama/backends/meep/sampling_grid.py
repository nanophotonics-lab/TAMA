"""Native field sampling helpers for Meep time-domain adjoint loops."""

import sys
import weakref

import meep as mp
import numpy as np
from numpy.typing import NDArray
from typing import Any, Mapping, Optional, Sequence

from . import native_sampler

_NATIVE_API_VERSION = 12
_REQUIRED_NATIVE_OPERATIONS = (
    "create_component_grid_plan",
    "sample_component_grid_plan_allreduced",
    "create_component_point_plan",
    "sample_component_point_plan_allreduced",
    "component_point_plan_indexed_stencil",
    "configure_component_point_plan_history",
    "populate_sourcedata",
    "merge_sourcedata",
    "sample_component_point_plan_local_into",
    "sample_component_point_plan_local_real_into",
    "create_eigenmode_overlap_plan",
    "sample_eigenmode_overlap_plan_local_into",
    "component_grid_plan_local_complete_mask",
    "component_grid_plan_local_boundary_mask",
    "configure_component_grid_plan_history",
    "sample_component_grid_plan_history_into",
    "configure_component_grid_plan_accumulation",
    "sample_component_grid_plan_points_local",
    "sample_component_grid_plan_points_support_reduced",
    "accumulate_component_product_allreduced",
    "accumulate_component_product_plan_local_inplace",
    "accumulate_component_product_plan_points_local_inplace",
    "accumulate_component_product_plan_configured_local_inplace",
    "accumulate_component_difference_product_plan_configured_local_inplace",
    "reduce_complex_grid_sum",
    "reduce_complex_grid_sum_inplace",
    "reduce_real_grid_sum_inplace",
    "create_tabulated_cubic",
    "create_tabulated_real_cubic_bank",
    "create_tabulated_bspline_bank",
    "create_tabulated_real_bspline_bank",
    "shift_tabulated_bspline",
    "create_native_design_plan",
    "native_design_plan_local_size",
    "native_design_plan_signature",
    "sample_native_design_plan_into",
    "sample_native_design_plan_real_into",
    "accumulate_native_design_product_local_inplace",
    "accumulate_native_design_real_product_local_inplace",
    "accumulate_native_design_midpoint_product_local_inplace",
    "accumulate_native_design_real_midpoint_product_local_inplace",
    "run_native_design_adjoint_segment",
    "native_forward_step_count",
    "run_native_forward_segment",
)

_missing_native_operations = [
    name
    for name in _REQUIRED_NATIVE_OPERATIONS
    if not callable(getattr(native_sampler, name, None))
]
_native_sampler_compatible = (
    getattr(native_sampler, "API_VERSION", None) == _NATIVE_API_VERSION
    and not _missing_native_operations
)
_native_sampler_state_synchronized = False


def _require_native_sampler() -> None:
    """Fail consistently if any active Meep rank has an incompatible extension."""
    global _native_sampler_compatible
    global _native_sampler_state_synchronized
    if _native_sampler_state_synchronized:
        if not _native_sampler_compatible:
            raise RuntimeError(
                "TAMA native sampler is incompatible with this package version"
            )
        return

    incompatible_ranks = int(not _native_sampler_compatible)
    if mp.count_processors() > 1:
        try:
            incompatible_ranks = int(mp.sum_to_all(incompatible_ranks))
        except Exception as exc:
            _handle_native_sampler_failure(
                "capability synchronization",
                exc,
            )
    _native_sampler_state_synchronized = True
    if incompatible_ranks:
        _native_sampler_compatible = False
        details = (
            ""
            if not _missing_native_operations
            else ". Missing: " + ", ".join(_missing_native_operations)
        )
        raise RuntimeError(
            "TAMA native sampler is incompatible on one or more active "
            "Meep ranks; reinstall TAMA from the active Meep/MPI "
            f"environment{details}"
        )


def _handle_native_sampler_failure(operation: str, exc: Exception) -> None:
    """Raise in serial or abort an MPI run after a native operation fails."""
    message = (
        f"TAMA native sampler {operation} failed on rank {mp.my_rank()}: {exc}"
    )
    if mp.count_processors() > 1:
        print(message, file=sys.stderr, flush=True)
        abort = getattr(mp, "abort", None)
        if callable(abort):
            abort("TAMA native sampler failure")
    raise RuntimeError(message) from exc


def native_sampler_available() -> bool:
    """Return whether this rank has the required native sampler API."""
    return _native_sampler_compatible


def history_storage_dtype(sim: mp.Simulation, requested_dtype) -> np.dtype:
    """Resolve a history dtype from Meep's initialized field representation."""
    dtype = np.dtype(requested_dtype)
    if not (
        np.issubdtype(dtype, np.floating)
        or np.issubdtype(dtype, np.complexfloating)
    ):
        raise ValueError("history_dtype must be a real or complex floating dtype")

    fields = getattr(sim, "fields", None)
    if fields is None:
        init_sim = getattr(sim, "init_sim", None)
        if callable(init_sim):
            init_sim()
            fields = getattr(sim, "fields", None)
    if fields is None or not hasattr(fields, "is_real"):
        if not np.issubdtype(dtype, np.complexfloating):
            raise ValueError(
                "a real history_dtype requires initialized Meep fields"
            )
        return dtype

    fields_are_real = bool(fields.is_real)
    if mp.count_processors() > 1:
        real_ranks = int(mp.sum_to_all(int(fields_are_real)))
        if real_ranks not in (0, mp.count_processors()):
            raise RuntimeError(
                "Meep field representation differs across active MPI ranks"
            )
        fields_are_real = real_ranks == mp.count_processors()

    if fields_are_real:
        if np.issubdtype(dtype, np.complexfloating):
            return np.empty((), dtype=dtype).real.dtype
        return dtype
    if not np.issubdtype(dtype, np.complexfloating):
        raise ValueError(
            "complex Meep fields require a complex history_dtype"
        )
    return dtype


def make_native_tabulated_cubic(
    breaks: NDArray[np.float64],
    coefficients: NDArray[np.complex128],
):
    """Create a native complex cubic callable."""
    _require_native_sampler()
    try:
        return native_sampler.create_tabulated_cubic(
            np.asarray(breaks, dtype=np.float64),
            np.asarray(coefficients, dtype=np.complex128),
        )
    except Exception as exc:
        _handle_native_sampler_failure("tabulated cubic construction", exc)
        raise


def make_native_tabulated_real_cubic_bank(
    breaks: NDArray[np.float64],
    coefficients: NDArray[np.float64],
):
    """Create real cubic callables sharing one breakpoint/coefficient bank."""
    _require_native_sampler()
    try:
        return tuple(
            native_sampler.create_tabulated_real_cubic_bank(
                np.asarray(breaks, dtype=np.float64),
                np.asarray(coefficients, dtype=np.float64),
            )
        )
    except Exception as exc:
        _handle_native_sampler_failure("real cubic-bank construction", exc)
        raise


def make_native_tabulated_bspline_bank(
    knots: NDArray[np.float64],
    coefficients: NDArray[np.complex128],
):
    """Create complex B-spline callables sharing knots and coefficients."""
    _require_native_sampler()
    try:
        return tuple(
            native_sampler.create_tabulated_bspline_bank(
                np.asarray(knots, dtype=np.float64),
                np.asarray(coefficients, dtype=np.complex128),
            )
        )
    except Exception as exc:
        _handle_native_sampler_failure("tabulated B-spline construction", exc)
        raise


def make_native_tabulated_real_bspline_bank(
    knots: NDArray[np.float64],
    coefficients: NDArray[np.float64],
):
    """Create real B-spline callables sharing knots and coefficients."""
    _require_native_sampler()
    try:
        return tuple(
            native_sampler.create_tabulated_real_bspline_bank(
                np.asarray(knots, dtype=np.float64),
                np.asarray(coefficients, dtype=np.float64),
            )
        )
    except Exception as exc:
        _handle_native_sampler_failure(
            "real tabulated B-spline construction",
            exc,
        )
        raise


def shift_native_tabulated_bspline(source, time_shift: float):
    """Return a native view evaluating `source(time - time_shift)`."""
    _require_native_sampler()
    try:
        return native_sampler.shift_tabulated_bspline(
            source,
            float(time_shift),
        )
    except Exception as exc:
        _handle_native_sampler_failure("tabulated B-spline shift", exc)
        raise


def merge_indexed_sourcedata(source_data: Sequence[Any]):
    """Pack singleton indices sharing one component and chunk."""
    _require_native_sampler()
    merged = mp.sourcedata()
    try:
        native_sampler.merge_sourcedata(
            int(merged.this),
            tuple(int(data.this) for data in source_data),
        )
    except Exception as exc:
        _handle_native_sampler_failure("indexed-source batching", exc)
        raise
    return merged


def _simulation_has_symmetry(sim: mp.Simulation) -> bool:
    return bool(getattr(sim, "symmetries", []))


def _require_no_meep_symmetries(sim: mp.Simulation) -> None:
    """Reject Meep symmetry reductions unsupported by exact native sampling."""
    if _simulation_has_symmetry(sim):
        raise ValueError("TAMA does not support Meep symmetries")


class _NativePlanFieldsGuard:
    """Reject cached native plans after Meep replaces their fields object."""

    def _bind_native_plan_fields(self, sim: mp.Simulation) -> None:
        self._native_plan_sim_ref = weakref.ref(sim)
        self._native_plan_fields_ref = weakref.ref(sim.fields)

    def _require_native_plan_fields(self) -> Any:
        sim = self._native_plan_sim_ref()
        fields = self._native_plan_fields_ref()
        if (
            sim is None
            or fields is None
            or getattr(sim, "fields", None) is not fields
        ):
            _handle_native_sampler_failure(
                "plan lifetime validation",
                RuntimeError(
                    "cached native plan is stale because Meep fields were replaced; "
                    "create a new TAMA monitor or design-field wrapper"
                ),
            )
        return fields


class FastPointMonitor(_NativePlanFieldsGuard):
    """Sample fixed Cartesian or cylindrical points with native reduction."""

    def __init__(
        self,
        sim: mp.Simulation,
        component: int,
        positions: Sequence[mp.Vector3],
    ) -> None:
        self.sim = sim
        self.component = component
        self.positions = list(positions)
        self.plan = None
        self._history_indices = None
        self.is_cylindrical = bool(
            getattr(sim, "is_cylindrical", False)
            or getattr(sim, "dimensions", None) == mp.CYLINDRICAL
        )
        _require_native_sampler()
        _require_no_meep_symmetries(sim)
        if getattr(sim, "dimensions", 2) not in (2, 3, mp.CYLINDRICAL):
            raise ValueError(
                "native point monitoring requires a 2D, 3D, or cylindrical simulation"
            )
        if not hasattr(getattr(sim, "fields", None), "this"):
            raise RuntimeError(
                "native point monitoring requires initialized Meep fields"
            )
        self._bind_native_plan_fields(sim)
        self.fields_are_real = bool(sim.fields.is_real)

    def ensure_plan(self) -> Any:
        """Create and return the reusable native point-monitor plan."""
        fields = self._require_native_plan_fields()
        if self.plan is None:
            try:
                coords_x = [position.x for position in self.positions]
                coords_y = [
                    position.z if self.is_cylindrical else position.y
                    for position in self.positions
                ]
                if getattr(self.sim, "dimensions", 2) == 3:
                    self.plan = native_sampler.create_component_point_plan(
                        int(fields.this),
                        coords_x,
                        coords_y,
                        [position.z for position in self.positions],
                        int(self.component),
                    )
                else:
                    self.plan = native_sampler.create_component_point_plan(
                        int(fields.this),
                        coords_x,
                        coords_y,
                        int(self.component),
                    )
            except ValueError:
                raise
            except Exception as exc:
                _handle_native_sampler_failure("packed monitor plan creation", exc)
        return self.plan

    def indexed_transpose_stencil(
        self,
    ) -> tuple[
        NDArray[np.intp],
        tuple[Any, ...],
        NDArray[np.complex128],
    ]:
        """Return exact rank-local adjoint-source metadata.

        The tuple contains monitor CSR offsets, one populated Meep
        `sourcedata` object per local Yee support point, and amplitudes that
        can be passed directly to `mp.IndexedSource`.
        """
        plan = self.ensure_plan()

        try:
            offsets, components, chunk_indices, local_indices, amplitudes = (
                native_sampler.component_point_plan_indexed_stencil(plan)
            )
        except Exception as exc:
            _handle_native_sampler_failure(
                "point-monitor transpose planning",
                exc,
            )
            raise
        offsets = np.asarray(offsets, dtype=np.intp)
        components = np.asarray(components, dtype=np.int64)
        chunk_indices = np.asarray(chunk_indices, dtype=np.int64)
        local_indices = np.asarray(local_indices, dtype=np.intp)
        amplitudes = np.asarray(amplitudes, dtype=np.complex128)
        if (
            offsets.shape != (len(self.positions) + 1,)
            or components.ndim != 1
            or chunk_indices.ndim != 1
            or components.shape != chunk_indices.shape
            or local_indices.shape != chunk_indices.shape
            or amplitudes.shape != chunk_indices.shape
            or offsets[0] != 0
            or offsets[-1] != amplitudes.size
        ):
            _handle_native_sampler_failure(
                "point-monitor transpose validation",
                RuntimeError(
                    "native indexed point-monitor stencil is malformed"
                ),
            )

        source_data = []
        for component, chunk_index, local_index in zip(
            components,
            chunk_indices,
            local_indices,
        ):
            try:
                data = mp.sourcedata()
                native_sampler.populate_sourcedata(
                    int(data.this),
                    int(component),
                    int(chunk_index),
                    int(local_index),
                )
            except Exception as exc:
                _handle_native_sampler_failure(
                    "indexed-source population",
                    exc,
                )
                raise
            source_data.append(data)
        return offsets, tuple(source_data), amplitudes

    def sample(self) -> NDArray[np.complex128]:
        """Return all monitor values in the original position order."""
        plan = self.ensure_plan()
        try:
            return np.asarray(
                native_sampler.sample_component_point_plan_allreduced(plan),
                dtype=np.complex128,
            )
        except Exception as exc:
            _handle_native_sampler_failure("packed monitor sampling", exc)
            raise

    def sample_history_into(self, destination: np.ndarray) -> None:
        """Write one history row, deferring MPI reduction when supported."""
        destination_array = np.asarray(destination)
        history_width = (
            len(self.positions)
            if self._history_indices is None
            else self._history_indices.size
        )
        if destination_array.ndim != 1 or destination_array.size != history_width:
            raise ValueError("point-monitor history destination must match the point count")
        if not destination_array.flags.writeable:
            raise ValueError(
                "local point-monitor history destination must be writable"
            )

        native_destination = destination_array
        native_sampler_fn = native_sampler.sample_component_point_plan_local_into
        if self.fields_are_real and np.issubdtype(
            destination_array.dtype,
            np.floating,
        ):
            native_sampler_fn = (
                native_sampler.sample_component_point_plan_local_real_into
            )
            if (
                destination_array.dtype != np.float64
                or not destination_array.flags.c_contiguous
            ):
                native_destination = np.empty(
                    destination_array.size,
                    dtype=np.float64,
                )
        elif not np.issubdtype(destination_array.dtype, np.complexfloating):
            raise ValueError(
                "complex Meep fields require a complex monitor-history destination"
            )
        elif (
            destination_array.dtype != np.complex128
            or not destination_array.flags.c_contiguous
        ):
            native_destination = np.empty(
                destination_array.size,
                dtype=np.complex128,
            )
        try:
            native_sampler_fn(
                self.ensure_plan(),
                native_destination,
            )
        except Exception as exc:
            _handle_native_sampler_failure("local monitor sampling", exc)
            raise
        if native_destination is not destination_array:
            destination_array[:] = native_destination

    def configure_history_indices(self, indices) -> np.ndarray:
        """Restrict subsequent local history sampling to selected points."""
        if self._history_indices is not None:
            raise RuntimeError("point-monitor history indices are already configured")
        history_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if np.any(history_indices < 0) or np.any(
            history_indices >= len(self.positions)
        ):
            raise IndexError("point-monitor history index out of range")
        if np.unique(history_indices).size != history_indices.size:
            raise ValueError(
                "point-monitor history indices must not contain duplicates"
            )
        try:
            native_sampler.configure_component_point_plan_history(
                self.ensure_plan(),
                history_indices,
            )
        except Exception as exc:
            _handle_native_sampler_failure(
                "point-monitor history configuration",
                exc,
            )
            raise
        self._history_indices = history_indices
        return np.array(history_indices, copy=True)

    def reduce_history(
        self,
        local_history: np.ndarray,
    ) -> np.ndarray:
        """Combine and consume rank-local monitor history once."""
        if np.issubdtype(np.asarray(local_history).dtype, np.floating):
            return FastFieldGrid.reduce_real_inplace(local_history)
        return FastFieldGrid.reduce_inplace(local_history)


class FastEigenmodeOverlapMonitor:
    """Accumulate a fixed modal projection without storing point histories."""

    def __init__(
        self,
        sim: mp.Simulation,
        positions: Sequence[mp.Vector3],
        electric_weights: Mapping[int, np.ndarray],
        magnetic_weights: Mapping[int, np.ndarray],
    ) -> None:
        self.positions = tuple(positions)
        if not self.positions:
            raise ValueError("eigenmode overlap monitor requires positions")
        if not electric_weights or not magnetic_weights:
            raise ValueError(
                "eigenmode overlap monitor requires electric and magnetic weights"
            )
        overlap = set(electric_weights).intersection(magnetic_weights)
        if overlap:
            raise ValueError(
                "eigenmode overlap components cannot be both electric and magnetic"
            )

        component_weights = []
        output_channels = []
        for channel, weights_by_component in enumerate(
            (electric_weights, magnetic_weights)
        ):
            for component, weights in weights_by_component.items():
                values = np.ascontiguousarray(weights, dtype=np.complex128)
                if values.shape != (len(self.positions),):
                    raise ValueError(
                        "eigenmode overlap weights must match the point count"
                    )
                if not np.all(np.isfinite(values)):
                    raise ValueError("eigenmode overlap weights must be finite")
                component_weights.append((int(component), values))
                output_channels.append(channel)

        self.component_weights = dict(component_weights)
        self.component_monitors = {
            component: FastPointMonitor(sim, component, self.positions)
            for component, _ in component_weights
        }
        self._ordered_components = tuple(
            component for component, _ in component_weights
        )
        self._weights = tuple(weights for _, weights in component_weights)
        self._output_channels = np.asarray(output_channels, dtype=np.intp)
        self.plan = None

    def ensure_plan(self) -> Any:
        """Create and return the persistent native overlap plan."""
        component_plans = [
            self.component_monitors[component].ensure_plan()
            for component in self._ordered_components
        ]
        if self.plan is None:
            try:
                self.plan = native_sampler.create_eigenmode_overlap_plan(
                    component_plans,
                    self._weights,
                    self._output_channels,
                )
            except ValueError:
                raise
            except Exception as exc:
                _handle_native_sampler_failure(
                    "eigenmode overlap plan creation",
                    exc,
                )
        return self.plan

    def sample_history_into(self, destination: np.ndarray) -> None:
        """Write one rank-local `[electric, magnetic]` overlap row."""
        destination_array = np.asarray(destination)
        if destination_array.shape != (2,):
            raise ValueError(
                "eigenmode overlap history destination must have shape (2,)"
            )
        if not destination_array.flags.writeable:
            raise ValueError(
                "eigenmode overlap history destination must be writable"
            )
        if not np.issubdtype(
            destination_array.dtype,
            np.complexfloating,
        ):
            raise ValueError(
                "eigenmode overlap history destination must be complex"
            )
        native_destination = destination_array
        if (
            destination_array.dtype != np.complex128
            or not destination_array.flags.c_contiguous
        ):
            native_destination = np.empty(2, dtype=np.complex128)
        try:
            native_sampler.sample_eigenmode_overlap_plan_local_into(
                self.ensure_plan(),
                native_destination,
            )
        except Exception as exc:
            _handle_native_sampler_failure(
                "local eigenmode overlap sampling",
                exc,
            )
            raise
        if native_destination is not destination_array:
            destination_array[:] = native_destination

    def reduce_history(self, local_history: np.ndarray) -> np.ndarray:
        """Combine and consume a rank-local complex overlap history once."""
        return FastFieldGrid.reduce_inplace(local_history)

    def indexed_transpose_stencils(self):
        """Return each component monitor's exact CSR transpose stencil."""
        return {
            component: monitor.indexed_transpose_stencil()
            for component, monitor in self.component_monitors.items()
        }


class FastFieldGrid(_NativePlanFieldsGuard):
    """Sample one Meep field component on a fixed 2D point grid.

    Args:
        sim: Active Meep simulation whose fields will be sampled.
        component: Meep field component, e.g. `mp.Ez`.
        coords_x: x coordinates of the sampling grid.
        coords_y: y coordinates of the sampling grid.

    Attributes:
        shape: Output grid shape `(len(coords_x), len(coords_y))`.
    """

    def __init__(
        self,
        sim: mp.Simulation,
        component: int,
        coords_x: Sequence[float],
        coords_y: Sequence[float],
    ) -> None:
        self.sim = sim
        self.component = component
        self.coords_x = list(coords_x)
        self.coords_y = list(coords_y)
        self.shape = (len(self.coords_x), len(self.coords_y))
        _require_native_sampler()
        _require_no_meep_symmetries(sim)
        if (
            getattr(sim, "dimensions", 2) != 2
            or bool(getattr(sim, "is_cylindrical", False))
        ):
            raise ValueError(
                "FastFieldGrid requires a 2D Cartesian Meep simulation"
            )
        if not hasattr(getattr(sim, "fields", None), "this"):
            raise RuntimeError(
                "native field-grid sampling requires initialized Meep fields"
            )
        self._bind_native_plan_fields(sim)
        self.plan = None
        self._history_sampling_indices = None
        self._history_plan_configured = False

    def ensure_plan(self) -> Any:
        """Create the native sampling plan once per simulation."""
        fields = self._require_native_plan_fields()
        if self.plan is None:
            try:
                self.plan = native_sampler.create_component_grid_plan(
                    int(fields.this),
                    self.coords_x,
                    self.coords_y,
                    int(self.component),
                )
            except ValueError:
                raise
            except Exception as exc:
                _handle_native_sampler_failure("plan creation", exc)
        return self.plan

    def sample(self) -> NDArray[np.complex128]:
        """Sample the field grid.

        Returns:
            Complex-valued array with shape `(len(coords_x), len(coords_y))`.
            Values match repeated Meep field sampling at the same physical
            coordinates.
        """
        plan = self.ensure_plan()
        try:
            return native_sampler.sample_component_grid_plan_allreduced(
                plan
            )
        except Exception as exc:
            _handle_native_sampler_failure("planned all-reduced sampling", exc)
            raise

    @staticmethod
    def _flat_index_array(indices) -> NDArray[np.int64]:
        return np.asarray(indices, dtype=np.int64).reshape(-1)

    @staticmethod
    def _empty_complex_values() -> NDArray[np.complex128]:
        return np.empty(0, dtype=np.complex128)

    def history_sampling_indices(self):
        """Return local-complete and boundary flat indices for history storage.

        Local-complete points have their full interpolation support on this MPI
        rank, so both forward and adjoint fields can be multiplied locally.
        Boundary points are not fully owned by any rank; only ranks that
        contribute to their interpolation support keep those forward values and
        later contribute their local adjoint fields.

        Returns:
            `(local_indices, boundary_indices)` as int64 flat-index arrays.
        """
        if self._history_sampling_indices is not None:
            return self._history_sampling_indices

        sample_plan = self.ensure_plan()

        try:
            local_complete = np.asarray(
                native_sampler.component_grid_plan_local_complete_mask(sample_plan),
                dtype=bool,
            ).reshape(-1)
            local_boundary = np.asarray(
                native_sampler.component_grid_plan_local_boundary_mask(sample_plan),
                dtype=bool,
            ).reshape(-1)
        except Exception as exc:
            _handle_native_sampler_failure("history ownership planning", exc)
            raise

        self._history_sampling_indices = (
            np.flatnonzero(local_complete).astype(np.int64),
            np.flatnonzero(local_boundary).astype(np.int64),
        )
        try:
            native_sampler.configure_component_grid_plan_history(
                sample_plan,
                self._history_sampling_indices[0],
                self._history_sampling_indices[1],
            )
            self._history_plan_configured = True
        except Exception as exc:
            _handle_native_sampler_failure("history plan configuration", exc)
            raise
        return self._history_sampling_indices

    def sample_history_values(self, local_indices, boundary_indices) -> NDArray[np.complex128]:
        """Sample forward values for a history row using ownership metadata.

        In MPI, this method is collective over the native support
        communicators. Every active rank must call it in the same order and
        pass the per-rank index arrays returned by
        :meth:`history_sampling_indices`. Caller-selected MPI subsets or
        orderings are unsupported because supporting ranks must enter each
        reduction with identical point order.
        """
        local_indices = self._flat_index_array(local_indices)
        boundary_indices = self._flat_index_array(boundary_indices)

        sample_plan = self.ensure_plan()

        try:
            local_values = (
                native_sampler.sample_component_grid_plan_points_local(sample_plan, local_indices)
                if local_indices.size
                else self._empty_complex_values()
            )
            boundary_values = (
                native_sampler.sample_component_grid_plan_points_support_reduced(sample_plan, boundary_indices)
                if boundary_indices.size
                else self._empty_complex_values()
            )
        except Exception as exc:
            _handle_native_sampler_failure("indexed history sampling", exc)
            raise RuntimeError("native history sampling failed") from exc

        return np.concatenate([
            np.asarray(local_values, dtype=np.complex128),
            np.asarray(boundary_values, dtype=np.complex128),
        ])

    def sample_history_values_into(
        self,
        local_indices,
        boundary_indices,
        destination: NDArray[np.complex128],
    ) -> None:
        """Sample one history row directly into a caller-provided array.

        The MPI collective contract is the same as for
        :meth:`sample_history_values`: use the arrays returned by
        :meth:`history_sampling_indices` on every active rank.
        """
        local_indices = self._flat_index_array(local_indices)
        boundary_indices = self._flat_index_array(boundary_indices)
        destination_array = np.asarray(destination)
        expected_width = local_indices.size + boundary_indices.size
        if destination_array.ndim != 1 or destination_array.size != expected_width:
            raise ValueError("history destination must match local and boundary index count")

        sample_plan = self.ensure_plan()
        configured_indices_match = (
            self._history_sampling_indices is not None
            and np.array_equal(local_indices, self._history_sampling_indices[0])
            and np.array_equal(boundary_indices, self._history_sampling_indices[1])
        )
        if (
            self._history_plan_configured
            and configured_indices_match
            and destination_array.dtype == np.complex128
            and destination_array.flags.c_contiguous
        ):
            try:
                native_sampler.sample_component_grid_plan_history_into(
                    sample_plan,
                    destination_array,
                )
                return
            except Exception as exc:
                _handle_native_sampler_failure("direct history sampling", exc)
                raise

        destination_array[:] = self.sample_history_values(
            local_indices,
            boundary_indices,
        ).astype(destination_array.dtype, copy=False)

    def product(self, multiplier: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Sample the field and multiply it by a grid-shaped factor.

        Args:
            multiplier: Array broadcastable to `self.shape`.

        Returns:
            `sample() * multiplier` as a complex grid.
        """
        try:
            return native_sampler.accumulate_component_product_allreduced(
                int(self.sim.fields.this),
                self.coords_x,
                self.coords_y,
                int(self.component),
                np.asarray(multiplier, dtype=np.complex128),
            )
        except Exception as exc:
            _handle_native_sampler_failure("all-reduced field product", exc)
            raise

    def accumulate_product(
        self,
        multiplier: NDArray[np.complex128],
        accumulator: NDArray[np.complex128],
    ) -> bool:
        """Accumulate `field * multiplier` into `accumulator`.

        Args:
            multiplier: Grid-shaped factor, typically `dE/dt` from the
                forward run.
            accumulator: Complex grid updated in place.

        Returns:
            `True` after native in-place accumulation succeeds.
        """
        try:
            native_sampler.accumulate_component_product_plan_local_inplace(
                self.ensure_plan(),
                np.asarray(multiplier, dtype=np.complex128),
                accumulator,
            )
            return True
        except Exception as exc:
            _handle_native_sampler_failure(
                "planned local gradient accumulation",
                exc,
            )
            raise

    def accumulate_indexed_product(
        self,
        indices: NDArray[np.int64],
        values: NDArray[np.complex128],
        accumulator: NDArray[np.complex128],
    ) -> bool:
        """Accumulate selected `field * value` entries into `accumulator`.

        Args:
            indices: Flat grid indices whose multiplier values are nonzero.
            values: Complex multiplier values matching `indices`.
            accumulator: Complex grid updated in place.

        Returns:
            `True` after native indexed accumulation succeeds.
        """
        try:
            native_sampler.accumulate_component_product_plan_points_local_inplace(
                self.ensure_plan(),
                self._flat_index_array(indices),
                np.asarray(values, dtype=np.complex128).reshape(-1),
                accumulator,
            )
            return True
        except Exception as exc:
            _handle_native_sampler_failure(
                "indexed local gradient accumulation",
                exc,
            )
            raise

    def configure_accumulation_indices(self, indices: NDArray[np.int64]) -> bool:
        """Cache fixed flat indices in the native sample plan."""
        try:
            native_sampler.configure_component_grid_plan_accumulation(
                self.ensure_plan(),
                self._flat_index_array(indices),
            )
        except Exception as exc:
            _handle_native_sampler_failure(
                "fixed-index accumulation configuration",
                exc,
            )
            raise
        return True

    def accumulate_configured_product(
        self,
        values: NDArray[np.complex128],
        accumulator: NDArray[np.complex128],
    ) -> bool:
        """Accumulate values using indices cached by the native sample plan."""
        try:
            native_sampler.accumulate_component_product_plan_configured_local_inplace(
                self.ensure_plan(),
                np.asarray(values, dtype=np.complex128).reshape(-1),
                accumulator,
            )
            return True
        except Exception as exc:
            _handle_native_sampler_failure(
                "configured local gradient accumulation",
                exc,
            )
            raise

    def accumulate_configured_difference_product(
        self,
        values: NDArray[np.complex128],
        dt: float,
        accumulator: NDArray[np.complex128],
    ) -> bool:
        """Accumulate values times the current-minus-previous field derivative.

        Returns `False` when the native plan only initialized its previous
        field cache and `True` after an accumulation.
        """
        try:
            return bool(
                native_sampler.accumulate_component_difference_product_plan_configured_local_inplace(
                    self.ensure_plan(),
                    np.asarray(values, dtype=np.complex128).reshape(-1),
                    float(dt),
                    accumulator,
                )
            )
        except Exception as exc:
            _handle_native_sampler_failure(
                "configured local adjoint-difference accumulation",
                exc,
            )
            raise

    @staticmethod
    def supports_local_accumulation() -> bool:
        """Return `True` for the required native accumulation path."""
        _require_native_sampler()
        return True

    @staticmethod
    def supports_reduction() -> bool:
        """Return `True` for the required native process-group reducer."""
        _require_native_sampler()
        return True

    @staticmethod
    def reduce(local_grid: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """MPI-sum a rank-local complex grid with the native reducer.

        Args:
            local_grid: Rank-local complex accumulator.

        Returns:
            MPI-summed grid on every rank in the active Meep process group.
        """
        _require_native_sampler()
        try:
            return native_sampler.reduce_complex_grid_sum(
                np.asarray(local_grid, dtype=np.complex128)
            )
        except Exception as exc:
            _handle_native_sampler_failure("gradient reduction", exc)
            raise

    @staticmethod
    def reduce_inplace(local_grid: np.ndarray) -> np.ndarray:
        """MPI-sum a writable rank-local complex grid with buffer reuse."""
        _require_native_sampler()
        local_dtype = np.asarray(local_grid).dtype
        native_dtype = (
            local_dtype
            if local_dtype in (np.dtype(np.complex64), np.dtype(np.complex128))
            else np.dtype(np.complex128)
        )
        local_array = np.require(
            local_grid,
            dtype=native_dtype,
            requirements=("C", "W", "A"),
        )
        try:
            return native_sampler.reduce_complex_grid_sum_inplace(local_array)
        except Exception as exc:
            _handle_native_sampler_failure("in-place array reduction", exc)
            raise

    @staticmethod
    def reduce_real_inplace(local_grid: np.ndarray) -> np.ndarray:
        """MPI-sum a writable real grid with active-group buffer reuse."""
        _require_native_sampler()
        local_dtype = np.asarray(local_grid).dtype
        native_dtype = (
            local_dtype
            if local_dtype in (np.dtype(np.float32), np.dtype(np.float64))
            else np.dtype(np.float64)
        )
        local_array = np.require(
            local_grid,
            dtype=native_dtype,
            requirements=("C", "W", "A"),
        )
        try:
            return native_sampler.reduce_real_grid_sum_inplace(local_array)
        except Exception as exc:
            _handle_native_sampler_failure("in-place real-array reduction", exc)
            raise


def _native_design_call(operation: str, function, *args):
    """Call an exact-design native operation without unsafe MPI fallback."""
    try:
        return function(*args)
    except Exception as exc:
        _handle_native_sampler_failure(operation, exc)
        raise


class NativeDesignField(_NativePlanFieldsGuard):
    """Exact rank-local Yee samples with a MaterialGrid-transpose plan.

    Args:
        sim: Initialized Meep simulation containing the design geometry.
        component: Electric-field component sampled on its native Yee points.
        design: Design-grid metadata and MaterialGrid being differentiated.
        expected_signature: Optional rank-local Yee-layout signature that the
            newly created plan must match.
    """

    def __init__(
        self,
        sim: mp.Simulation,
        component: int,
        design,
        expected_signature: Optional[NDArray[np.int64]] = None,
    ) -> None:
        _require_native_sampler()
        design_is_cylindrical = bool(
            getattr(design, "is_cylindrical", False)
        )
        simulation_is_cylindrical = bool(
            getattr(sim, "is_cylindrical", False)
            or getattr(sim, "dimensions", None) == mp.CYLINDRICAL
        )
        if design_is_cylindrical != simulation_is_cylindrical:
            raise ValueError(
                "DesignGrid coordinate system must match the Meep simulation"
            )
        if design_is_cylindrical and component not in (mp.Er, mp.Ep, mp.Ez):
            raise ValueError(
                "cylindrical native design plans require Er, Ep, or Ez"
            )
        _require_no_meep_symmetries(sim)
        k_point = getattr(sim, "k_point", None)
        if k_point is not None and k_point is not False and not np.allclose(
            (k_point.x, k_point.y, k_point.z),
            (0.0, 0.0, 0.0),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("exact native design sampling requires k_point=0")
        if getattr(sim, "eps_averaging", True):
            raise ValueError("exact native design sampling requires eps_averaging=False")
        if (
            getattr(sim, "fields", None) is not None
            and int(getattr(sim, "dimensions", design.dimensions)) != design.dimensions
        ):
            raise ValueError(
                "DesignGrid dimensionality must match the Meep simulation"
            )
        matching_objects = [
            obj for obj in getattr(sim, "geometry", ())
            if getattr(obj, "material", None) is design.material_grid
        ]
        if len(matching_objects) != 1 or not isinstance(matching_objects[0], mp.Block):
            raise ValueError(
                "exact native design sampling requires one MaterialGrid-backed Block"
            )
        block = matching_objects[0]
        expected_vectors = (
            (block.e1, (1.0, 0.0, 0.0)),
            (block.e2, (0.0, 1.0, 0.0)),
            (block.e3, (0.0, 0.0, 1.0)),
        )
        if any(
            not np.allclose((axis.x, axis.y, axis.z), expected, rtol=0.0, atol=1e-12)
            for axis, expected in expected_vectors
        ):
            raise ValueError("exact native design sampling requires an axis-aligned Block")
        if design_is_cylindrical:
            active_axes = (("x", mp.R), ("z", mp.Z))
        elif design.dimensions == 3:
            active_axes = (("x", mp.X), ("y", mp.Y), ("z", mp.Z))
        else:
            active_axes = (("x", mp.X), ("y", mp.Y))
        if not np.allclose(
            tuple(
                float(getattr(vector, axis))
                for vector in (block.center, block.size)
                for axis, _ in active_axes
            ),
            tuple(
                float(getattr(vector, axis))
                for vector in (design.center, design.size)
                for axis, _ in active_axes
            ),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("DesignGrid center/size must match its MaterialGrid Block")
        cell_center = getattr(sim, "geometry_center", mp.Vector3())
        cell_size = sim.cell_size
        for layer in getattr(sim, "boundary_layers", ()):
            thickness = float(getattr(layer, "thickness", 0.0))
            direction = getattr(layer, "direction", mp.ALL)
            side = getattr(layer, "side", mp.ALL)
            for axis, axis_direction in active_axes:
                if direction not in (mp.ALL, axis_direction):
                    continue
                domain_center = float(getattr(cell_center, axis))
                domain_size = float(getattr(cell_size, axis))
                block_center = float(getattr(block.center, axis))
                block_size = float(getattr(block.size, axis))
                if design_is_cylindrical and axis == "x":
                    domain_low = domain_center
                    domain_high = domain_center + domain_size
                else:
                    domain_low = domain_center - 0.5 * domain_size
                    domain_high = domain_center + 0.5 * domain_size
                block_low = block_center - 0.5 * block_size
                block_high = block_center + 0.5 * block_size
                overlaps_low = (
                    side in (mp.ALL, mp.Low)
                    and (
                        not design_is_cylindrical
                        or axis != "x"
                        or domain_low > 1e-12
                    )
                    and block_low < domain_low + thickness - 1e-12
                )
                overlaps_high = (
                    side in (mp.ALL, mp.High)
                    and block_high > domain_high - thickness + 1e-12
                )
                if overlaps_low or overlaps_high:
                    raise ValueError(
                        "exact native design sampling does not support a design "
                        "Block overlapping PML or absorber layers"
                    )
        medium_1 = design.material_grid.medium1
        medium_2 = design.material_grid.medium2
        expected_material_factor = (
            float(medium_2.epsilon_diag.x) - float(medium_1.epsilon_diag.x)
        )
        if not np.isclose(
            float(design.material_factor),
            expected_material_factor,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                "DesignGrid material_factor must match its MaterialGrid media contrast"
            )
        geps = getattr(sim, "geps", None)
        if geps is None:
            raise RuntimeError("Meep geom_epsilon is unavailable after simulation initialization")

        self.component = component
        self.shape = tuple(design.shape)
        self.plan = None
        self._bind_native_plan_fields(sim)
        self.fields_are_real = bool(sim.fields.is_real)
        try:
            self.plan = _native_design_call(
                "design-plan creation",
                native_sampler.create_native_design_plan,
                int(sim.fields.this),
                int(geps.this),
                float(design.center.x),
                float(design.center.y),
                float(design.center.z),
                float(design.size.x),
                float(design.size.y),
                float(design.size.z),
                int(self.shape[0]),
                int(self.shape[1]),
                int(self.shape[2]) if design.dimensions == 3 else 1,
                int(component),
            )
            self.local_size = int(
                _native_design_call(
                    "design-plan size query",
                    native_sampler.native_design_plan_local_size,
                    self.plan,
                )
            )
            self.signature = np.asarray(
                _native_design_call(
                    "design-plan signature query",
                    native_sampler.native_design_plan_signature,
                    self.plan,
                ),
                dtype=np.int64,
            )
        except Exception:
            self.plan = None
            raise
        local_signature_mismatch = expected_signature is not None and not np.array_equal(
            self.signature,
            np.asarray(expected_signature, dtype=np.int64),
        )
        signature_mismatch = local_signature_mismatch
        if expected_signature is not None and mp.count_processors() > 1:
            signature_mismatch = bool(mp.sum_to_all(int(local_signature_mismatch)))
        if signature_mismatch:
            self.plan = None
            raise RuntimeError(
                "forward and adjoint simulations have different rank-local Yee layouts"
            )

    def ensure_plan(self) -> Any:
        """Return the plan after verifying its bound Meep fields are current."""
        self._require_native_plan_fields()
        if self.plan is None:
            raise RuntimeError("native design plan has been released")
        return self.plan

    def sample_into(self, destination: np.ndarray) -> None:
        """Write current rank-local native Yee samples into an array.

        Args:
            destination: Writable one-dimensional array with `local_size`
                entries. Real Meep fields may use a real destination; complex
                fields require a complex destination.
        """
        destination = np.asarray(destination)
        if destination.ndim != 1 or destination.size != self.local_size:
            raise ValueError("native design history destination has the wrong shape")
        if not destination.flags.writeable:
            raise ValueError("native design history destination must be writable")
        if (
            self.fields_are_real
            and destination.dtype == np.float64
            and destination.flags.c_contiguous
        ):
            _native_design_call(
                "exact real design sampling",
                native_sampler.sample_native_design_plan_real_into,
                self.ensure_plan(),
                destination,
            )
            return
        if destination.dtype == np.complex128 and destination.flags.c_contiguous:
            _native_design_call(
                "exact design sampling",
                native_sampler.sample_native_design_plan_into,
                self.ensure_plan(),
                destination,
            )
            return
        if not self.fields_are_real and not np.issubdtype(
            destination.dtype,
            np.complexfloating,
        ):
            raise ValueError(
                "complex Meep fields require a complex design-history destination"
            )
        sampled = np.empty(
            self.local_size,
            dtype=np.float64 if self.fields_are_real else np.complex128,
        )
        _native_design_call(
            "exact real design sampling" if self.fields_are_real else "exact design sampling",
            (
                native_sampler.sample_native_design_plan_real_into
                if self.fields_are_real
                else native_sampler.sample_native_design_plan_into
            ),
            self.ensure_plan(),
            sampled,
        )
        destination[:] = sampled.astype(destination.dtype, copy=False)

    def accumulate(
        self,
        forward_values: np.ndarray,
        accumulator: np.ndarray,
    ) -> None:
        """Accumulate one exact field-product row into a design gradient.

        Args:
            forward_values: Rank-local forward-field values in plan order.
            accumulator: Writable float64 accumulator for real fields or
                complex128 accumulator for complex fields.
        """
        forward_array = np.asarray(forward_values)
        accumulator_array = np.asarray(accumulator)
        if (
            self.fields_are_real
            and np.issubdtype(forward_array.dtype, np.floating)
            and accumulator_array.dtype == np.float64
        ):
            _native_design_call(
                "exact real design accumulation",
                native_sampler.accumulate_native_design_real_product_local_inplace,
                self.ensure_plan(),
                np.ascontiguousarray(forward_array, dtype=np.float64),
                self._accumulator_view(accumulator_array, np.float64),
            )
            return
        _native_design_call(
            "exact design accumulation",
            native_sampler.accumulate_native_design_product_local_inplace,
            self.ensure_plan(),
            np.asarray(forward_array, dtype=np.complex128),
            self._accumulator_view(accumulator_array, np.complex128),
        )

    def accumulate_midpoint(
        self,
        forward_values: np.ndarray,
        accumulator: np.ndarray,
    ) -> bool:
        """Accumulate using the midpoint of consecutive adjoint samples.

        Args:
            forward_values: Rank-local forward-field values in plan order.
            accumulator: Writable float64 accumulator for real fields or
                complex128 accumulator for complex fields.

        Returns:
            `False` while the first adjoint row is cached; `True` after a
            midpoint contribution is accumulated.
        """
        forward_array = np.asarray(forward_values)
        accumulator_array = np.asarray(accumulator)
        if (
            self.fields_are_real
            and np.issubdtype(forward_array.dtype, np.floating)
            and accumulator_array.dtype == np.float64
        ):
            return bool(
                _native_design_call(
                    "exact real midpoint design accumulation",
                    native_sampler.accumulate_native_design_real_midpoint_product_local_inplace,
                    self.ensure_plan(),
                    np.ascontiguousarray(forward_array, dtype=np.float64),
                    self._accumulator_view(accumulator_array, np.float64),
                )
            )
        return bool(
            _native_design_call(
                "exact midpoint design accumulation",
                native_sampler.accumulate_native_design_midpoint_product_local_inplace,
                self.ensure_plan(),
                np.asarray(forward_array, dtype=np.complex128),
                self._accumulator_view(accumulator_array, np.complex128),
            )
        )

    def _accumulator_view(
        self,
        accumulator: np.ndarray,
        dtype: np.dtype,
    ) -> np.ndarray:
        array = np.asarray(accumulator)
        if (
            array.dtype != np.dtype(dtype)
            or array.shape != self.shape
            or not array.flags.c_contiguous
            or not array.flags.writeable
        ):
            raise ValueError(
                "native design accumulator must be a writable C-contiguous "
                f"{np.dtype(dtype).name} array with the DesignGrid shape"
            )
        return array.reshape(-1)

    def release(self) -> None:
        """Release the native plan reference held by this wrapper."""
        self.plan = None


class FastGradientGrid:
    """Rank-local gradient accumulator with an MPI reduction at finalize().

    Args:
        sim: Active adjoint Meep simulation.
        component: Meep field component sampled from the adjoint fields.
        coords_x: x coordinates of the design grid.
        coords_y: y coordinates of the design grid.
    """

    def __init__(
        self,
        sim: mp.Simulation,
        component: int,
        coords_x: Sequence[float],
        coords_y: Sequence[float],
    ) -> None:
        self.field = FastFieldGrid(sim, component, coords_x, coords_y)
        self.local = np.zeros(self.field.shape, dtype=complex)
        self.needs_reduce = False
        self._fixed_indices = None

    def accumulate(self, multiplier: NDArray[np.complex128]) -> None:
        """Add one time-slice contribution to the local gradient accumulator.

        Args:
            multiplier: Grid-shaped forward factor multiplied by the current
                adjoint field sample.
        """
        self.field.accumulate_product(multiplier, self.local)
        self.needs_reduce = True

    def accumulate_indexed(
        self,
        indices: NDArray[np.int64],
        values: NDArray[np.complex128],
    ) -> None:
        """Add selected time-slice contributions to the local accumulator.

        Args:
            indices: Flat design-grid indices with nonzero forward factors.
            values: Forward factors matching `indices`.
        """
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        values = np.asarray(values, dtype=np.complex128).reshape(-1)
        if values.shape[0] != indices.shape[0]:
            raise ValueError("values must match indices length")
        self.field.accumulate_indexed_product(indices, values, self.local)
        self.needs_reduce = True

    def configure_fixed_indices(self, indices: NDArray[np.int64]) -> None:
        """Configure the immutable index order used by repeated accumulation."""
        self._fixed_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        self.field.configure_accumulation_indices(self._fixed_indices)

    def accumulate_fixed(self, values: NDArray[np.complex128]) -> None:
        """Accumulate one row using indices configured by `configure_fixed_indices`."""
        if self._fixed_indices is None:
            raise RuntimeError("fixed accumulation indices are not configured")
        values = np.asarray(values, dtype=np.complex128).reshape(-1)
        if values.size != self._fixed_indices.size:
            raise ValueError("values must match configured fixed indices")
        self.field.accumulate_configured_product(values, self.local)
        self.needs_reduce = True

    def accumulate_fixed_time_difference(
        self,
        values: NDArray[np.complex128],
        dt: float,
    ) -> bool:
        """Accumulate values times the adjoint field's forward time difference.

        The first call caches the current adjoint field and returns `False`.
        Each later call accumulates `values * (current - previous) / dt` and
        returns `True`.
        """
        if self._fixed_indices is None:
            raise RuntimeError("fixed accumulation indices are not configured")
        values = np.asarray(values, dtype=np.complex128).reshape(-1)
        if values.size != self._fixed_indices.size:
            raise ValueError("values must match configured fixed indices")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a positive finite number")

        accumulated = self.field.accumulate_configured_difference_product(
            values,
            dt,
            self.local,
        )
        if accumulated:
            self.needs_reduce = True
        return accumulated

    def finalize(self) -> NDArray[np.complex128]:
        """Return the accumulated complex gradient grid.

        Returns:
            MPI-reduced grid after native rank-local accumulation. All active
            ranks must call this collective even when their local grid is zero.
        """
        return FastFieldGrid.reduce(self.local)
