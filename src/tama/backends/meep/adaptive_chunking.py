"""Adaptive Meep chunk balancing with TDA callback timing."""

import copy
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import meep as mp
import numpy as np
from meep.chunk_balancer import ChunkBalancer
from meep.timing_measurements import MeepTimingMeasurements

from . import native_sampler
from .chunking import (
    _meep_grid_axis_boundaries,
    regularize_source_size_and_amplitude,
)


@dataclass(frozen=True)
class AdaptiveChunkBalanceReport:
    """Summary of one adaptive chunk-layout decision.

    Attributes:
        rank_work_seconds: Per-rank work values used by the decision. In MPI
            these combine Meep and TAMA timing; in serial this contains the
            supplied extra-work time.
        max_to_mean: Maximum rank work divided by mean rank work.
        moved_boundaries: Number of partition split planes whose positions
            changed.
        applied: Whether the selected layout differs from the current layout.
        reason: Human-readable explanation of the decision.
        critical_path_seconds: Maximum rank-work estimate, or maximum gathered
            wall time when wall-time measurements were supplied. In serial it
            is the supplied wall time or extra-work time.
        topology: Split-axis label for the current partition tree.
        next_topology: Split-axis label for the selected partition tree.
        topology_changed: Reserved topology-change flag; currently always
            `False` because the native partition tree is preserved.
        exploring: Whether workload calibration remains active after this
            decision.
    """

    rank_work_seconds: Tuple[float, ...]
    max_to_mean: float
    moved_boundaries: int
    applied: bool
    reason: str
    critical_path_seconds: float = 0.0
    topology: str = ""
    next_topology: str = ""
    topology_changed: bool = False
    exploring: bool = False


@dataclass(frozen=True)
class AdaptiveSourceBoundaryDecision:
    """Effective adjoint-source settings for a Meep-native chunk layout.

    Attributes:
        method: Selected handling mode: `serial`, `layout`, or `finite`.
        reason: Human-readable explanation of the selected mode.
        chunk_layout: Effective Meep partition assigned to the simulation
            specification, or `None` when no explicit layout is used.
        source_sizes: Effective adjoint-source size for each target.
        source_amplitudes: Effective adjoint-source amplitude for each target.
        changed_axes: Source axes widened by finite-source fallback for each
            target.
    """

    method: str
    reason: str
    chunk_layout: Optional[mp.BinaryPartition]
    source_sizes: Tuple[mp.Vector3, ...]
    source_amplitudes: Tuple[complex, ...]
    changed_axes: Tuple[Tuple[str, ...], ...]


def _combine_timing_measurements(
    measurements: Sequence[MeepTimingMeasurements],
) -> MeepTimingMeasurements:
    if not measurements:
        raise ValueError("at least one Meep timing measurement is required")
    names = measurements[0].measurement_names
    combined = {
        name: np.sum(
            [np.asarray(item.measurements[name], dtype=float) for item in measurements],
            axis=0,
        ).tolist()
        for name in names
    }
    return MeepTimingMeasurements(
        measurements=combined,
        elapsed_time=float(
            sum(max(float(item.elapsed_time), 0.0) for item in measurements)
        ),
        num_time_steps=int(sum(item.num_time_steps for item in measurements)),
        time_per_step=[],
        dft_relative_change=[],
        overlap_relative_change=[],
        relative_energy=[],
    )


def _is_cylindrical(simulation) -> bool:
    return bool(
        getattr(simulation, "is_cylindrical", False)
        or getattr(simulation, "dimensions", None) == mp.CYLINDRICAL
    )


def _axis_index(axis) -> int:
    if axis in (mp.X, mp.R):
        return 0
    if axis == mp.Y:
        return 1
    if axis == mp.Z:
        return 2
    raise ValueError("partition split direction must be mp.X, mp.Y, mp.Z, or mp.R")


def _axis_name(axis) -> str:
    if axis == mp.R:
        return "R"
    return ("X", "Y", "Z")[_axis_index(axis)]


def _topology_label(partition: mp.BinaryPartition) -> str:
    axes = []

    def collect(node):
        if node.proc_id is not None:
            return
        axis_name = _axis_name(node.split_dir)
        if axis_name not in axes:
            axes.append(axis_name)
        collect(node.left)
        collect(node.right)

    collect(partition)
    return axes[0] if len(axes) == 1 else "mixed(" + ",".join(axes) + ")"


def _leaf_proc_ids(partition: mp.BinaryPartition) -> Tuple[int, ...]:
    if partition.proc_id is not None:
        return (int(partition.proc_id),)
    return (*_leaf_proc_ids(partition.left), *_leaf_proc_ids(partition.right))


def _simulation_partition(sim: mp.Simulation):
    structure = getattr(sim, "structure", None)
    get_binary_partition = getattr(structure, "get_binary_partition", None)
    if callable(get_binary_partition):
        partition = get_binary_partition()
        if isinstance(partition, mp.BinaryPartition):
            return partition
    return getattr(sim, "chunk_layout", None)


def _validate_partition_owners(
    partition: mp.BinaryPartition,
    chunk_owners,
) -> None:
    proc_ids = _leaf_proc_ids(partition)
    if len(set(proc_ids)) != len(proc_ids):
        raise ValueError(
            "adaptive balancing requires a unique process ID for each chunk"
        )
    owner_ids = {int(owner) for owner in np.asarray(chunk_owners).reshape(-1)}
    if set(proc_ids) != owner_ids:
        raise ValueError(
            "partition process IDs must match the initialized Meep chunk owners"
        )


def _rebalance_cylindrical_partition(
    timing_measurements: MeepTimingMeasurements,
    partition: mp.BinaryPartition,
    chunk_volumes,
    chunk_owners,
    *,
    low: np.ndarray,
    high: np.ndarray,
    sensitivity: float,
    meep_balancer: ChunkBalancer,
) -> mp.BinaryPartition:
    """Resize an R-Z partition using Meep's timing-based split formula."""
    rank_work = np.asarray(
        meep_balancer._compute_working_times_per_process(timing_measurements),
        dtype=float,
    )
    cells_by_rank = np.zeros(rank_work.size, dtype=float)
    for volume, owner in zip(chunk_volumes, chunk_owners):
        cells_by_rank[int(owner)] += volume.num_direction(mp.R) * volume.num_direction(
            mp.Z
        )

    candidate = copy.deepcopy(partition)

    def update(current, updated, node_low, node_high):
        if current.proc_id is not None:
            return
        if current.split_dir not in (mp.R, mp.Z):
            raise ValueError(
                "cylindrical adaptive balancing requires R-Z partition axes"
            )

        left_ids = _leaf_proc_ids(current.left)
        right_ids = _leaf_proc_ids(current.right)
        t_left = float(np.sum(rank_work[list(left_ids)]))
        t_right = float(np.sum(rank_work[list(right_ids)]))
        v_left = float(np.sum(cells_by_rank[list(left_ids)]))
        v_right = float(np.sum(cells_by_rank[list(right_ids)]))
        left_weight = v_left * t_right * len(left_ids)
        right_weight = v_right * t_left * len(right_ids)
        total_weight = left_weight + right_weight

        axis_index = _axis_index(current.split_dir)
        if total_weight > 0.0 and np.isfinite(total_weight):
            split_fraction = left_weight / total_weight
            predicted = (
                node_low[axis_index]
                + (node_high[axis_index] - node_low[axis_index]) * split_fraction
            )
            updated.split_pos = (
                sensitivity * predicted + (1.0 - sensitivity) * current.split_pos
            )

        left_high = node_high.copy()
        left_high[axis_index] = updated.split_pos
        right_low = node_low.copy()
        right_low[axis_index] = updated.split_pos
        update(current.left, updated.left, node_low, left_high)
        update(current.right, updated.right, right_low, node_high)

    update(partition, candidate, low.copy(), high.copy())
    return candidate


def _restore_nonfinite_split_positions(
    candidate: mp.BinaryPartition,
    previous: mp.BinaryPartition,
) -> None:
    """Keep prior boundaries where Meep's zero-work ratio is undefined."""
    if candidate.proc_id is not None:
        return
    if not np.isfinite(candidate.split_pos):
        candidate.split_pos = previous.split_pos
    _restore_nonfinite_split_positions(candidate.left, previous.left)
    _restore_nonfinite_split_positions(candidate.right, previous.right)


def _vector_array(vector: mp.Vector3) -> np.ndarray:
    return np.asarray((vector.x, vector.y, vector.z), dtype=float)


def _point_bounds(points: Sequence[mp.Vector3]) -> np.ndarray:
    bounds = []
    for point in points:
        center = _vector_array(point)
        bounds.append((center, center))
    return np.asarray(bounds, dtype=float).reshape(-1, 2, 3)


def _source_bounds(volumes: Sequence[mp.Volume]) -> np.ndarray:
    bounds = []
    for volume in volumes:
        center = _vector_array(volume.center)
        size = _vector_array(volume.size)
        if not np.all(np.isfinite(center)):
            raise ValueError("source volume centers must be finite")
        if not np.all(np.isfinite(size)):
            raise ValueError("source volume sizes must be finite")
        if np.any(size < 0):
            raise ValueError("source volume sizes must be nonnegative")
        half_size = 0.5 * size
        bounds.append((center - half_size, center + half_size))
    return np.asarray(bounds, dtype=float).reshape(-1, 2, 3)


def _forward_source_volumes(sources: Sequence[mp.Source]) -> Tuple[mp.Volume, ...]:
    try:
        sources = tuple(sources)
    except TypeError as exc:
        raise TypeError(
            "forward_sources must be a sequence of mp.Source instances"
        ) from exc

    volumes = []
    for source in sources:
        if not isinstance(source, mp.Source):
            raise TypeError("forward_sources must contain only mp.Source instances")
        if isinstance(source, mp.IndexedSource):
            continue
        center = getattr(source, "center", None)
        size = getattr(source, "size", None)
        if not isinstance(center, mp.Vector3) or not isinstance(size, mp.Vector3):
            raise ValueError(
                "each non-indexed forward source must expose Vector3 center and size"
            )
        volumes.append(
            mp.Volume(
                center=mp.Vector3(center.x, center.y, center.z),
                size=mp.Vector3(size.x, size.y, size.z),
            )
        )
    result = tuple(volumes)
    _source_bounds(result)
    return result


def _cell_bounds(
    cell_size: mp.Vector3,
    geometry_center: Optional[mp.Vector3],
    resolution: float,
    *,
    cylindrical: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    center_vector = geometry_center if geometry_center is not None else mp.Vector3()
    center = _vector_array(center_vector)
    size = _vector_array(cell_size)
    if cylindrical:
        grid = mp.volcyl(float(size[0]), float(size[2]), resolution)
        grid.center_origin()
        grid.shift_origin(mp.veccyl(float(center[0]), float(center[2])))
        surroundings = grid.surroundings()
        minimum = surroundings.get_min_corner()
        maximum = surroundings.get_max_corner()
        return (
            np.asarray((minimum.r(), center[1], minimum.z()), dtype=float),
            np.asarray((maximum.r(), center[1], maximum.z()), dtype=float),
        )

    low = center - 0.5 * size
    high = center + 0.5 * size
    for axis_index, length in enumerate(size):
        if length <= 0:
            continue
        boundaries = _meep_grid_axis_boundaries(
            length=length,
            center=center[axis_index],
            resolution=resolution,
        )
        low[axis_index] = boundaries[0]
        high[axis_index] = boundaries[-1]
    return low, high


def _minimum_axis_cells(
    partition: mp.BinaryPartition,
    axis,
    min_chunk_cells: int,
) -> int:
    if partition.proc_id is not None:
        return min_chunk_cells
    left = _minimum_axis_cells(partition.left, axis, min_chunk_cells)
    right = _minimum_axis_cells(partition.right, axis, min_chunk_cells)
    return left + right if partition.split_dir == axis else max(left, right)


def _split_positions(partition: mp.BinaryPartition) -> Tuple[float, ...]:
    if partition.proc_id is not None:
        return ()
    return (
        float(partition.split_pos),
        *_split_positions(partition.left),
        *_split_positions(partition.right),
    )


def _partition_axes(partition: mp.BinaryPartition) -> Tuple:
    axes = []

    def collect(node):
        if node.proc_id is not None:
            return
        if node.split_dir not in axes:
            axes.append(node.split_dir)
        collect(node.left)
        collect(node.right)

    collect(partition)
    return tuple(axes)


def _combine_bounds(*arrays: np.ndarray) -> np.ndarray:
    nonempty = [array for array in arrays if array.size]
    if not nonempty:
        return np.empty((0, 2, 3), dtype=float)
    return np.concatenate(nonempty, axis=0)


def _simulation_dimensions(simulation_spec) -> int:
    if _is_cylindrical(simulation_spec):
        return 2
    dimensions = getattr(simulation_spec, "dimensions", None)
    if dimensions in (2, 3):
        return int(dimensions)
    size = _vector_array(simulation_spec.cell_size)
    return 3 if size[2] > 0 else 2


def _source_volumes(
    positions: Sequence[mp.Vector3],
    sizes: Sequence[mp.Vector3],
) -> Tuple[mp.Volume, ...]:
    return tuple(
        mp.Volume(center=position, size=size)
        for position, size in zip(positions, sizes)
    )


class AdaptiveAdjointChunkBalancer:
    """Adapt boundaries of Meep's native topology using measured TDA work.

    `prepare_initial_layout` preserves Meep's native partition tree and
    projects only its split positions. Ordinary forward sources discovered
    from `SimulationSpec` impose source-boundary constraints; indexed sources
    do not. Other Cartesian point targets use a layout-first policy and, in
    `auto` mode, a one-cell finite-source fallback when source placement alone
    blocks every safe split position. Cylindrical indexed adjoint sources
    remain exact zero-size sources. Workload calibration uses ordinary
    gradient evaluations: it records at least three observations and freezes the
    best measured layout after two consecutive balanced observations, two
    consecutive observations without at least one-percent best-score
    improvement, or eight observations.
    A trial whose critical-rank work regresses by at least two percent is
    rolled back immediately without ending calibration.

    Args:
        protected_points: Permanent point constraints that are never
            regularized.
        source_volumes: Additional permanent source supports that are not
            discoverable from `SimulationSpec.sources`. Adjoint targets and
            ordinary forward sources are supplied automatically by TDA
            objectives.
        sensitivity: Meep chunk-balancer boundary-update sensitivity.
        imbalance_threshold: Maximum accepted rank-work max/mean ratio.
        protected_gap_cells: Minimum split distance from zero-extent sources
            and permanent protected points, in Meep grid cells.
        min_chunk_cells: Minimum grid cells retained by each partition leaf.
        source_boundary_mode: `auto` tries layout projection before finite
            fallback for non-indexed targets, `layout` disables fallback,
            and `finite` selects ordinary finite Cartesian adjoint sources
            on the topology's split axes (or all simulated axes in serial).
            Cylindrical simulations support `auto` and `layout` only
            because their exact indexed adjoint sources must remain zero size.
        finite_source_width_cells: Width assigned to each regularized source
            axis, in Meep grid cells.
    """

    _MIN_REBALANCE_OBSERVATIONS = 3
    _MAX_REBALANCE_OBSERVATIONS = 8
    _REQUIRED_CONVERGENCE_STREAK = 2
    _MIN_BEST_SCORE_IMPROVEMENT = 0.01
    _MAX_SCORE_REGRESSION = 0.02

    def __init__(
        self,
        *,
        protected_points: Sequence[mp.Vector3] = (),
        source_volumes: Sequence[mp.Volume] = (),
        sensitivity: float = 0.5,
        imbalance_threshold: float = 1.05,
        protected_gap_cells: float = 2.0,
        min_chunk_cells: int = 4,
        source_boundary_mode: str = "auto",
        finite_source_width_cells: float = 1.0,
    ) -> None:
        if not 0.0 <= sensitivity <= 1.0:
            raise ValueError("sensitivity must be between 0 and 1")
        if imbalance_threshold < 1.0:
            raise ValueError("imbalance_threshold must be at least 1")
        if protected_gap_cells < 0:
            raise ValueError("protected_gap_cells must be nonnegative")
        if min_chunk_cells <= 0:
            raise ValueError("min_chunk_cells must be positive")
        if source_boundary_mode not in ("auto", "layout", "finite"):
            raise ValueError(
                "source_boundary_mode must be 'auto', 'layout', or 'finite'"
            )
        if finite_source_width_cells <= 0:
            raise ValueError("finite_source_width_cells must be positive")
        self.protected_points = tuple(protected_points)
        self.source_volumes = tuple(source_volumes)
        _source_bounds(self.source_volumes)
        self.sensitivity = float(sensitivity)
        self.imbalance_threshold = float(imbalance_threshold)
        self.protected_gap_cells = float(protected_gap_cells)
        self.min_chunk_cells = int(min_chunk_cells)
        self.source_boundary_mode = source_boundary_mode
        self.finite_source_width_cells = float(finite_source_width_cells)
        self.last_report: Optional[AdaptiveChunkBalanceReport] = None
        self.last_source_boundary_decision: Optional[AdaptiveSourceBoundaryDecision] = (
            None
        )
        self._meep_balancer = ChunkBalancer()
        self._prepared_simulation_spec = None
        self._prepared_source_signature = None
        self._effective_source_volumes: Tuple[mp.Volume, ...] = ()
        self._effective_forward_source_volumes: Tuple[mp.Volume, ...] = ()
        self._rebalance_observations = 0
        self._best_layout: Optional[mp.BinaryPartition] = None
        self._best_rank_work_seconds = np.inf
        self._frozen_layout: Optional[mp.BinaryPartition] = None
        self._balanced_observation_streak = 0
        self._stalled_best_score_streak = 0

    @property
    def is_frozen(self) -> bool:
        """Whether calibration has selected its final best layout."""
        return self._frozen_layout is not None

    @property
    def needs_rebalance(self) -> bool:
        """Whether this process group still needs workload calibration."""
        return mp.count_processors() > 1 and not self.is_frozen

    @staticmethod
    def capture_timing(sim: mp.Simulation) -> MeepTimingMeasurements:
        """Capture per-rank Meep timing before the simulation is reset.

        Args:
            sim: Initialized Meep simulation whose timing data is captured.

        Returns:
            Snapshot containing per-process Meep timing measurements for the
            active process group.
        """
        return MeepTimingMeasurements.new_from_simulation(sim)

    def prepare_initial_layout(
        self,
        simulation_spec,
        *,
        adjoint_source_positions: Sequence[mp.Vector3] = (),
        adjoint_source_size: Optional[mp.Vector3] = None,
        adjoint_source_amplitude: complex = 1.0,
        adjoint_source_sizes: Optional[Sequence[mp.Vector3]] = None,
        adjoint_source_amplitudes: Optional[Sequence[complex]] = None,
        indexed_source_mask: Optional[Sequence[bool]] = None,
        forward_sources: Sequence[mp.Source] = (),
    ) -> AdaptiveSourceBoundaryDecision:
        """Project Meep's native topology and resolve adjoint-source boundaries.

        In MPI, `simulation_spec.chunk_layout` must initially be unset. This
        method assigns the projected layout to that attribute. Returned source
        settings follow the order of `adjoint_source_positions`. Plural
        source-setting arguments override their shared singular counterparts.

        Args:
            simulation_spec: Simulation specification to probe and update.
            adjoint_source_positions: Target positions for the adjoint sources.
            adjoint_source_size: Shared source size used when
                `adjoint_source_sizes` is omitted.
            adjoint_source_amplitude: Shared source amplitude used when
                `adjoint_source_amplitudes` is omitted.
            adjoint_source_sizes: Optional per-target source sizes.
            adjoint_source_amplitudes: Optional per-target source amplitudes.
            indexed_source_mask: Optional per-target flags. `True` marks an
                exact zero-size indexed source that does not impose a geometric
                chunk-boundary constraint.
            forward_sources: Resolved forward Meep sources. Ordinary `Source`
                instances and subclasses constrain zero-extent split axes;
                `IndexedSource` instances are excluded.

        Returns:
            Effective chunk layout and per-target adjoint-source settings.
        """
        positions = tuple(adjoint_source_positions)
        forward_source_volumes = _forward_source_volumes(forward_sources)
        cylindrical = _is_cylindrical(simulation_spec)
        if cylindrical and self.source_boundary_mode == "finite":
            raise ValueError(
                "cylindrical adaptive balancing does not support finite "
                "adjoint-source regularization"
            )
        if adjoint_source_sizes is None:
            base_size = (
                adjoint_source_size if adjoint_source_size is not None else mp.Vector3()
            )
            base_sizes = tuple(
                mp.Vector3(base_size.x, base_size.y, base_size.z) for _ in positions
            )
        else:
            if len(adjoint_source_sizes) != len(positions):
                raise ValueError(
                    "adjoint_source_sizes must match adjoint_source_positions"
                )
            base_sizes = tuple(
                mp.Vector3(size.x, size.y, size.z) for size in adjoint_source_sizes
            )
        if adjoint_source_amplitudes is None:
            base_amplitudes = tuple(
                complex(adjoint_source_amplitude) for _ in positions
            )
        else:
            if len(adjoint_source_amplitudes) != len(positions):
                raise ValueError(
                    "adjoint_source_amplitudes must match " "adjoint_source_positions"
                )
            base_amplitudes = tuple(
                complex(amplitude) for amplitude in adjoint_source_amplitudes
            )
        if indexed_source_mask is None:
            indexed_source_mask = tuple(cylindrical for _ in positions)
        else:
            if len(indexed_source_mask) != len(positions):
                raise ValueError(
                    "indexed_source_mask must match adjoint_source_positions"
                )
            if any(
                not isinstance(value, (bool, np.bool_)) for value in indexed_source_mask
            ):
                raise ValueError("indexed_source_mask values must be boolean")
            indexed_source_mask = tuple(bool(value) for value in indexed_source_mask)
        if any(
            indexed and not np.array_equal(_vector_array(size), np.zeros(3))
            for indexed, size in zip(indexed_source_mask, base_sizes)
        ):
            raise ValueError("indexed adjoint sources require zero source size")
        if self.source_boundary_mode == "finite" and any(indexed_source_mask):
            raise ValueError(
                "indexed adjoint sources are incompatible with finite "
                "source-boundary handling"
            )
        signature = (
            tuple(tuple(_vector_array(position)) for position in positions),
            tuple(tuple(_vector_array(size)) for size in base_sizes),
            base_amplitudes,
            indexed_source_mask,
            tuple(
                (
                    tuple(_vector_array(volume.center)),
                    tuple(_vector_array(volume.size)),
                )
                for volume in forward_source_volumes
            ),
        )
        if self.last_source_boundary_decision is not None:
            if (
                simulation_spec is not self._prepared_simulation_spec
                or signature != self._prepared_source_signature
            ):
                raise ValueError(
                    "an adaptive chunk balancer cannot be reused with a different "
                    "simulation or adjoint-source/forward-source configuration"
                )
            return self.last_source_boundary_decision

        if mp.count_processors() <= 1:
            if self.source_boundary_mode == "finite":
                source_sizes, source_amplitudes, changed_axes = (
                    self._regularize_adjoint_sources(
                        base_sizes,
                        base_amplitudes,
                        tuple(
                            ("x", "y", "z")[: _simulation_dimensions(simulation_spec)]
                            for _ in positions
                        ),
                        dimensions=_simulation_dimensions(simulation_spec),
                        resolution=float(simulation_spec.resolution),
                    )
                )
                method = "finite"
                reason = "finite adjoint sources selected by policy"
            else:
                source_sizes = base_sizes
                source_amplitudes = base_amplitudes
                changed_axes = tuple(() for _ in positions)
                method = "serial"
                reason = "single process run does not require source-boundary handling"
            decision = AdaptiveSourceBoundaryDecision(
                method=method,
                reason=reason,
                chunk_layout=simulation_spec.chunk_layout,
                source_sizes=source_sizes,
                source_amplitudes=source_amplitudes,
                changed_axes=changed_axes,
            )
            self._cache_source_boundary_decision(
                simulation_spec,
                signature,
                decision,
                positions,
                indexed_source_mask,
                forward_source_volumes,
            )
            return decision
        if simulation_spec.chunk_layout is not None:
            raise ValueError(
                "Meep-native adaptive balancing requires "
                "SimulationSpec.chunk_layout=None"
            )

        probe = simulation_spec.make(())
        try:
            probe.init_sim()
            native_layout = _simulation_partition(probe)
            if not isinstance(native_layout, mp.BinaryPartition):
                raise RuntimeError(
                    "initialized Meep simulation did not expose a BinaryPartition"
                )
            native_layout = copy.deepcopy(native_layout)
            _validate_partition_owners(
                native_layout,
                probe.structure.get_chunk_owners(),
            )
            resolution = float(simulation_spec.resolution)
            low, high = _cell_bounds(
                simulation_spec.cell_size,
                simulation_spec.geometry_center,
                resolution,
                cylindrical=cylindrical,
            )
            protected = _point_bounds(self.protected_points)
            permanent_sources = _source_bounds(
                (*self.source_volumes, *forward_source_volumes)
            )
            layout_indices = tuple(
                index
                for index, indexed in enumerate(indexed_source_mask)
                if not indexed
            )
            layout_positions = tuple(positions[index] for index in layout_indices)

            def project(source_sizes, initial_layout=None):
                layout = copy.deepcopy(
                    native_layout if initial_layout is None else initial_layout
                )
                previous = copy.deepcopy(layout)
                layout_sizes = (
                    ()
                    if not source_sizes
                    else tuple(source_sizes[index] for index in layout_indices)
                )
                adjoint_volumes = _source_volumes(layout_positions, layout_sizes)
                sources = _combine_bounds(
                    permanent_sources,
                    _source_bounds(adjoint_volumes),
                )
                self._constrain_partition(
                    layout,
                    previous,
                    low=low,
                    high=high,
                    resolution=resolution,
                    protected=protected,
                    sources=sources,
                    active_axes=(0, 2) if cylindrical else (0, 1, 2),
                )
                return layout

            if self.source_boundary_mode == "finite":
                finite_axes = tuple(
                    _axis_name(axis).lower() for axis in _partition_axes(native_layout)
                )
                source_sizes, source_amplitudes, changed_axes = (
                    self._regularize_adjoint_sources(
                        base_sizes,
                        base_amplitudes,
                        tuple(finite_axes for _ in positions),
                        dimensions=_simulation_dimensions(simulation_spec),
                        resolution=resolution,
                    )
                )
                layout = project(source_sizes)
                method = "finite"
                reason = "finite adjoint sources selected by policy"
            else:
                try:
                    layout = project(base_sizes)
                    source_sizes = base_sizes
                    source_amplitudes = base_amplitudes
                    changed_axes = tuple(() for _ in positions)
                    method = "layout"
                    reason = (
                        "indexed adjoint sources require no " "boundary regularization"
                        if all(indexed_source_mask)
                        else (
                            "Meep-native chunk boundaries avoid non-indexed "
                            "adjoint sources"
                        )
                    )
                except ValueError as source_error:
                    if cylindrical:
                        raise ValueError(
                            "Meep-native cylindrical chunk topology cannot "
                            "satisfy permanent boundary constraints"
                        ) from source_error
                    if self.source_boundary_mode == "layout":
                        if all(indexed_source_mask):
                            raise ValueError(
                                "Meep-native chunk topology cannot satisfy "
                                "permanent boundary constraints"
                            ) from source_error
                        raise ValueError(
                            "Meep-native chunk topology cannot place all adjoint "
                            "sources safely"
                        ) from source_error
                    try:
                        permanent_layout = project(())
                    except ValueError as permanent_error:
                        raise ValueError(
                            "Meep-native chunk topology cannot satisfy structural "
                            "or permanent source-boundary constraints"
                        ) from permanent_error
                    layout_conflict_axes = self._conflicting_source_axes(
                        permanent_layout,
                        low=low,
                        high=high,
                        resolution=resolution,
                        sources=_source_bounds(
                            _source_volumes(
                                layout_positions,
                                tuple(base_sizes[index] for index in layout_indices),
                            )
                        ),
                    )
                    layout_conflicts = iter(layout_conflict_axes)
                    conflict_axes = tuple(
                        () if indexed else next(layout_conflicts)
                        for indexed in indexed_source_mask
                    )
                    if not any(conflict_axes):
                        raise ValueError(
                            "Meep-native chunk topology cannot satisfy source "
                            "constraints with finite-source fallback"
                        ) from source_error
                    source_sizes, source_amplitudes, changed_axes = (
                        self._regularize_adjoint_sources(
                            base_sizes,
                            base_amplitudes,
                            conflict_axes,
                            dimensions=_simulation_dimensions(simulation_spec),
                            resolution=resolution,
                        )
                    )
                    try:
                        layout = project(source_sizes, permanent_layout)
                    except ValueError as finite_error:
                        raise ValueError(
                            "finite adjoint sources cannot satisfy the "
                            "Meep-native chunk topology"
                        ) from finite_error
                    method = "finite"
                    reason = (
                        "Meep-native layout required finite adjoint-source "
                        "regularization"
                    )
        finally:
            probe.reset_meep()

        simulation_spec.chunk_layout = layout
        decision = AdaptiveSourceBoundaryDecision(
            method=method,
            reason=reason,
            chunk_layout=layout,
            source_sizes=tuple(source_sizes),
            source_amplitudes=tuple(source_amplitudes),
            changed_axes=tuple(changed_axes),
        )
        self._cache_source_boundary_decision(
            simulation_spec,
            signature,
            decision,
            positions,
            indexed_source_mask,
            forward_source_volumes,
        )
        return decision

    def _cache_source_boundary_decision(
        self,
        simulation_spec,
        signature,
        decision: AdaptiveSourceBoundaryDecision,
        positions: Sequence[mp.Vector3],
        indexed_source_mask: Sequence[bool],
        forward_source_volumes: Sequence[mp.Volume],
    ) -> None:
        self._prepared_simulation_spec = simulation_spec
        self._prepared_source_signature = signature
        self.last_source_boundary_decision = decision
        self._effective_forward_source_volumes = tuple(forward_source_volumes)
        self._effective_source_volumes = _source_volumes(
            tuple(
                position
                for position, indexed in zip(positions, indexed_source_mask)
                if not indexed
            ),
            tuple(
                size
                for size, indexed in zip(
                    decision.source_sizes,
                    indexed_source_mask,
                )
                if not indexed
            ),
        )

    @staticmethod
    def _gather_rank_times(local_seconds: float) -> np.ndarray:
        valid = bool(np.isfinite(local_seconds) and local_seconds >= 0)
        nproc = mp.count_processors()
        if nproc <= 1:
            if not valid:
                raise ValueError("local callback timing must be finite and nonnegative")
            return np.asarray([local_seconds], dtype=float)
        local_times = np.zeros((nproc, 1), dtype=np.complex128)
        local_times[mp.my_rank(), 0] = complex(local_seconds, 0.0) if valid else 1j
        group_times = native_sampler.reduce_complex_grid_sum(local_times)
        if np.any(np.asarray(group_times[:, 0]).imag != 0.0):
            raise ValueError(
                "local callback timing must be finite and nonnegative on every rank"
            )
        return np.asarray(group_times[:, 0].real, dtype=float)

    @staticmethod
    def _gather_optional_rank_times(
        local_seconds: Optional[float],
    ) -> Optional[np.ndarray]:
        nproc = mp.count_processors()
        present = local_seconds is not None
        valid = not present or bool(np.isfinite(local_seconds) and local_seconds >= 0)
        if nproc <= 1:
            if not valid:
                raise ValueError("local wall timing must be finite and nonnegative")
            return (
                None
                if local_seconds is None
                else np.asarray([local_seconds], dtype=float)
            )

        local_times = np.zeros((nproc, 1), dtype=np.complex128)
        state = 1.0 if present and valid else (-1.0 if not valid else 0.0)
        local_times[mp.my_rank(), 0] = complex(
            0.0 if local_seconds is None or not valid else local_seconds,
            state,
        )
        group_times = native_sampler.reduce_complex_grid_sum(local_times)
        states = np.asarray(group_times[:, 0]).imag
        if np.any(states < 0.0):
            raise ValueError(
                "local wall timing must be finite and nonnegative on every rank"
            )
        supplied = states > 0.0
        if np.any(supplied) and not np.all(supplied):
            raise ValueError(
                "local_wall_seconds must be supplied on every rank or omitted on every rank"
            )
        if not np.any(supplied):
            return None
        return np.asarray(group_times[:, 0].real, dtype=float)

    def _regularize_adjoint_sources(
        self,
        base_sizes: Sequence[mp.Vector3],
        base_amplitudes: Sequence[complex],
        source_axes: Sequence[Sequence[str]],
        *,
        dimensions: int,
        resolution: float,
    ):
        sizes = []
        amplitudes = []
        changed_axes = []
        for base_size, base_amplitude, axes in zip(
            base_sizes,
            base_amplitudes,
            source_axes,
        ):
            size, amplitude, changed = regularize_source_size_and_amplitude(
                base_size,
                base_amplitude,
                dimensions=dimensions,
                resolution=resolution,
                width_cells=self.finite_source_width_cells,
                axes=axes,
            )
            sizes.append(size)
            amplitudes.append(complex(amplitude))
            changed_axes.append(changed)
        return tuple(sizes), tuple(amplitudes), tuple(changed_axes)

    def _conflicting_source_axes(
        self,
        partition: mp.BinaryPartition,
        *,
        low: np.ndarray,
        high: np.ndarray,
        resolution: float,
        sources: np.ndarray,
    ) -> Tuple[Tuple[str, ...], ...]:
        conflicts = [[] for _ in range(sources.shape[0])]
        tolerance = 1e-12
        protected_gap = max(self.protected_gap_cells / resolution, tolerance)

        def collect(node, node_low, node_high):
            if node.proc_id is not None:
                return
            axis_index = _axis_index(node.split_dir)
            source_extent = sources[:, 1, axis_index] - sources[:, 0, axis_index]
            axis_sensitive = np.isclose(source_extent, 0.0)
            crosses_region = np.ones(sources.shape[0], dtype=bool)
            for other_axis in range(3):
                if other_axis == axis_index:
                    continue
                crosses_region &= (
                    sources[:, 1, other_axis] >= node_low[other_axis] - tolerance
                ) & (sources[:, 0, other_axis] <= node_high[other_axis] + tolerance)
            blocked = (
                axis_sensitive
                & crosses_region
                & (node.split_pos > sources[:, 0, axis_index] - protected_gap)
                & (node.split_pos < sources[:, 1, axis_index] + protected_gap)
            )
            axis_name = _axis_name(node.split_dir).lower()
            for source_index in np.flatnonzero(blocked):
                if axis_name not in conflicts[source_index]:
                    conflicts[source_index].append(axis_name)

            left_high = node_high.copy()
            left_high[axis_index] = node.split_pos
            right_low = node_low.copy()
            right_low[axis_index] = node.split_pos
            collect(node.left, node_low, left_high)
            collect(node.right, right_low, node_high)

        collect(partition, low, high)
        return tuple(tuple(axes) for axes in conflicts)

    def _constrain_partition(
        self,
        candidate: mp.BinaryPartition,
        previous: mp.BinaryPartition,
        *,
        low: np.ndarray,
        high: np.ndarray,
        resolution: float,
        protected: np.ndarray,
        sources: np.ndarray,
        active_axes: Tuple[int, ...] = (0, 1, 2),
    ) -> None:
        if candidate.proc_id is not None:
            return
        if candidate.split_dir != previous.split_dir:
            raise ValueError("adaptive balancing cannot change partition topology")

        axis = candidate.split_dir
        axis_index = _axis_index(axis)
        n_axis_cells = int(np.rint((high[axis_index] - low[axis_index]) * resolution))
        left_required = _minimum_axis_cells(candidate.left, axis, self.min_chunk_cells)
        right_required = _minimum_axis_cells(
            candidate.right, axis, self.min_chunk_cells
        )
        tolerance = 1e-12
        protected_gap = max(self.protected_gap_cells / resolution, tolerance)
        active_protected = protected
        if sources.size:
            source_extent = sources[:, 1, axis_index] - sources[:, 0, axis_index]
            axis_sensitive_sources = sources[np.isclose(source_extent, 0.0)]
            if axis_sensitive_sources.size:
                active_protected = np.concatenate(
                    [active_protected, axis_sensitive_sources],
                    axis=0,
                )
        valid_positions = []
        for offset_cells in range(left_required, n_axis_cells - right_required + 1):
            position = low[axis_index] + offset_cells / resolution
            if active_protected.size:
                other_axes = [index for index in active_axes if index != axis_index]
                crosses_region = np.ones(active_protected.shape[0], dtype=bool)
                for other_axis in other_axes:
                    crosses_region &= (
                        active_protected[:, 1, other_axis]
                        >= low[other_axis] - tolerance
                    ) & (
                        active_protected[:, 0, other_axis]
                        <= high[other_axis] + tolerance
                    )
                if np.any(
                    crosses_region
                    & (position > active_protected[:, 0, axis_index] - protected_gap)
                    & (position < active_protected[:, 1, axis_index] + protected_gap)
                ):
                    continue
            valid_positions.append(position)
        if not valid_positions:
            raise ValueError(
                "no safe grid-aligned position remains for a chunk boundary"
            )

        candidate.split_pos = min(
            valid_positions,
            key=lambda position: (
                abs(position - candidate.split_pos),
                abs(position - previous.split_pos),
                position,
            ),
        )
        left_high = high.copy()
        left_high[axis_index] = candidate.split_pos
        right_low = low.copy()
        right_low[axis_index] = candidate.split_pos
        self._constrain_partition(
            candidate.left,
            previous.left,
            low=low,
            high=left_high,
            resolution=resolution,
            protected=protected,
            sources=sources,
            active_axes=active_axes,
        )
        self._constrain_partition(
            candidate.right,
            previous.right,
            low=right_low,
            high=high,
            resolution=resolution,
            protected=protected,
            sources=sources,
            active_axes=active_axes,
        )

    def rebalance(
        self,
        sim: mp.Simulation,
        *,
        local_extra_work_seconds: float,
        local_wall_seconds: Optional[float] = None,
        previous_measurements: Sequence[MeepTimingMeasurements] = (),
    ) -> Optional[mp.BinaryPartition]:
        """Return a safe workload-balanced layout for the next iteration.

        Args:
            sim: Completed Meep simulation whose current layout and timing are
                used as the baseline.
            local_extra_work_seconds: Rank-local time spent in TAMA work
                outside Meep's own timing measurements.
            local_wall_seconds: Optional rank-local end-to-end wall time used
                for the reported critical-path time.
            previous_measurements: Optional timing snapshots from earlier
                simulations in the same objective evaluation.

        Returns:
            Current or updated Meep binary partition. Serial execution may
            return `None` when no partition exists.

        Notes:
            The decision summary is stored in `last_report`.
        """
        current_layout = _simulation_partition(sim)
        if mp.count_processors() <= 1:
            extra_work = self._gather_rank_times(local_extra_work_seconds)
            wall_times = self._gather_optional_rank_times(local_wall_seconds)
            self.last_report = AdaptiveChunkBalanceReport(
                rank_work_seconds=(float(extra_work[0]),),
                max_to_mean=1.0,
                moved_boundaries=0,
                applied=False,
                reason="serial run",
                critical_path_seconds=float(
                    extra_work[0] if wall_times is None else wall_times[0]
                ),
                topology=(
                    _topology_label(current_layout)
                    if isinstance(current_layout, mp.BinaryPartition)
                    else "serial"
                ),
                next_topology=(
                    _topology_label(current_layout)
                    if isinstance(current_layout, mp.BinaryPartition)
                    else "serial"
                ),
            )
            return current_layout
        if not isinstance(current_layout, mp.BinaryPartition):
            raise ValueError(
                "adaptive balancing requires a BinaryPartition chunk layout"
            )

        extra_work = self._gather_rank_times(local_extra_work_seconds)
        wall_times = self._gather_optional_rank_times(local_wall_seconds)
        measurements = [*previous_measurements, self.capture_timing(sim)]
        combined = _combine_timing_measurements(measurements)
        combined.measurements["time_stepping"] = (
            np.asarray(combined.measurements["time_stepping"], dtype=float) + extra_work
        ).tolist()
        rank_work = self._meep_balancer._compute_working_times_per_process(combined)
        mean_work = float(np.mean(rank_work))
        imbalance = float(np.max(rank_work) / mean_work) if mean_work > 0 else 1.0
        topology_score = (
            float(np.max(rank_work))
            if wall_times is None
            else float(np.max(wall_times))
        )

        topology_label = _topology_label(current_layout)
        if self._frozen_layout is not None:
            selected = copy.deepcopy(self._frozen_layout)
            moved = sum(
                not np.isclose(old_position, new_position)
                for old_position, new_position in zip(
                    _split_positions(current_layout),
                    _split_positions(selected),
                )
            )
            self.last_report = AdaptiveChunkBalanceReport(
                rank_work_seconds=tuple(float(value) for value in rank_work),
                max_to_mean=imbalance,
                moved_boundaries=moved,
                applied=moved > 0,
                reason="best observed chunk layout is frozen",
                critical_path_seconds=topology_score,
                topology=topology_label,
                next_topology=_topology_label(selected),
                exploring=False,
            )
            return selected

        rank_score = float(np.max(rank_work))
        previous_best_score = self._best_rank_work_seconds
        regressed = bool(
            self._best_layout is not None
            and rank_score > self._best_rank_work_seconds
            and rank_score
            >= self._best_rank_work_seconds * (1.0 + self._MAX_SCORE_REGRESSION)
        )
        improved_best = rank_score < previous_best_score
        if improved_best:
            self._best_rank_work_seconds = rank_score
            self._best_layout = copy.deepcopy(current_layout)

        self._rebalance_observations += 1
        maximum_observed = (
            self._rebalance_observations >= self._MAX_REBALANCE_OBSERVATIONS
        )

        if regressed and not maximum_observed:
            self._balanced_observation_streak = 0
            self._stalled_best_score_streak = 0
            selected = copy.deepcopy(self._best_layout)
            moved = sum(
                not np.isclose(old_position, new_position)
                for old_position, new_position in zip(
                    _split_positions(current_layout),
                    _split_positions(selected),
                )
            )
            self.last_report = AdaptiveChunkBalanceReport(
                rank_work_seconds=tuple(float(value) for value in rank_work),
                max_to_mean=imbalance,
                moved_boundaries=moved,
                applied=moved > 0,
                reason="observed rank work regressed; restored best chunk layout",
                critical_path_seconds=topology_score,
                topology=topology_label,
                next_topology=_topology_label(selected),
                exploring=True,
            )
            return selected

        if imbalance <= self.imbalance_threshold:
            self._balanced_observation_streak += 1
        else:
            self._balanced_observation_streak = 0

        if np.isfinite(previous_best_score):
            relative_improvement = (
                (previous_best_score - rank_score) / previous_best_score
                if improved_best and previous_best_score > 0.0
                else 0.0
            )
            if relative_improvement < self._MIN_BEST_SCORE_IMPROVEMENT:
                self._stalled_best_score_streak += 1
            else:
                self._stalled_best_score_streak = 0

        minimum_observed = (
            self._rebalance_observations >= self._MIN_REBALANCE_OBSERVATIONS
        )
        balanced_converged = bool(
            minimum_observed
            and self._balanced_observation_streak >= self._REQUIRED_CONVERGENCE_STREAK
        )
        score_converged = bool(
            minimum_observed
            and self._stalled_best_score_streak >= self._REQUIRED_CONVERGENCE_STREAK
        )
        freeze_reason = None
        if maximum_observed:
            freeze_reason = (
                "maximum calibration observations reached; best layout is frozen"
            )
        elif balanced_converged:
            freeze_reason = "rank workload converged within the imbalance threshold"
        elif score_converged:
            freeze_reason = "critical-rank work improvement converged below one percent"

        if freeze_reason is not None:
            self._frozen_layout = copy.deepcopy(self._best_layout)
            selected = copy.deepcopy(self._frozen_layout)
            moved = sum(
                not np.isclose(old_position, new_position)
                for old_position, new_position in zip(
                    _split_positions(current_layout),
                    _split_positions(selected),
                )
            )
            self.last_report = AdaptiveChunkBalanceReport(
                rank_work_seconds=tuple(float(value) for value in rank_work),
                max_to_mean=imbalance,
                moved_boundaries=moved,
                applied=moved > 0,
                reason=freeze_reason,
                critical_path_seconds=topology_score,
                topology=topology_label,
                next_topology=_topology_label(selected),
                exploring=False,
            )
            return selected

        if imbalance <= self.imbalance_threshold:
            self.last_report = AdaptiveChunkBalanceReport(
                rank_work_seconds=tuple(float(value) for value in rank_work),
                max_to_mean=imbalance,
                moved_boundaries=0,
                applied=False,
                reason="rank workload is within the imbalance threshold",
                critical_path_seconds=topology_score,
                topology=topology_label,
                next_topology=topology_label,
                exploring=True,
            )
            return current_layout

        chunk_volumes = sim.structure.get_chunk_volumes()
        chunk_owners = sim.structure.get_chunk_owners()
        _validate_partition_owners(current_layout, chunk_owners)
        cylindrical = _is_cylindrical(sim)
        low, high = _cell_bounds(
            sim.cell_size,
            sim.geometry_center,
            float(sim.resolution),
            cylindrical=cylindrical,
        )
        if cylindrical:
            candidate = _rebalance_cylindrical_partition(
                combined,
                current_layout,
                chunk_volumes,
                chunk_owners,
                low=low,
                high=high,
                sensitivity=self.sensitivity,
                meep_balancer=self._meep_balancer,
            )
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                try:
                    candidate = self._meep_balancer.compute_new_chunk_layout(
                        combined,
                        current_layout,
                        chunk_volumes,
                        chunk_owners,
                        sensitivity=self.sensitivity,
                    )
                except ZeroDivisionError:
                    candidate = copy.deepcopy(current_layout)
            _restore_nonfinite_split_positions(candidate, current_layout)
        protected = _point_bounds(self.protected_points)
        sources = _source_bounds(
            (
                *self.source_volumes,
                *self._effective_forward_source_volumes,
                *self._effective_source_volumes,
            )
        )
        previous_positions = _split_positions(current_layout)
        try:
            self._constrain_partition(
                candidate,
                current_layout,
                low=low,
                high=high,
                resolution=float(sim.resolution),
                protected=protected,
                sources=sources,
                active_axes=(0, 2) if cylindrical else (0, 1, 2),
            )
        except ValueError as exc:
            self.last_report = AdaptiveChunkBalanceReport(
                rank_work_seconds=tuple(float(value) for value in rank_work),
                max_to_mean=imbalance,
                moved_boundaries=0,
                applied=False,
                reason=str(exc),
                critical_path_seconds=topology_score,
                topology=topology_label,
                next_topology=topology_label,
                exploring=True,
            )
            return current_layout

        candidate_positions = _split_positions(candidate)
        moved = sum(
            not np.isclose(old_position, new_position)
            for old_position, new_position in zip(
                previous_positions, candidate_positions
            )
        )
        applied = moved > 0
        self.last_report = AdaptiveChunkBalanceReport(
            rank_work_seconds=tuple(float(value) for value in rank_work),
            max_to_mean=imbalance,
            moved_boundaries=moved,
            applied=applied,
            reason=(
                "chunk boundaries updated"
                if applied
                else "safe projection kept the current layout"
            ),
            critical_path_seconds=topology_score,
            topology=topology_label,
            next_topology=topology_label,
            exploring=True,
        )
        return candidate if applied else current_layout
