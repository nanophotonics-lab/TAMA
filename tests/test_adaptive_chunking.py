from contextlib import contextmanager
import copy

import meep as mp
import numpy as np
import pytest
from meep.timing_measurements import MeepTimingMeasurements

import tama as tm
import tama.adaptive_chunking as adaptive_module


def _fake_group_sum(local_values):
    local_value = local_values[0, 0]
    return np.asarray(
        [[local_value], [4.0 * local_value]],
        dtype=np.complex128,
    )


class _FakeStructure:
    def get_chunk_volumes(self):
        return ()

    def get_chunk_owners(self):
        return np.asarray([0, 1])


class _FakeCylindricalVolume:
    def __init__(self, nr, nz):
        self.nr = nr
        self.nz = nz

    def num_direction(self, axis):
        if axis == mp.R:
            return self.nr
        if axis == mp.Z:
            return self.nz
        return 0


class _FakeCylindricalStructure:
    def get_chunk_volumes(self):
        return (
            _FakeCylindricalVolume(4, 8),
            _FakeCylindricalVolume(4, 8),
        )

    def get_chunk_owners(self):
        return np.asarray([0, 1])


class _FakeSimulation:
    def __init__(self, layout):
        self.chunk_layout = layout
        self.cell_size = mp.Vector3(8, 4)
        self.geometry_center = mp.Vector3()
        self.resolution = 2
        self.structure = _FakeStructure()


class _FakeCylindricalSimulation(_FakeSimulation):
    def __init__(self, layout):
        super().__init__(layout)
        self.cell_size = mp.Vector3(4, 0, 4)
        self.is_cylindrical = True
        self.structure = _FakeCylindricalStructure()


class _FakeMeepBalancer:
    def _compute_working_times_per_process(self, timing):
        return np.asarray(timing.measurements["time_stepping"], dtype=float)

    def _validate_sim(self, sim):
        pass

    def compute_new_chunk_layout(
        self,
        timing,
        current_layout,
        chunk_volumes,
        chunk_owners,
        sensitivity,
    ):
        candidate = copy.deepcopy(current_layout)
        candidate.split_pos = 1.3
        return candidate


class _NonfiniteMeepBalancer(_FakeMeepBalancer):
    def compute_new_chunk_layout(self, *args, **kwargs):
        candidate = copy.deepcopy(args[1])
        candidate.split_pos = np.nan
        return candidate


class _SequenceMeepBalancer(_FakeMeepBalancer):
    def __init__(self, split_positions):
        self.split_positions = iter(split_positions)
        self.compute_calls = 0

    def compute_new_chunk_layout(
        self,
        timing,
        current_layout,
        chunk_volumes,
        chunk_owners,
        sensitivity,
    ):
        self.compute_calls += 1
        candidate = copy.deepcopy(current_layout)
        candidate.split_pos = next(self.split_positions)
        return candidate


class _FakeProbeStructure:
    def __init__(self, layout):
        self.layout = layout
        self.get_binary_partition_calls = 0

    def get_binary_partition(self):
        self.get_binary_partition_calls += 1
        return copy.deepcopy(self.layout)

    def get_chunk_owners(self):
        return np.asarray([0, 1, 2, 3])


class _FakeProbeSimulation:
    def __init__(self, structure_layout):
        self.chunk_layout = mp.BinaryPartition(
            data=[
                (mp.Z, 0.0),
                [(mp.Z, -1.0), 0, 1],
                [(mp.Z, 1.0), 2, 3],
            ],
        )
        self.structure = _FakeProbeStructure(structure_layout)
        self.cell_size = mp.Vector3(8.0, 8.0, 8.0)
        self.geometry_center = mp.Vector3(-0.25, 0.25, -0.25)
        self.resolution = 1
        self.init_calls = 0
        self.reset_calls = 0

    def init_sim(self):
        self.init_calls += 1

    def reset_meep(self):
        self.reset_calls += 1


class _FakeSimulationSpec:
    def __init__(
        self,
        probe,
        *,
        cell_size=mp.Vector3(8.0, 8.0, 8.0),
        geometry_center=mp.Vector3(-0.25, 0.25, -0.25),
        chunk_layout=None,
    ):
        self.cell_size = cell_size
        self.geometry_center = geometry_center
        self.resolution = 1
        self.chunk_layout = chunk_layout
        self.probe = probe
        self.make_calls = []

    def make(self, sources=None):
        self.make_calls.append(sources)
        return self.probe


def _timing(rank_seconds):
    return MeepTimingMeasurements(
        measurements={"time_stepping": list(rank_seconds)},
        elapsed_time=float(max(rank_seconds)),
        num_time_steps=10,
        time_per_step=[],
        dft_relative_change=[],
        overlap_relative_change=[],
        relative_energy=[],
    )


def _axis_boundaries(length, center, resolution):
    axis_grid = mp.vol1d(length, resolution)
    axis_grid.center_origin()
    axis_grid.shift_origin(mp.vec(center))
    cell_count = axis_grid.num_direction(mp.Z)
    grid_low = axis_grid.surroundings().get_min_corner().z()
    return grid_low + np.arange(cell_count + 1, dtype=float) / resolution


def _leaf_ids(partition):
    if partition.proc_id is not None:
        return [int(partition.proc_id)]
    return [*_leaf_ids(partition.left), *_leaf_ids(partition.right)]


def _partition_signature(partition):
    if partition.proc_id is not None:
        return ("rank", int(partition.proc_id))
    return (
        int(partition.split_dir),
        _partition_signature(partition.left),
        _partition_signature(partition.right),
    )


def _vector_values(vector):
    return (float(vector.x), float(vector.y), float(vector.z))


class _DerivedSource(mp.Source):
    pass


@contextmanager
def _two_rank_meep():
    original_count_processors = mp.count_processors
    original_my_rank = mp.my_rank
    original_group_sum = adaptive_module.native_sampler.reduce_complex_grid_sum
    try:
        mp.count_processors = lambda: 2
        mp.my_rank = lambda: 0
        adaptive_module.native_sampler.reduce_complex_grid_sum = _fake_group_sum
        yield
    finally:
        mp.count_processors = original_count_processors
        mp.my_rank = original_my_rank
        adaptive_module.native_sampler.reduce_complex_grid_sum = original_group_sum


@contextmanager
def _processor_count(count):
    original_count_processors = mp.count_processors
    try:
        mp.count_processors = lambda: count
        yield
    finally:
        mp.count_processors = original_count_processors


def test_prepare_initial_layout_uses_and_safely_projects_meep_mixed_topology():
    meep_layout = mp.BinaryPartition(
        data=[
            (mp.X, 0.1),
            [(mp.Y, 0.1), 0, 1],
            [(mp.Y, 0.1), 2, 3],
        ]
    )
    probe = _FakeProbeSimulation(meep_layout)
    simulation = _FakeSimulationSpec(probe)
    protected_point = mp.Vector3(0.5, 0.0, 0.5)
    adjoint_source_position = mp.Vector3(-0.5, 0.5, -0.5)
    balancer = tm.AdaptiveAdjointChunkBalancer(
        protected_points=[protected_point],
        protected_gap_cells=1.0,
        min_chunk_cells=2,
    )

    with _processor_count(4):
        decision = balancer.prepare_initial_layout(
            simulation,
            adjoint_source_positions=(adjoint_source_position,),
            adjoint_source_size=mp.Vector3(),
            adjoint_source_amplitude=2.0,
        )

    assert isinstance(decision, tm.AdaptiveSourceBoundaryDecision)
    assert decision.method == "layout"
    assert decision.chunk_layout is simulation.chunk_layout
    assert decision.source_sizes == (mp.Vector3(),)
    assert decision.source_amplitudes == (2.0,)
    assert decision.changed_axes == ((),)
    layout = decision.chunk_layout
    assert layout is simulation.chunk_layout
    assert simulation.make_calls == [()]
    assert probe.init_calls == 1
    assert probe.reset_calls == 1
    assert probe.structure.get_binary_partition_calls == 1
    assert _partition_signature(layout) == _partition_signature(meep_layout)
    assert layout.split_dir == mp.X
    assert layout.left.split_dir == mp.Y
    assert layout.right.split_dir == mp.Y

    x_boundaries = _axis_boundaries(
        simulation.cell_size.x,
        simulation.geometry_center.x,
        simulation.resolution,
    )
    y_boundaries = _axis_boundaries(
        simulation.cell_size.y,
        simulation.geometry_center.y,
        simulation.resolution,
    )
    assert np.any(np.isclose(layout.split_pos, x_boundaries))
    assert np.any(np.isclose(layout.left.split_pos, y_boundaries))
    assert np.any(np.isclose(layout.right.split_pos, y_boundaries))
    assert (
        abs(layout.split_pos - protected_point.x)
        >= balancer.protected_gap_cells / simulation.resolution
    )
    point_child = (
        layout.left
        if protected_point.x <= layout.split_pos
        else layout.right
    )
    assert (
        abs(point_child.split_pos - protected_point.y)
        >= balancer.protected_gap_cells / simulation.resolution
    )
    assert (
        abs(layout.split_pos - adjoint_source_position.x)
        >= balancer.protected_gap_cells / simulation.resolution
    )
    source_child = (
        layout.left
        if adjoint_source_position.x <= layout.split_pos
        else layout.right
    )
    assert (
        abs(source_child.split_pos - adjoint_source_position.y)
        >= balancer.protected_gap_cells / simulation.resolution
    )
    assert sorted(_leaf_ids(layout)) == [0, 1, 2, 3]


def test_prepare_initial_layout_automatically_constrains_forward_source_subclasses():
    meep_layout = mp.BinaryPartition(
        data=[
            (mp.X, 0.0),
            [(mp.X, -1.0), 0, 1],
            [(mp.X, 1.0), 2, 3],
        ]
    )
    probe = _FakeProbeSimulation(meep_layout)
    simulation = _FakeSimulationSpec(
        probe,
        cell_size=mp.Vector3(4.0, 4.0),
        geometry_center=mp.Vector3(),
    )
    simulation.resolution = 2
    source_time = mp.ContinuousSource(frequency=1.0)
    forward_source = _DerivedSource(
        source_time,
        mp.Ez,
        center=mp.Vector3(),
    )
    indexed_source = mp.IndexedSource(
        source_time,
        object(),
        np.ones(1, dtype=np.complex128),
    )
    balancer = tm.AdaptiveAdjointChunkBalancer(
        protected_gap_cells=0.0,
        min_chunk_cells=1,
    )

    with _processor_count(4):
        decision = balancer.prepare_initial_layout(
            simulation,
            forward_sources=(indexed_source, forward_source),
        )

    assert decision.method == "layout"
    assert not np.isclose(decision.chunk_layout.split_pos, 0.0)
    assert len(balancer._effective_forward_source_volumes) == 1
    assert balancer._effective_forward_source_volumes[0].center == mp.Vector3()
    assert balancer._effective_forward_source_volumes[0].size == mp.Vector3()
    assert simulation.make_calls == [()]


@pytest.mark.parametrize(
    ("forward_sources", "error", "match"),
    [
        ((object(),), TypeError, "forward_sources"),
        (
            (
                mp.Source(
                    mp.ContinuousSource(frequency=1.0),
                    mp.Ez,
                    center=mp.Vector3(),
                    size=mp.Vector3(-1.0, 0.0, 0.0),
                ),
            ),
            ValueError,
            "nonnegative",
        ),
    ],
)
def test_prepare_initial_layout_validates_forward_sources(
    forward_sources,
    error,
    match,
):
    simulation = _FakeSimulationSpec(_FakeProbeSimulation(None))
    balancer = tm.AdaptiveAdjointChunkBalancer()

    with _processor_count(1), pytest.raises(error, match=match):
        balancer.prepare_initial_layout(
            simulation,
            forward_sources=forward_sources,
        )


def test_adaptive_balancer_only_calibrates_for_unfrozen_mpi_groups():
    balancer = tm.AdaptiveAdjointChunkBalancer()

    with _processor_count(1):
        assert not balancer.needs_rebalance
    with _processor_count(2):
        assert balancer.needs_rebalance

    balancer._frozen_layout = mp.BinaryPartition(data=0)
    with _processor_count(2):
        assert not balancer.needs_rebalance


def test_prepare_initial_layout_reuses_only_matching_forward_source_supports():
    simulation = _FakeSimulationSpec(_FakeProbeSimulation(None))
    source_time = mp.ContinuousSource(frequency=1.0)
    first_source = mp.Source(
        source_time,
        mp.Ez,
        center=mp.Vector3(-0.5, 0.0),
    )
    equivalent_source = _DerivedSource(
        source_time,
        mp.Hz,
        center=mp.Vector3(-0.5, 0.0),
    )
    changed_source = mp.Source(
        source_time,
        mp.Ez,
        center=mp.Vector3(0.5, 0.0),
    )
    balancer = tm.AdaptiveAdjointChunkBalancer()

    with _processor_count(1):
        first_decision = balancer.prepare_initial_layout(
            simulation,
            forward_sources=(first_source,),
        )
        repeated_decision = balancer.prepare_initial_layout(
            simulation,
            forward_sources=(equivalent_source,),
        )
        with pytest.raises(
            ValueError,
            match="different simulation or adjoint-source",
        ):
            balancer.prepare_initial_layout(
                simulation,
                forward_sources=(changed_source,),
            )

    assert repeated_decision is first_decision


def test_cylindrical_cell_bounds_match_meep_radial_grid():
    low, high = adaptive_module._cell_bounds(
        mp.Vector3(4.13, 0.0, 5.27),
        mp.Vector3(0.17, 0.0, -0.23),
        7.5,
        cylindrical=True,
    )

    assert np.allclose(low, [0.2, 0.0, -2.8666666666666667])
    assert np.allclose(high, [4.333333333333333, 0.0, 2.466666666666667])


def test_prepare_initial_layout_preserves_cylindrical_rz_topology():
    meep_layout = mp.BinaryPartition(
        data=[
            (mp.Z, 0.0),
            [(mp.R, 2.0), 0, 1],
            [(mp.R, 2.0), 2, 3],
        ]
    )
    probe = _FakeProbeSimulation(meep_layout)
    simulation = _FakeSimulationSpec(
        probe,
        cell_size=mp.Vector3(4.0, 0.0, 6.0),
        geometry_center=mp.Vector3(),
    )
    simulation.dimensions = mp.CYLINDRICAL
    simulation.is_cylindrical = True
    target_on_native_boundary = mp.Vector3(2.0, 0.0, 0.0)
    balancer = tm.AdaptiveAdjointChunkBalancer(min_chunk_cells=2)

    assert adaptive_module._simulation_dimensions(simulation) == 2

    with _processor_count(4):
        decision = balancer.prepare_initial_layout(
            simulation,
            adjoint_source_positions=(target_on_native_boundary,),
            adjoint_source_size=mp.Vector3(),
            adjoint_source_amplitude=1.5,
        )

    assert decision.method == "layout"
    assert "indexed adjoint sources" in decision.reason
    assert decision.source_sizes == (mp.Vector3(),)
    assert decision.source_amplitudes == (1.5,)
    assert decision.changed_axes == ((),)
    assert balancer._effective_source_volumes == ()
    assert _partition_signature(decision.chunk_layout) == _partition_signature(
        meep_layout
    )
    assert np.isclose(decision.chunk_layout.split_pos, 0.0)
    assert np.isclose(decision.chunk_layout.left.split_pos, 2.0)
    assert np.isclose(decision.chunk_layout.right.split_pos, 2.0)


def test_cylindrical_adaptive_balancing_rejects_finite_source_mode():
    probe = _FakeProbeSimulation(mp.BinaryPartition(data=[(mp.R, 2.0), 0, 1]))
    simulation = _FakeSimulationSpec(
        probe,
        cell_size=mp.Vector3(4.0, 0.0, 4.0),
        geometry_center=mp.Vector3(),
    )
    simulation.dimensions = mp.CYLINDRICAL
    simulation.is_cylindrical = True
    balancer = tm.AdaptiveAdjointChunkBalancer(source_boundary_mode="finite")

    with pytest.raises(ValueError, match="does not support finite"):
        balancer.prepare_initial_layout(
            simulation,
            adjoint_source_positions=(mp.Vector3(),),
            adjoint_source_size=mp.Vector3(),
            adjoint_source_amplitude=1.0,
        )


def test_prepare_initial_layout_uses_per_source_finite_fallback_without_changing_topology():
    meep_layout = mp.BinaryPartition(
        data=[
            (mp.X, 0.0),
            [(mp.Y, 0.0), 0, 1],
            [(mp.Z, 0.0), 2, 3],
        ]
    )
    probe = _FakeProbeSimulation(meep_layout)
    simulation = _FakeSimulationSpec(
        probe,
        cell_size=mp.Vector3(8.0, 8.0, 8.0),
        geometry_center=mp.Vector3(),
    )
    simulation.resolution = 2
    source_positions = (
        mp.Vector3(-3.5, 0.0, 0.0),
        mp.Vector3(3.5, 0.0, 0.0),
    )
    balancer = tm.AdaptiveAdjointChunkBalancer(
        protected_gap_cells=7.0,
        min_chunk_cells=2,
        finite_source_width_cells=1.0,
    )

    with _processor_count(4):
        decision = balancer.prepare_initial_layout(
            simulation,
            adjoint_source_positions=source_positions,
            adjoint_source_size=mp.Vector3(),
            adjoint_source_amplitude=2.0,
        )

    assert isinstance(decision, tm.AdaptiveSourceBoundaryDecision)
    assert decision.method == "finite"
    assert decision.chunk_layout is simulation.chunk_layout
    assert _partition_signature(decision.chunk_layout) == _partition_signature(meep_layout)
    assert tuple(_vector_values(size) for size in decision.source_sizes) == (
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
    )
    assert decision.source_amplitudes == (4.0, 4.0)
    assert decision.changed_axes == (("y",), ("z",))
    assert probe.reset_calls == 1


def test_prepare_initial_layout_treats_exact_split_as_conflict_with_zero_gap():
    meep_layout = mp.BinaryPartition(
        data=[
            (mp.X, 0.0),
            [(mp.X, -0.5), 0, 1],
            [(mp.X, 0.5), 2, 3],
        ]
    )
    probe = _FakeProbeSimulation(meep_layout)
    simulation = _FakeSimulationSpec(
        probe,
        cell_size=mp.Vector3(2.0, 2.0),
        geometry_center=mp.Vector3(),
    )
    simulation.resolution = 2
    balancer = tm.AdaptiveAdjointChunkBalancer(
        protected_gap_cells=0.0,
        min_chunk_cells=1,
    )

    with _processor_count(4):
        decision = balancer.prepare_initial_layout(
            simulation,
            adjoint_source_positions=(mp.Vector3(),),
            adjoint_source_size=mp.Vector3(),
            adjoint_source_amplitude=2.0,
        )

    assert decision.method == "finite"
    assert _partition_signature(decision.chunk_layout) == _partition_signature(
        meep_layout
    )
    assert _vector_values(decision.source_sizes[0]) == (0.5, 0.0, 0.0)
    assert decision.source_amplitudes == (4.0,)
    assert decision.changed_axes == (("x",),)


def test_prepare_initial_layout_excludes_indexed_target_from_boundary_constraints():
    meep_layout = mp.BinaryPartition(
        data=[
            (mp.X, 0.0),
            [(mp.X, -1.0), 0, 1],
            [(mp.X, 1.0), 2, 3],
        ]
    )
    probe = _FakeProbeSimulation(meep_layout)
    simulation = _FakeSimulationSpec(
        probe,
        cell_size=mp.Vector3(4.0, 4.0),
        geometry_center=mp.Vector3(),
    )
    simulation.resolution = 2
    balancer = tm.AdaptiveAdjointChunkBalancer(
        protected_points=(mp.Vector3(1.0, 0.25),),
        protected_gap_cells=0.0,
        min_chunk_cells=1,
    )

    with _processor_count(4):
        decision = balancer.prepare_initial_layout(
            simulation,
            adjoint_source_positions=(mp.Vector3(),),
            adjoint_source_size=mp.Vector3(),
            adjoint_source_amplitude=2.0,
            indexed_source_mask=(True,),
        )

    assert decision.method == "layout"
    assert _partition_signature(decision.chunk_layout) == _partition_signature(
        meep_layout
    )
    assert decision.source_sizes == (mp.Vector3(),)
    assert decision.source_amplitudes == (2.0,)
    assert decision.changed_axes == ((),)
    assert balancer._effective_source_volumes == ()
    assert np.isclose(decision.chunk_layout.split_pos, 0.0)
    assert not np.isclose(decision.chunk_layout.right.split_pos, 1.0)


def test_prepare_initial_layout_regularizes_only_nonindexed_mixed_target():
    meep_layout = mp.BinaryPartition(
        data=[
            (mp.X, 0.0),
            [(mp.Y, 0.0), 0, 1],
            [(mp.Z, 0.0), 2, 3],
        ]
    )
    probe = _FakeProbeSimulation(meep_layout)
    simulation = _FakeSimulationSpec(
        probe,
        cell_size=mp.Vector3(8.0, 8.0, 8.0),
        geometry_center=mp.Vector3(),
    )
    simulation.resolution = 2
    positions = (mp.Vector3(), mp.Vector3(3.5, 0.0, 0.0))
    balancer = tm.AdaptiveAdjointChunkBalancer(
        protected_gap_cells=7.0,
        min_chunk_cells=2,
        finite_source_width_cells=1.0,
    )

    with _processor_count(4):
        decision = balancer.prepare_initial_layout(
            simulation,
            adjoint_source_positions=positions,
            adjoint_source_sizes=(mp.Vector3(), mp.Vector3()),
            adjoint_source_amplitudes=(2.0, 3.0),
            indexed_source_mask=(True, False),
        )

    assert decision.method == "finite"
    assert tuple(_vector_values(size) for size in decision.source_sizes) == (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.5),
    )
    assert decision.source_amplitudes == (2.0, 6.0)
    assert decision.changed_axes == ((), ("z",))
    assert len(balancer._effective_source_volumes) == 1
    assert balancer._effective_source_volumes[0].center == positions[1]
    assert _vector_values(balancer._effective_source_volumes[0].size) == (
        0.0,
        0.0,
        0.5,
    )


def test_prepare_initial_layout_rejects_changed_indexed_source_configuration():
    simulation = _FakeSimulationSpec(_FakeProbeSimulation(None))
    balancer = tm.AdaptiveAdjointChunkBalancer()

    with _processor_count(1):
        balancer.prepare_initial_layout(
            simulation,
            adjoint_source_positions=(mp.Vector3(),),
            adjoint_source_size=mp.Vector3(),
            indexed_source_mask=(True,),
        )
        with pytest.raises(ValueError, match="different simulation or adjoint-source"):
            balancer.prepare_initial_layout(
                simulation,
                adjoint_source_positions=(mp.Vector3(),),
                adjoint_source_size=mp.Vector3(),
                indexed_source_mask=(False,),
            )


def test_prepare_initial_layout_rejects_permanently_blocked_mixed_topology():
    meep_layout = mp.BinaryPartition(
        data=[
            (mp.X, 0.0),
            [(mp.Y, 0.0), 0, 1],
            [(mp.Z, 0.0), 2, 3],
        ]
    )
    probe = _FakeProbeSimulation(meep_layout)
    simulation = _FakeSimulationSpec(
        probe,
        cell_size=mp.Vector3(8.0, 8.0, 8.0),
        geometry_center=mp.Vector3(),
    )
    simulation.resolution = 2
    balancer = tm.AdaptiveAdjointChunkBalancer(
        protected_points=(mp.Vector3(-3.5, 0.0, 0.0),),
        protected_gap_cells=7.0,
        min_chunk_cells=2,
    )

    with _processor_count(4):
        with pytest.raises(ValueError):
            balancer.prepare_initial_layout(
                simulation,
                adjoint_source_positions=(mp.Vector3(3.5, 3.5, 3.5),),
                adjoint_source_size=mp.Vector3(),
                adjoint_source_amplitude=1.0,
            )

    assert simulation.chunk_layout is None
    assert probe.reset_calls == 1


def test_prepare_initial_layout_rejects_structurally_infeasible_mixed_topology():
    meep_layout = mp.BinaryPartition(
        data=[
            (mp.X, 0.0),
            [(mp.Y, 0.0), 0, 1],
            [(mp.Y, 0.0), 2, 3],
        ]
    )
    probe = _FakeProbeSimulation(meep_layout)
    simulation = _FakeSimulationSpec(
        probe,
        cell_size=mp.Vector3(16.0, 2.0, 2.0),
        geometry_center=mp.Vector3(),
    )
    balancer = tm.AdaptiveAdjointChunkBalancer(
        protected_gap_cells=0.0,
        min_chunk_cells=2,
    )

    with _processor_count(4):
        with pytest.raises(ValueError):
            balancer.prepare_initial_layout(
                simulation,
                adjoint_source_positions=(mp.Vector3(),),
                adjoint_source_size=mp.Vector3(),
                adjoint_source_amplitude=1.0,
            )

    assert simulation.chunk_layout is None
    assert probe.reset_calls == 1


def test_prepare_initial_layout_rejects_existing_layout_and_skips_serial_probe():
    existing = mp.BinaryPartition(data=[(mp.X, 0.0), 0, 1])
    probe = _FakeProbeSimulation(existing)
    simulation = _FakeSimulationSpec(probe, chunk_layout=existing)
    balancer = tm.AdaptiveAdjointChunkBalancer()

    with _processor_count(2):
        with pytest.raises(ValueError, match="chunk_layout=None"):
            balancer.prepare_initial_layout(
                simulation,
                adjoint_source_positions=(mp.Vector3(),),
                adjoint_source_size=mp.Vector3(),
                adjoint_source_amplitude=3.0,
            )
    assert simulation.make_calls == []

    simulation.chunk_layout = None
    with _processor_count(1):
        serial_decision = balancer.prepare_initial_layout(
            simulation,
            adjoint_source_positions=(mp.Vector3(),),
            adjoint_source_size=mp.Vector3(),
            adjoint_source_amplitude=3.0,
        )
    assert serial_decision.chunk_layout is None
    assert serial_decision.source_sizes == (mp.Vector3(),)
    assert serial_decision.source_amplitudes == (3.0,)
    assert serial_decision.changed_axes == ((),)
    assert simulation.make_calls == []


def test_prepare_initial_layout_honors_explicit_finite_mode_in_serial():
    simulation = _FakeSimulationSpec(
        _FakeProbeSimulation(None),
        cell_size=mp.Vector3(4.0, 4.0),
    )
    simulation.resolution = 4
    balancer = tm.AdaptiveAdjointChunkBalancer(source_boundary_mode="finite")

    with _processor_count(1):
        decision = balancer.prepare_initial_layout(
            simulation,
            adjoint_source_positions=(mp.Vector3(),),
            adjoint_source_size=mp.Vector3(),
            adjoint_source_amplitude=2.0,
        )

    assert decision.method == "finite"
    assert _vector_values(decision.source_sizes[0]) == (0.25, 0.25, 0.0)
    assert decision.source_amplitudes == (32.0,)
    assert decision.changed_axes == (("x", "y"),)
    assert simulation.make_calls == []


def test_prepare_initial_layout_regularizes_each_adjoint_source_setting():
    simulation = _FakeSimulationSpec(
        _FakeProbeSimulation(None),
        cell_size=mp.Vector3(4.0, 4.0),
    )
    simulation.resolution = 4
    balancer = tm.AdaptiveAdjointChunkBalancer(source_boundary_mode="finite")

    with _processor_count(1):
        decision = balancer.prepare_initial_layout(
            simulation,
            adjoint_source_positions=(mp.Vector3(-0.5), mp.Vector3(0.5)),
            adjoint_source_sizes=(
                mp.Vector3(),
                mp.Vector3(0.5, 0.0),
            ),
            adjoint_source_amplitudes=(2.0 + 1.0j, 3.0 - 2.0j),
        )

    assert tuple(_vector_values(size) for size in decision.source_sizes) == (
        (0.25, 0.25, 0.0),
        (0.5, 0.25, 0.0),
    )
    assert decision.source_amplitudes == (
        32.0 + 16.0j,
        12.0 - 8.0j,
    )
    assert decision.changed_axes == (("x", "y"), ("y",))


def test_adaptive_balancer_combines_timing_and_projects_safe_grid_boundary():
    current_layout = mp.BinaryPartition(data=[(mp.X, -1.0), 0, 1])
    sim = _FakeSimulation(current_layout)
    balancer = tm.AdaptiveAdjointChunkBalancer(
        protected_points=[mp.Vector3(1.0, 0.0)],
        sensitivity=0.5,
        imbalance_threshold=1.01,
        protected_gap_cells=2.0,
        min_chunk_cells=4,
    )
    balancer._meep_balancer = _FakeMeepBalancer()
    balancer.capture_timing = lambda _sim: _timing([2.0, 2.0])

    with _two_rank_meep():
        next_layout = balancer.rebalance(
            sim,
            local_extra_work_seconds=0.1,
            previous_measurements=(_timing([1.0, 1.0]),),
        )

    assert next_layout is not current_layout
    assert np.isclose(next_layout.split_pos, 2.0)
    assert np.isclose(next_layout.split_pos * sim.resolution, 4.0)
    assert abs(next_layout.split_pos - 1.0) >= 1.0
    assert balancer.last_report.applied
    assert balancer.last_report.moved_boundaries == 1
    assert np.allclose(balancer.last_report.rank_work_seconds, [3.1, 3.4])


def test_adaptive_balancer_projects_to_shifted_rounded_meep_grid_boundary():
    current_layout = mp.BinaryPartition(data=[(mp.X, -0.25), 0, 1])
    sim = _FakeSimulation(current_layout)
    sim.cell_size = mp.Vector3(1.25, 4.0)
    sim.geometry_center = mp.Vector3(-0.25, 0.0)
    sim.resolution = 2
    balancer = tm.AdaptiveAdjointChunkBalancer(
        sensitivity=0.5,
        imbalance_threshold=1.01,
        protected_gap_cells=0.0,
        min_chunk_cells=1,
    )
    balancer._meep_balancer = _FakeMeepBalancer()
    balancer.capture_timing = lambda _sim: _timing([2.0, 2.0])

    with _two_rank_meep():
        next_layout = balancer.rebalance(
            sim,
            local_extra_work_seconds=0.1,
            previous_measurements=(_timing([1.0, 1.0]),),
        )

    assert next_layout is not current_layout
    assert np.isclose(next_layout.split_pos, 0.25)
    assert balancer.last_report.applied


def test_adaptive_balancer_updates_cylindrical_radial_boundary():
    current_layout = mp.BinaryPartition(data=[(mp.R, 2.0), 0, 1])
    sim = _FakeCylindricalSimulation(current_layout)
    balancer = tm.AdaptiveAdjointChunkBalancer(
        sensitivity=1.0,
        imbalance_threshold=1.01,
        protected_gap_cells=0.0,
        min_chunk_cells=2,
    )
    balancer._meep_balancer = _FakeMeepBalancer()
    balancer.capture_timing = lambda _sim: _timing([1.0, 1.0])

    with _two_rank_meep():
        next_layout = balancer.rebalance(
            sim,
            local_extra_work_seconds=1.0,
        )

    assert next_layout is not current_layout
    assert next_layout.split_dir == mp.R
    assert np.isclose(next_layout.split_pos, 3.0)
    assert np.isfinite(next_layout.split_pos)
    assert balancer.last_report.applied
    assert balancer.last_report.moved_boundaries == 1
    assert balancer.last_report.topology == "R"


def test_adaptive_balancer_keeps_layout_below_threshold():
    current_layout = mp.BinaryPartition(data=[(mp.X, 0.0), 0, 1])
    sim = _FakeSimulation(current_layout)
    balancer = tm.AdaptiveAdjointChunkBalancer(imbalance_threshold=1.2)
    balancer._meep_balancer = _FakeMeepBalancer()
    balancer.capture_timing = lambda _sim: _timing([1.0, 1.0])

    with _two_rank_meep():
        next_layout = balancer.rebalance(sim, local_extra_work_seconds=0.0)

    assert next_layout is current_layout
    assert not balancer.last_report.applied
    assert balancer.last_report.moved_boundaries == 0


def test_adaptive_balancer_evaluates_third_candidate_on_fourth_call():
    first_layout = mp.BinaryPartition(data=[(mp.X, -1.0), 0, 1])
    sim = _FakeSimulation(first_layout)
    balancer = tm.AdaptiveAdjointChunkBalancer(
        sensitivity=1.0,
        imbalance_threshold=1.01,
        protected_gap_cells=0.0,
        min_chunk_cells=1,
    )
    meep_balancer = _SequenceMeepBalancer((-0.5, 0.0, 0.5, 1.0))
    balancer._meep_balancer = meep_balancer
    timings = iter(
        (
            _timing([10.0, 1.0]),
            _timing([8.0, 1.0]),
            _timing([6.0, 1.0]),
            _timing([4.0, 1.0]),
        )
    )
    balancer.capture_timing = lambda _sim: next(timings)

    with _two_rank_meep():
        second_layout = balancer.rebalance(
            sim,
            local_extra_work_seconds=0.0,
        )
        sim.chunk_layout = second_layout
        third_layout = balancer.rebalance(
            sim,
            local_extra_work_seconds=0.0,
        )
        sim.chunk_layout = third_layout
        fourth_layout = balancer.rebalance(
            sim,
            local_extra_work_seconds=0.0,
        )
        assert not balancer.is_frozen
        assert balancer.last_report.exploring
        sim.chunk_layout = fourth_layout
        fifth_layout = balancer.rebalance(
            sim,
            local_extra_work_seconds=0.0,
        )

    assert np.isclose(second_layout.split_pos, -0.5)
    assert np.isclose(third_layout.split_pos, 0.0)
    assert np.isclose(fourth_layout.split_pos, 0.5)
    assert np.isclose(fifth_layout.split_pos, 1.0)
    assert np.isclose(balancer._best_layout.split_pos, 0.5)
    assert meep_balancer.compute_calls == 4


def test_adaptive_balancer_freezes_after_two_consecutive_balanced_observations():
    initial_layout = mp.BinaryPartition(data=[(mp.X, -1.0), 0, 1])
    sim = _FakeSimulation(initial_layout)
    balancer = tm.AdaptiveAdjointChunkBalancer(
        sensitivity=1.0,
        protected_gap_cells=0.0,
        min_chunk_cells=1,
    )
    meep_balancer = _SequenceMeepBalancer((0.0, 0.5))
    balancer._meep_balancer = meep_balancer
    timings = iter(
        (
            _timing([10.0, 1.0]),
            _timing([9.0, 8.8]),
            _timing([8.5, 1.0]),
            _timing([8.0, 7.8]),
            _timing([7.5, 7.4]),
        )
    )
    balancer.capture_timing = lambda _sim: next(timings)

    with _two_rank_meep():
        layout = balancer.rebalance(sim, local_extra_work_seconds=0.0)
        assert not balancer.is_frozen
        sim.chunk_layout = layout
        layout = balancer.rebalance(sim, local_extra_work_seconds=0.0)
        assert not balancer.is_frozen
        sim.chunk_layout = layout
        layout = balancer.rebalance(sim, local_extra_work_seconds=0.0)
        assert not balancer.is_frozen
        sim.chunk_layout = layout
        layout = balancer.rebalance(sim, local_extra_work_seconds=0.0)
        assert not balancer.is_frozen
        sim.chunk_layout = layout
        frozen_layout = balancer.rebalance(sim, local_extra_work_seconds=0.0)

    assert balancer.is_frozen
    assert not balancer.last_report.exploring
    assert np.isclose(frozen_layout.split_pos, 0.5)
    assert meep_balancer.compute_calls == 2
    assert "converged" in balancer.last_report.reason


def test_adaptive_balancer_freezes_after_two_subpercent_best_improvements():
    initial_layout = mp.BinaryPartition(data=[(mp.X, -1.0), 0, 1])
    sim = _FakeSimulation(initial_layout)
    balancer = tm.AdaptiveAdjointChunkBalancer(
        sensitivity=1.0,
        imbalance_threshold=1.01,
        protected_gap_cells=0.0,
        min_chunk_cells=1,
    )
    meep_balancer = _SequenceMeepBalancer((0.0, 0.5, 1.0, 1.5, 2.0))
    balancer._meep_balancer = meep_balancer
    timings = iter(
        (
            _timing([10.0, 1.0]),
            _timing([9.0, 1.0]),
            _timing([8.95, 1.0]),
            _timing([8.0, 1.0]),
            _timing([7.95, 1.0]),
            _timing([7.96, 1.0]),
        )
    )
    balancer.capture_timing = lambda _sim: next(timings)

    with _two_rank_meep():
        layout = balancer.rebalance(sim, local_extra_work_seconds=0.0)
        sim.chunk_layout = layout
        layout = balancer.rebalance(sim, local_extra_work_seconds=0.0)
        sim.chunk_layout = layout
        layout = balancer.rebalance(sim, local_extra_work_seconds=0.0)
        assert not balancer.is_frozen
        sim.chunk_layout = layout
        layout = balancer.rebalance(sim, local_extra_work_seconds=0.0)
        assert not balancer.is_frozen
        sim.chunk_layout = layout
        layout = balancer.rebalance(sim, local_extra_work_seconds=0.0)
        assert not balancer.is_frozen
        sim.chunk_layout = layout
        frozen_layout = balancer.rebalance(sim, local_extra_work_seconds=0.0)

    assert balancer.is_frozen
    assert not balancer.last_report.exploring
    assert np.isclose(frozen_layout.split_pos, 1.5)
    assert meep_balancer.compute_calls == 5
    assert "converged" in balancer.last_report.reason


def test_adaptive_balancer_freezes_after_repeated_no_move_proposals_stall_score():
    initial_layout = mp.BinaryPartition(data=[(mp.X, 0.0), 0, 1])
    sim = _FakeSimulation(initial_layout)
    balancer = tm.AdaptiveAdjointChunkBalancer(
        sensitivity=1.0,
        imbalance_threshold=1.01,
        protected_gap_cells=0.0,
        min_chunk_cells=1,
    )
    meep_balancer = _SequenceMeepBalancer((0.0, 0.0))
    balancer._meep_balancer = meep_balancer
    balancer.capture_timing = lambda _sim: _timing([10.0, 1.0])

    with _two_rank_meep():
        layout = balancer.rebalance(sim, local_extra_work_seconds=0.0)
        assert not balancer.is_frozen
        sim.chunk_layout = layout
        layout = balancer.rebalance(sim, local_extra_work_seconds=0.0)
        assert not balancer.is_frozen
        sim.chunk_layout = layout
        frozen_layout = balancer.rebalance(sim, local_extra_work_seconds=0.0)

    assert frozen_layout is not None
    assert np.isclose(frozen_layout.split_pos, 0.0)
    assert balancer.is_frozen
    assert not balancer.needs_rebalance
    assert not balancer.last_report.exploring
    assert meep_balancer.compute_calls == 2
    assert "converged" in balancer.last_report.reason


def test_adaptive_balancer_freezes_best_layout_after_eight_observations():
    initial_layout = mp.BinaryPartition(data=[(mp.X, -1.0), 0, 1])
    sim = _FakeSimulation(initial_layout)
    balancer = tm.AdaptiveAdjointChunkBalancer(
        sensitivity=1.0,
        imbalance_threshold=1.01,
        protected_gap_cells=0.0,
        min_chunk_cells=1,
    )
    meep_balancer = _SequenceMeepBalancer(
        (-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
    )
    balancer._meep_balancer = meep_balancer
    timings = iter(
        _timing([score, 1.0])
        for score in (100.0, 80.0, 64.0, 51.0, 40.0, 32.0, 25.0, 20.0)
    )
    balancer.capture_timing = lambda _sim: next(timings)

    with _two_rank_meep():
        layout = initial_layout
        for observation in range(8):
            sim.chunk_layout = layout
            layout = balancer.rebalance(sim, local_extra_work_seconds=0.0)
            if observation < 7:
                assert not balancer.is_frozen

    assert balancer.is_frozen
    assert not balancer.last_report.exploring
    assert np.isclose(layout.split_pos, 2.5)
    assert meep_balancer.compute_calls == 7
    assert "maximum" in balancer.last_report.reason


def test_adaptive_balancer_selects_best_by_rank_work_not_wall_time():
    initial_layout = mp.BinaryPartition(data=[(mp.X, -1.0), 0, 1])
    sim = _FakeSimulation(initial_layout)
    balancer = tm.AdaptiveAdjointChunkBalancer(
        sensitivity=1.0,
        imbalance_threshold=1.01,
        protected_gap_cells=0.0,
        min_chunk_cells=1,
    )
    balancer._meep_balancer = _SequenceMeepBalancer((0.0, 0.5))
    timings = iter((_timing([10.0, 1.0]), _timing([8.0, 1.0])))
    balancer.capture_timing = lambda _sim: next(timings)

    with _two_rank_meep():
        layout = balancer.rebalance(
            sim,
            local_extra_work_seconds=0.0,
            local_wall_seconds=1.0,
        )
        sim.chunk_layout = layout
        balancer.rebalance(
            sim,
            local_extra_work_seconds=0.0,
            local_wall_seconds=100.0,
        )

    assert np.isclose(balancer._best_layout.split_pos, 0.0)
    assert np.isclose(balancer._best_rank_work_seconds, 8.0)
    assert np.isclose(balancer.last_report.critical_path_seconds, 400.0)


def test_adaptive_balancer_rolls_back_two_percent_regression_immediately():
    initial_layout = mp.BinaryPartition(data=[(mp.X, -1.0), 0, 1])
    sim = _FakeSimulation(initial_layout)
    balancer = tm.AdaptiveAdjointChunkBalancer(
        sensitivity=1.0,
        imbalance_threshold=1.01,
        protected_gap_cells=0.0,
        min_chunk_cells=1,
    )
    meep_balancer = _SequenceMeepBalancer((0.0, 1.0))
    balancer._meep_balancer = meep_balancer
    timings = iter((_timing([5.0, 1.0]), _timing([5.10, 1.0])))
    balancer.capture_timing = lambda _sim: next(timings)

    with _two_rank_meep():
        explored_layout = balancer.rebalance(
            sim,
            local_extra_work_seconds=0.0,
        )
        sim.chunk_layout = explored_layout
        rollback_layout = balancer.rebalance(
            sim,
            local_extra_work_seconds=0.0,
        )

    assert np.isclose(explored_layout.split_pos, 0.0)
    assert np.isclose(rollback_layout.split_pos, -1.0)
    assert rollback_layout is not initial_layout
    assert meep_balancer.compute_calls == 1
    assert not balancer.is_frozen
    assert "regressed" in balancer.last_report.reason


def test_adaptive_balancer_rolls_back_regression_before_convergence_freeze():
    initial_layout = mp.BinaryPartition(data=[(mp.X, -1.0), 0, 1])
    sim = _FakeSimulation(initial_layout)
    balancer = tm.AdaptiveAdjointChunkBalancer(
        sensitivity=1.0,
        imbalance_threshold=1.01,
        protected_gap_cells=0.0,
        min_chunk_cells=1,
    )
    meep_balancer = _SequenceMeepBalancer((0.0, 0.5))
    balancer._meep_balancer = meep_balancer
    timings = iter(
        (
            _timing([10.0, 1.0]),
            _timing([9.95, 1.0]),
            _timing([10.20, 1.0]),
        )
    )
    balancer.capture_timing = lambda _sim: next(timings)

    with _two_rank_meep():
        layout = balancer.rebalance(sim, local_extra_work_seconds=0.0)
        sim.chunk_layout = layout
        layout = balancer.rebalance(sim, local_extra_work_seconds=0.0)
        sim.chunk_layout = layout
        rollback_layout = balancer.rebalance(sim, local_extra_work_seconds=0.0)

    assert np.isclose(rollback_layout.split_pos, 0.0)
    assert not balancer.is_frozen
    assert balancer.last_report.exploring
    assert "regressed" in balancer.last_report.reason


def test_rank_timing_validation_reports_remote_invalid_value(monkeypatch):
    monkeypatch.setattr(mp, "count_processors", lambda: 2)
    monkeypatch.setattr(mp, "my_rank", lambda: 0)
    monkeypatch.setattr(
        adaptive_module.native_sampler,
        "reduce_complex_grid_sum",
        lambda _values: np.asarray([[0.2], [1j]], dtype=np.complex128),
    )

    with pytest.raises(ValueError, match="on every rank"):
        tm.AdaptiveAdjointChunkBalancer._gather_rank_times(0.2)


def test_optional_wall_timing_requires_same_presence_on_every_rank(monkeypatch):
    monkeypatch.setattr(mp, "count_processors", lambda: 2)
    monkeypatch.setattr(mp, "my_rank", lambda: 0)
    monkeypatch.setattr(
        adaptive_module.native_sampler,
        "reduce_complex_grid_sum",
        lambda _values: np.asarray([[0.2 + 1j], [0.0]], dtype=np.complex128),
    )

    with pytest.raises(ValueError, match="supplied on every rank"):
        tm.AdaptiveAdjointChunkBalancer._gather_optional_rank_times(0.2)


def test_cartesian_balancer_restores_nonfinite_zero_work_boundary():
    current_layout = mp.BinaryPartition(data=[(mp.X, 0.0), 0, 1])
    sim = _FakeSimulation(current_layout)
    balancer = tm.AdaptiveAdjointChunkBalancer(
        imbalance_threshold=1.01,
        protected_gap_cells=0.0,
        min_chunk_cells=1,
    )
    balancer._meep_balancer = _NonfiniteMeepBalancer()
    balancer.capture_timing = lambda _sim: _timing([1.0, 1.0])

    with _two_rank_meep():
        next_layout = balancer.rebalance(
            sim,
            local_extra_work_seconds=0.1,
        )

    assert next_layout is current_layout
    assert np.isfinite(next_layout.split_pos)
    assert next_layout.split_pos == 0.0
    assert not balancer.last_report.applied


if __name__ == "__main__":
    test_prepare_initial_layout_uses_and_safely_projects_meep_mixed_topology()
    test_prepare_initial_layout_uses_per_source_finite_fallback_without_changing_topology()
    test_prepare_initial_layout_rejects_permanently_blocked_mixed_topology()
    test_prepare_initial_layout_rejects_structurally_infeasible_mixed_topology()
    test_prepare_initial_layout_rejects_existing_layout_and_skips_serial_probe()
    test_adaptive_balancer_combines_timing_and_projects_safe_grid_boundary()
    test_adaptive_balancer_projects_to_shifted_rounded_meep_grid_boundary()
    test_adaptive_balancer_keeps_layout_below_threshold()
