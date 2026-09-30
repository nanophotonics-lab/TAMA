import copy

import meep as mp
import numpy as np
import pytest

import tama as tm


def _axis_index(axis):
    if axis == mp.R:
        return 0
    return (mp.X, mp.Y, mp.Z).index(axis)


def _vector_values(vector):
    return (float(vector.x), float(vector.y), float(vector.z))


def _actual_grid_boundaries(cell_size, geometry_center, resolution):
    boundaries = []
    for length, center in zip(
        _vector_values(cell_size),
        _vector_values(geometry_center),
    ):
        if length <= 0:
            boundaries.append(np.asarray([center], dtype=float))
            continue
        axis_grid = mp.vol1d(length, resolution)
        axis_grid.center_origin()
        axis_grid.shift_origin(mp.vec(center))
        cell_count = axis_grid.num_direction(mp.Z)
        grid_low = axis_grid.surroundings().get_min_corner().z()
        boundaries.append(
            grid_low + np.arange(cell_count + 1, dtype=float) / resolution
        )
    return tuple(boundaries)


def _actual_cylindrical_grid_boundaries(
    cell_size,
    geometry_center,
    resolution,
):
    grid = mp.volcyl(cell_size.x, cell_size.z, resolution)
    grid.center_origin()
    grid.shift_origin(mp.veccyl(geometry_center.x, geometry_center.z))
    surroundings = grid.surroundings()
    low = surroundings.get_min_corner()
    nr = grid.num_direction(mp.R)
    nz = grid.num_direction(mp.Z)
    return (
        low.r() + np.arange(nr + 1, dtype=float) / resolution,
        np.asarray([geometry_center.y], dtype=float),
        low.z() + np.arange(nz + 1, dtype=float) / resolution,
    )


def _assert_grid_aligned_and_point_safe(
    partition,
    point,
    resolution,
    gap_cells,
    boundaries,
    min_chunk_cells=1,
    ranges=None,
    active_axes=(0, 1, 2),
):
    if ranges is None:
        ranges = tuple((0, values.size - 1) for values in boundaries)
    if partition.proc_id is not None:
        return [int(partition.proc_id)]

    axis_index = _axis_index(partition.split_dir)
    matches = np.flatnonzero(
        np.isclose(
            boundaries[axis_index],
            partition.split_pos,
            rtol=0.0,
            atol=1e-12,
        )
    )
    assert matches.size == 1
    split_index = int(matches[0])
    low_index, high_index = ranges[axis_index]
    assert split_index - low_index >= min_chunk_cells
    assert high_index - split_index >= min_chunk_cells

    point_values = _vector_values(point)
    crosses_region = all(
        boundaries[other_axis][ranges[other_axis][0]] - 1e-12
        <= point_values[other_axis]
        <= boundaries[other_axis][ranges[other_axis][1]] + 1e-12
        for other_axis in active_axes
        if other_axis != axis_index
    )
    if crosses_region:
        assert (
            abs(partition.split_pos - point_values[axis_index])
            >= gap_cells / resolution
        )

    left_ranges = list(ranges)
    left_ranges[axis_index] = (low_index, split_index)
    right_ranges = list(ranges)
    right_ranges[axis_index] = (split_index, high_index)
    return [
        *_assert_grid_aligned_and_point_safe(
            partition.left,
            point,
            resolution,
            gap_cells,
            boundaries,
            min_chunk_cells,
            tuple(left_ranges),
            active_axes,
        ),
        *_assert_grid_aligned_and_point_safe(
            partition.right,
            point,
            resolution,
            gap_cells,
            boundaries,
            min_chunk_cells,
            tuple(right_ranges),
            active_axes,
        ),
    ]


def _split_axes(partition):
    if partition.proc_id is not None:
        return set()
    return {
        partition.split_dir,
        *_split_axes(partition.left),
        *_split_axes(partition.right),
    }


def _topology_signature(partition):
    if partition.proc_id is not None:
        return ("rank", int(partition.proc_id))
    return (
        int(partition.split_dir),
        _topology_signature(partition.left),
        _topology_signature(partition.right),
    )


def _native_layout(simulation):
    probe = simulation.make(())
    try:
        probe.init_sim()
        return copy.deepcopy(probe.structure.get_binary_partition())
    finally:
        probe.reset_meep()


def _run_effective_source(simulation, decision, position, component):
    source = mp.Source(
        mp.GaussianSource(frequency=1.0, fwidth=0.5),
        component=component,
        center=position,
        size=decision.source_sizes[0],
        amplitude=decision.source_amplitudes[0],
    )
    sim = simulation.make([source])
    try:
        sim.run(until=0.1)
        assert _topology_signature(sim.chunk_layout) == _topology_signature(
            decision.chunk_layout
        )
    finally:
        sim.reset_meep()


def _make_cylindrical_objective(objective_kind, *, adaptive=True):
    resolution = 6
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=2.25)
    shape = (2, 2)
    design_center = mp.Vector3(0.9, 0.0, 0.0)
    design_size = mp.Vector3(0.4, 0.0, 0.4)
    material_grid = mp.MaterialGrid(
        mp.Vector3(shape[0], 1, shape[1]),
        air,
        dielectric,
    )
    design = tm.DesignGrid(
        material_grid=material_grid,
        center=design_center,
        size=design_size,
        shape=shape,
        background=air,
        design_material=dielectric,
        coordinate_system="cylindrical",
    )
    source_position = mp.Vector3(0.8, 0.0, -0.65)
    target = tm.PointTarget(
        position=mp.Vector3(0.8, 0.0, 0.65),
        component=mp.Er,
    )
    simulation = tm.SimulationSpec(
        cell_size=mp.Vector3(4.0, 0.0, 4.0),
        boundary_layers=[
            mp.PML(0.3, direction=mp.R),
            mp.PML(0.3, direction=mp.Z),
        ],
        geometry=[
            mp.Block(
                center=design_center,
                size=design_size,
                material=material_grid,
            )
        ],
        sources=[
            mp.Source(
                mp.GaussianSource(
                    frequency=0.8,
                    fwidth=0.6,
                    cutoff=3,
                ),
                component=mp.Er,
                center=source_position,
            )
        ],
        resolution=resolution,
        dimensions=mp.CYLINDRICAL,
        m=1,
        eps_averaging=False,
    )
    common = dict(
        design=design,
        simulation=simulation,
        t_final=4.0,
        dt=0.5 / resolution,
    )
    if not adaptive:
        common["chunk_balancer"] = None
    if objective_kind == "tda":
        objective = tm.TDAObjective(target=target, **common)
    else:
        objective = tm.MultiTDAObjective(
            targets=[target],
            wavelength_bands=[(0.9, 2.0)],
            weights=[1.0],
            kernel_length=7,
            pixel_chunk=2,
            **common,
        )
    return objective, shape


def _make_cartesian_3d_objective():
    resolution = 4
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=2.25)
    shape = (2, 2, 2)
    design_center = mp.Vector3()
    design_size = mp.Vector3(0.5, 0.5, 0.5)
    material_grid = mp.MaterialGrid(
        mp.Vector3(*shape),
        air,
        dielectric,
        do_averaging=False,
    )
    design = tm.DesignGrid(
        material_grid=material_grid,
        center=design_center,
        size=design_size,
        shape=shape,
        background=air,
        design_material=dielectric,
    )
    source_position = mp.Vector3(-0.75, -0.25, 0.125)
    simulation = tm.SimulationSpec(
        cell_size=mp.Vector3(3.0, 3.0, 3.0),
        geometry=[
            mp.Block(
                center=design_center,
                size=design_size,
                material=material_grid,
            )
        ],
        sources=[
            mp.Source(
                mp.GaussianSource(frequency=0.8, fwidth=0.6, cutoff=3),
                component=mp.Ex,
                center=source_position,
            )
        ],
        resolution=resolution,
        dimensions=3,
        eps_averaging=False,
    )
    objective = tm.TDAObjective(
        design=design,
        simulation=simulation,
        target=tm.PointTarget(
            position=mp.Vector3(0.75, 0.25, -0.125),
            component=mp.Ex,
        ),
        t_final=2.0,
        dt=0.5 / resolution,
    )
    return objective, shape


@pytest.mark.mpi2
def test_adaptive_balancer_updates_a_real_meep_partition():
    nproc = mp.count_processors()
    if nproc != 2:
        pytest.skip("requires exactly two MPI ranks")

    resolution = 8
    source_center = mp.Vector3(-2.3, 0.0)
    cell_size = mp.Vector3(8, 4)
    layout = mp.BinaryPartition(data=[(mp.X, 0.0), 0, 1])
    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        chunk_layout=layout,
        sources=[
            mp.Source(
                mp.GaussianSource(frequency=1.0, fwidth=0.5),
                component=mp.Ez,
                center=source_center,
            )
        ],
    )
    sim.run(until=0.5)

    balancer = tm.AdaptiveAdjointChunkBalancer(
        protected_points=[source_center],
        sensitivity=0.5,
        imbalance_threshold=1.001,
        protected_gap_cells=2.0,
        min_chunk_cells=4,
    )
    next_layout = balancer.rebalance(
        sim,
        local_extra_work_seconds=2.0 * mp.my_rank() / (nproc - 1),
    )

    assert isinstance(next_layout, mp.BinaryPartition)
    assert balancer.last_report is not None
    assert balancer.last_report.applied
    assert balancer.last_report.moved_boundaries > 0
    boundaries = _actual_grid_boundaries(
        cell_size,
        mp.Vector3(),
        resolution,
    )
    _assert_grid_aligned_and_point_safe(
        next_layout,
        source_center,
        resolution,
        2.0,
        boundaries,
    )

    sim.reset_meep()


@pytest.mark.mpi4
def test_prepare_initial_layout_bootstraps_real_meep_mixed_2d_topology():
    nproc = mp.count_processors()
    if nproc < 4:
        pytest.skip("requires at least four MPI ranks")

    resolution = 4
    min_chunk_cells = 2
    protected_gap_cells = 2.0
    cell_size = mp.Vector3(8.0, 8.0)
    geometry_center = mp.Vector3(0.125, -0.125)
    boundaries = _actual_grid_boundaries(
        cell_size,
        geometry_center,
        resolution,
    )
    target_position = mp.Vector3(
        boundaries[0][boundaries[0].size // 2],
        boundaries[1][boundaries[1].size // 2],
    )
    forward_source_position = mp.Vector3(
        boundaries[0][6],
        boundaries[1][8],
    )
    forward_source = mp.Source(
        mp.GaussianSource(frequency=1.0, fwidth=0.5),
        component=mp.Ez,
        center=forward_source_position,
    )
    simulation = tm.SimulationSpec(
        cell_size=cell_size,
        boundary_layers=[],
        geometry=[],
        sources=[forward_source],
        resolution=resolution,
        dimensions=2,
        geometry_center=geometry_center,
    )
    balancer = tm.AdaptiveAdjointChunkBalancer(
        protected_gap_cells=protected_gap_cells,
        min_chunk_cells=min_chunk_cells,
    )

    decision = balancer.prepare_initial_layout(
        simulation,
        forward_sources=(forward_source,),
        adjoint_source_positions=(target_position,),
        adjoint_source_size=mp.Vector3(),
        adjoint_source_amplitude=1.0,
    )

    assert decision.method == "layout"
    assert decision.source_sizes == (mp.Vector3(),)
    assert decision.source_amplitudes == (1.0,)
    assert decision.changed_axes == ((),)
    layout = decision.chunk_layout
    assert layout is simulation.chunk_layout
    assert _split_axes(layout) == {mp.X, mp.Y}
    proc_ids = _assert_grid_aligned_and_point_safe(
        layout,
        forward_source_position,
        resolution,
        protected_gap_cells,
        boundaries,
        min_chunk_cells,
    )
    target_proc_ids = _assert_grid_aligned_and_point_safe(
        layout,
        target_position,
        resolution,
        protected_gap_cells,
        boundaries,
        min_chunk_cells,
    )
    assert sorted(proc_ids) == list(range(nproc))
    assert sorted(target_proc_ids) == list(range(nproc))
    _run_effective_source(simulation, decision, target_position, mp.Ez)


@pytest.mark.mpi4
@pytest.mark.mpi8
def test_prepare_initial_layout_bootstraps_real_meep_mixed_3d_topology():
    nproc = mp.count_processors()
    if nproc < 4:
        pytest.skip("requires at least four MPI ranks")

    resolution = 4
    min_chunk_cells = 2
    protected_gap_cells = 2.0
    cell_size = mp.Vector3(4.0, 4.0, 4.0)
    geometry_center = mp.Vector3(0.125, -0.125, 0.125)
    boundaries = _actual_grid_boundaries(
        cell_size,
        geometry_center,
        resolution,
    )
    target_position = mp.Vector3(
        boundaries[0][boundaries[0].size // 2],
        boundaries[1][boundaries[1].size // 2],
        boundaries[2][boundaries[2].size // 2],
    )
    forward_source_position = mp.Vector3(
        boundaries[0][3],
        boundaries[1][4],
        boundaries[2][5],
    )
    forward_source = mp.Source(
        mp.GaussianSource(frequency=1.0, fwidth=0.5),
        component=mp.Ex,
        center=forward_source_position,
    )
    simulation = tm.SimulationSpec(
        cell_size=cell_size,
        boundary_layers=[],
        geometry=[],
        sources=[forward_source],
        resolution=resolution,
        dimensions=3,
        geometry_center=geometry_center,
    )
    balancer = tm.AdaptiveAdjointChunkBalancer(
        protected_gap_cells=protected_gap_cells,
        min_chunk_cells=min_chunk_cells,
    )

    decision = balancer.prepare_initial_layout(
        simulation,
        forward_sources=(forward_source,),
        adjoint_source_positions=(target_position,),
        adjoint_source_size=mp.Vector3(),
        adjoint_source_amplitude=1.0,
    )

    assert decision.method == "layout"
    assert decision.source_sizes == (mp.Vector3(),)
    assert decision.source_amplitudes == (1.0,)
    assert decision.changed_axes == ((),)
    layout = decision.chunk_layout
    assert layout is simulation.chunk_layout
    axes = _split_axes(layout)
    assert len(axes) >= 2
    proc_ids = _assert_grid_aligned_and_point_safe(
        layout,
        forward_source_position,
        resolution,
        protected_gap_cells,
        boundaries,
        min_chunk_cells,
    )
    target_proc_ids = _assert_grid_aligned_and_point_safe(
        layout,
        target_position,
        resolution,
        protected_gap_cells,
        boundaries,
        min_chunk_cells,
    )
    assert len(proc_ids) == nproc
    assert sorted(proc_ids) == list(range(nproc))
    assert sorted(target_proc_ids) == list(range(nproc))

    adjoint_source = mp.Source(
        mp.GaussianSource(frequency=1.0, fwidth=0.5),
        component=mp.Ex,
        center=target_position,
        size=decision.source_sizes[0],
        amplitude=decision.source_amplitudes[0],
    )
    sim = simulation.make([adjoint_source])
    sim.run(until=0.1)
    try:
        assert _split_axes(sim.chunk_layout) == axes
        assert set(sim.structure.get_chunk_owners()) == set(proc_ids)
        for volume in sim.structure.get_chunk_volumes():
            assert volume.num_direction(mp.X) >= min_chunk_cells
            assert volume.num_direction(mp.Y) >= min_chunk_cells
            assert volume.num_direction(mp.Z) >= min_chunk_cells
    finally:
        sim.reset_meep()


@pytest.mark.mpi4
def test_cartesian_3d_objective_gradient_runs_twice_with_adaptive_layout():
    if mp.count_processors() != 4:
        pytest.skip("requires exactly four MPI ranks")

    objective, shape = _make_cartesian_3d_objective()
    weights = np.full(np.prod(shape), 0.45)
    values = []
    gradients = []
    for offset in (0.0, 0.05):
        value, gradient = objective.fom_and_grad(weights + offset)
        values.append(value)
        gradients.append(gradient)

        assert objective.last_chunk_balance is not None
        assert len(_split_axes(objective._simulation_spec.chunk_layout)) >= 2

    assert np.all(np.isfinite(values))
    assert all(gradient.shape == (np.prod(shape),) for gradient in gradients)
    assert all(np.all(np.isfinite(gradient)) for gradient in gradients)
    assert all(np.linalg.norm(gradient) > 0.0 for gradient in gradients)


@pytest.mark.mpi2
def test_adaptive_balancer_updates_real_cylindrical_partition():
    nproc = mp.count_processors()
    if nproc < 2:
        pytest.skip("requires at least two MPI ranks")

    resolution = 4
    min_chunk_cells = 2
    cell_size = mp.Vector3(8.0, 0.0, 4.0)
    geometry_center = mp.Vector3()
    source_position = mp.Vector3(1.25, 0.0, -0.75)
    source = mp.Source(
        mp.GaussianSource(frequency=1.0, fwidth=0.5),
        component=mp.Ep,
        center=source_position,
    )
    sim = mp.Simulation(
        cell_size=cell_size,
        geometry_center=geometry_center,
        resolution=resolution,
        dimensions=mp.CYLINDRICAL,
        m=0,
        sources=[source],
    )
    sim.run(until=0.5)
    initial_signature = _topology_signature(sim.chunk_layout)

    balancer = tm.AdaptiveAdjointChunkBalancer(
        protected_points=[source_position],
        sensitivity=1.0,
        imbalance_threshold=1.001,
        protected_gap_cells=1.0,
        min_chunk_cells=min_chunk_cells,
    )
    next_layout = balancer.rebalance(
        sim,
        local_extra_work_seconds=10.0 * mp.my_rank() / (nproc - 1),
    )

    assert isinstance(next_layout, mp.BinaryPartition)
    assert balancer.last_report is not None
    assert balancer.last_report.applied
    assert balancer.last_report.moved_boundaries > 0
    assert _topology_signature(next_layout) == initial_signature
    boundaries = _actual_cylindrical_grid_boundaries(
        cell_size,
        geometry_center,
        resolution,
    )
    proc_ids = _assert_grid_aligned_and_point_safe(
        next_layout,
        source_position,
        resolution,
        1.0,
        boundaries,
        min_chunk_cells,
        active_axes=(0, 2),
    )
    sim.reset_meep()

    rebalanced = mp.Simulation(
        cell_size=cell_size,
        geometry_center=geometry_center,
        resolution=resolution,
        dimensions=mp.CYLINDRICAL,
        m=0,
        sources=[source],
        chunk_layout=next_layout,
    )
    try:
        rebalanced.run(until=0.1)
        assert set(rebalanced.structure.get_chunk_owners()) == set(proc_ids)
        for volume in rebalanced.structure.get_chunk_volumes():
            assert volume.num_direction(mp.R) >= min_chunk_cells
            assert volume.num_direction(mp.Z) >= min_chunk_cells
    finally:
        rebalanced.reset_meep()


@pytest.mark.parametrize("objective_kind", ["tda", "multi"])
@pytest.mark.mpi2
def test_cylindrical_objective_runs_with_adaptive_balancer(objective_kind):
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    objective, shape = _make_cylindrical_objective(objective_kind)
    native_layout = _native_layout(objective._simulation_spec)
    value, gradient = objective.fom_and_grad(
        np.full(np.prod(shape), 0.5),
    )
    reference, _ = _make_cylindrical_objective(
        objective_kind,
        adaptive=False,
    )
    reference_value, reference_gradient = reference.fom_and_grad(
        np.full(np.prod(shape), 0.5),
    )

    assert np.isfinite(value)
    assert gradient.shape == (np.prod(shape),)
    assert np.all(np.isfinite(gradient))
    assert np.allclose(value, reference_value, rtol=1e-10, atol=1e-12)
    assert np.allclose(
        gradient,
        reference_gradient,
        rtol=1e-9,
        atol=1e-12,
    )
    assert objective.last_source_boundary_decision.method == "layout"
    assert all(
        size == mp.Vector3()
        for size in objective.last_source_boundary_decision.source_sizes
    )
    assert objective.last_chunk_balance is not None
    assert set(objective.last_chunk_balance.topology) & {"R", "Z"}
    assert _topology_signature(
        objective._simulation_spec.chunk_layout
    ) == _topology_signature(native_layout)


@pytest.mark.mpi4
def test_auto_finite_fallback_preserves_real_meep_mixed_2d_topology():
    nproc = mp.count_processors()
    if nproc < 4:
        pytest.skip("requires at least four MPI ranks")

    resolution = 4
    target_position = mp.Vector3()
    simulation = tm.SimulationSpec(
        cell_size=mp.Vector3(4.0, 4.0),
        boundary_layers=[],
        geometry=[],
        sources=[],
        resolution=resolution,
        dimensions=2,
        geometry_center=mp.Vector3(),
    )
    native_layout = _native_layout(simulation)
    native_axes = _split_axes(native_layout)
    assert native_axes == {mp.X, mp.Y}
    balancer = tm.AdaptiveAdjointChunkBalancer(
        protected_gap_cells=100.0,
        min_chunk_cells=2,
    )

    decision = balancer.prepare_initial_layout(
        simulation,
        adjoint_source_positions=(target_position,),
        adjoint_source_size=mp.Vector3(),
        adjoint_source_amplitude=1.0,
    )

    assert decision.method == "finite"
    assert _topology_signature(decision.chunk_layout) == _topology_signature(
        native_layout
    )
    assert _vector_values(decision.source_sizes[0]) == (0.25, 0.25, 0.0)
    assert decision.source_amplitudes == (16.0,)
    assert set(decision.changed_axes[0]) == {"x", "y"}
    _run_effective_source(simulation, decision, target_position, mp.Ez)


@pytest.mark.mpi8
def test_auto_finite_fallback_preserves_real_meep_mixed_3d_topology():
    nproc = mp.count_processors()
    if nproc < 8:
        pytest.skip("requires at least eight MPI ranks")

    resolution = 4
    target_position = mp.Vector3()
    simulation = tm.SimulationSpec(
        cell_size=mp.Vector3(4.0, 4.0, 4.0),
        boundary_layers=[],
        geometry=[],
        sources=[],
        resolution=resolution,
        dimensions=3,
        geometry_center=mp.Vector3(),
    )
    native_layout = _native_layout(simulation)
    native_axes = _split_axes(native_layout)
    assert native_axes == {mp.X, mp.Y, mp.Z}
    balancer = tm.AdaptiveAdjointChunkBalancer(
        protected_gap_cells=100.0,
        min_chunk_cells=2,
    )

    decision = balancer.prepare_initial_layout(
        simulation,
        adjoint_source_positions=(target_position,),
        adjoint_source_size=mp.Vector3(),
        adjoint_source_amplitude=1.0,
    )

    assert decision.method == "finite"
    assert _topology_signature(decision.chunk_layout) == _topology_signature(
        native_layout
    )
    assert _vector_values(decision.source_sizes[0]) == (0.25, 0.25, 0.25)
    assert decision.source_amplitudes == (64.0,)
    assert set(decision.changed_axes[0]) == {"x", "y", "z"}
    _run_effective_source(simulation, decision, target_position, mp.Ex)


if __name__ == "__main__":
    test_adaptive_balancer_updates_a_real_meep_partition()
    test_prepare_initial_layout_bootstraps_real_meep_mixed_2d_topology()
    test_prepare_initial_layout_bootstraps_real_meep_mixed_3d_topology()
    test_auto_finite_fallback_preserves_real_meep_mixed_2d_topology()
    test_auto_finite_fallback_preserves_real_meep_mixed_3d_topology()
