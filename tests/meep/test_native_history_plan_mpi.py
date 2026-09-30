import gc
import tempfile

import meep as mp
import numpy as np
import pytest

import tama as tm
import tama.sampling_grid as sampling_grid
from tama.sampling_grid import FastPointMonitor


@pytest.mark.mpi2
def test_configured_history_path_matches_legacy_native_path_exactly():
    if mp.count_processors() != 2:
        pytest.skip("requires exactly two MPI ranks")
    assert tm.native_sampler_available()

    source_center = mp.Vector3(-0.23, -0.31)
    layout = mp.BinaryPartition(data=[(mp.X, 0.0), 0, 1])
    sim = mp.Simulation(
        cell_size=mp.Vector3(8, 8),
        resolution=8,
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

    monitor_positions = [
        mp.Vector3(-0.41, -0.17),
        mp.Vector3(0.03, 0.29),
        mp.Vector3(0.47, -0.33),
        mp.Vector3(-0.19, 0.51),
    ]
    legacy_monitors = np.asarray([
        sim.get_field_point(mp.Ez, position) for position in monitor_positions
    ])
    packed_monitors = FastPointMonitor(sim, mp.Ez, monitor_positions).sample()
    assert np.array_equal(packed_monitors, legacy_monitors)

    call_counts = {"local": 0, "reduced": 0, "immediate": 0}
    original_local = sampling_grid.native_sampler.sample_component_point_plan_local_into
    original_reduce = sampling_grid.native_sampler.reduce_complex_grid_sum_inplace
    original_immediate = sampling_grid.native_sampler.sample_component_point_plan_allreduced

    def counted_local(plan, destination):
        call_counts["local"] += 1
        return original_local(plan, destination)

    def counted_reduce(history):
        call_counts["reduced"] += 1
        return original_reduce(history)

    def counted_immediate(plan):
        call_counts["immediate"] += 1
        return original_immediate(plan)

    try:
        sampling_grid.native_sampler.sample_component_point_plan_local_into = counted_local
        sampling_grid.native_sampler.reduce_complex_grid_sum_inplace = counted_reduce
        sampling_grid.native_sampler.sample_component_point_plan_allreduced = counted_immediate
        deferred_monitor = FastPointMonitor(sim, mp.Ez, monitor_positions)
        local_monitor_history = np.empty((3, len(monitor_positions)), dtype=np.complex128)
        for row in local_monitor_history:
            deferred_monitor.sample_history_into(row)
        reduced_monitor_history = deferred_monitor.reduce_history(local_monitor_history)
    finally:
        sampling_grid.native_sampler.sample_component_point_plan_local_into = original_local
        sampling_grid.native_sampler.reduce_complex_grid_sum_inplace = original_reduce
        sampling_grid.native_sampler.sample_component_point_plan_allreduced = original_immediate

    assert deferred_monitor.ensure_plan() is not None
    assert call_counts == {"local": 3, "reduced": 1, "immediate": 0}
    assert reduced_monitor_history is local_monitor_history
    assert np.allclose(
        reduced_monitor_history,
        np.tile(legacy_monitors, (3, 1)),
        rtol=1.0e-12,
        atol=1.0e-13,
    )

    coords = np.linspace(-3.5, 3.5, 57)
    field = tm.FastFieldGrid(sim, mp.Ez, coords, coords)
    local_indices, boundary_indices = field.history_sampling_indices()
    history_indices = np.concatenate([local_indices, boundary_indices])

    legacy_values = field.sample_history_values(local_indices, boundary_indices)
    configured_values = np.empty_like(legacy_values)
    field.sample_history_values_into(
        local_indices,
        boundary_indices,
        configured_values,
    )
    assert np.array_equal(configured_values, legacy_values)

    with tempfile.TemporaryDirectory() as temp_dir:
        history = np.memmap(
            f"{temp_dir}/history.dat",
            mode="w+",
            dtype=np.complex128,
            shape=(2, legacy_values.size),
        )
        field.sample_history_values_into(
            local_indices,
            boundary_indices,
            history[1],
        )
        assert np.array_equal(history[1], legacy_values)
        del history

    configured_values64 = np.empty(legacy_values.shape, dtype=np.complex64)
    field.sample_history_values_into(
        local_indices,
        boundary_indices,
        configured_values64,
    )
    assert np.array_equal(configured_values64, legacy_values.astype(np.complex64))

    multipliers = (
        np.arange(history_indices.size, dtype=float)
        + 1j * np.arange(history_indices.size, dtype=float)[::-1]
    ) / max(history_indices.size, 1)
    legacy_gradient = tm.FastGradientGrid(sim, mp.Ez, coords, coords)
    legacy_gradient.accumulate_indexed(history_indices, multipliers)
    legacy_result = legacy_gradient.finalize()

    configured_gradient = tm.FastGradientGrid(sim, mp.Ez, coords, coords)
    configured_gradient.configure_fixed_indices(history_indices)
    configured_gradient.accumulate_fixed(multipliers)
    configured_result = configured_gradient.finalize()
    assert np.array_equal(configured_result, legacy_result)

    del configured_gradient, legacy_gradient, field
    gc.collect()
    sim.reset_meep()


@pytest.mark.mpi4
def test_history_boundary_reduction_stays_within_meep_process_group():
    if mp.comm.Get_size() != 4:
        pytest.skip("requires exactly four MPI ranks")

    group_index = mp.divide_parallel_processes(2)
    sim = None
    field = None
    try:
        sim = mp.Simulation(
            cell_size=mp.Vector3(2, 2),
            resolution=10,
            chunk_layout=mp.BinaryPartition(data=[(mp.X, 0.0), 0, 1]),
            sources=[
                mp.Source(
                    mp.GaussianSource(frequency=1.0, fwidth=0.5),
                    component=mp.Ez,
                    center=mp.Vector3(-0.35, 0.17),
                    amplitude=group_index + 1,
                )
            ],
        )
        sim.run(until=1.0)

        point = mp.Vector3(0.05, 0.03)
        expected = sim.get_field_point(mp.Ez, point)
        field = tm.FastFieldGrid(sim, mp.Ez, [point.x], [point.y])
        local_indices, boundary_indices = field.history_sampling_indices()
        sampled = np.empty(
            local_indices.size + boundary_indices.size,
            dtype=np.complex128,
        )
        field.sample_history_values_into(
            local_indices,
            boundary_indices,
            sampled,
        )

        assert local_indices.size == 0
        assert boundary_indices.size == 1
        assert abs(expected) > 1.0e-12
        assert np.allclose(sampled, [expected], rtol=1.0e-12, atol=1.0e-13)
    finally:
        if field is not None:
            field.plan = None
        gc.collect()
        if sim is not None:
            sim.reset_meep()
        mp.end_divide_parallel()


@pytest.mark.mpi2
def test_gradient_finalize_reduces_when_only_one_rank_accumulated():
    if mp.count_processors() != 2:
        pytest.skip("requires exactly two MPI ranks")

    gradient = object.__new__(tm.FastGradientGrid)
    gradient.local = np.zeros((2, 1), dtype=np.complex128)
    gradient.needs_reduce = mp.my_rank() == 0
    if gradient.needs_reduce:
        gradient.local[0, 0] = 3.0 + 4.0j

    reduced = gradient.finalize()

    assert np.array_equal(
        reduced,
        np.asarray([[3.0 + 4.0j], [0.0]], dtype=np.complex128),
    )


if __name__ == "__main__":
    test_configured_history_path_matches_legacy_native_path_exactly()
    test_gradient_finalize_reduces_when_only_one_rank_accumulated()
