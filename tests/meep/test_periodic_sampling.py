import meep as mp
import numpy as np
import pytest

from tama import native_sampler
from tama.sampling_grid import FastFieldGrid, FastPointMonitor


@pytest.fixture(
    scope="module",
    params=(
        pytest.param(mp.Vector3(), id="periodic-k0"),
        pytest.param(mp.Vector3(0.23, -0.17), id="bloch"),
    ),
)
def periodic_simulation(request):
    mp.verbosity(0)
    k_point = request.param
    simulation = mp.Simulation(
        cell_size=mp.Vector3(2.0, 2.0),
        resolution=10,
        k_point=k_point,
        force_complex_fields=True,
        sources=[
            mp.Source(
                mp.GaussianSource(frequency=0.8, fwidth=0.5),
                component=mp.Ez,
                center=mp.Vector3(0.17, -0.23),
            )
        ],
    )
    simulation.run(until=2.0)
    try:
        yield simulation, k_point
    finally:
        simulation.reset_meep()


@pytest.mark.mpi2
def test_periodic_point_monitor_matches_meep_across_cell_boundary(
    periodic_simulation,
):
    simulation, _ = periodic_simulation
    positions = (
        mp.Vector3(-1.0, 0.17),
        mp.Vector3(1.0, 0.17),
        mp.Vector3(-0.99, 0.17),
        mp.Vector3(0.13, -1.0),
        mp.Vector3(-1.0, -1.0),
    )
    expected = np.asarray(
        [simulation.get_field_point(mp.Ez, position) for position in positions]
    )
    monitor = FastPointMonitor(simulation, mp.Ez, positions)

    local = np.empty(len(positions), dtype=np.complex128)
    monitor.sample_history_into(local)
    history = monitor.reduce_history(local[np.newaxis, :])[0]

    np.testing.assert_allclose(
        monitor.sample(), expected, rtol=2e-13, atol=1e-15
    )
    np.testing.assert_allclose(
        history, expected, rtol=2e-13, atol=1e-15
    )


@pytest.mark.mpi2
def test_periodic_point_monitor_indexed_transpose_includes_bloch_phase(
    periodic_simulation,
):
    simulation, k_point = periodic_simulation
    monitor = FastPointMonitor(
        simulation,
        mp.Ez,
        (mp.Vector3(-1.0, 0.17), mp.Vector3(1.0, 0.17)),
    )

    offsets, source_data, amplitudes = monitor.indexed_transpose_stencil()
    raw_offsets, components, chunks, local_indices, raw_amplitudes = (
        native_sampler.component_point_plan_indexed_stencil(
            monitor.ensure_plan()
        )
    )
    assert np.array_equal(offsets, raw_offsets)
    np.testing.assert_allclose(amplitudes, raw_amplitudes)
    assert len(source_data) == amplitudes.size

    def stencil_map(point_index):
        point_slice = slice(
            int(offsets[point_index]), int(offsets[point_index + 1])
        )
        keys = [
            (int(component), int(chunk), int(local_index))
            for component, chunk, local_index in zip(
                components[point_slice],
                chunks[point_slice],
                local_indices[point_slice],
            )
        ]
        assert len(keys) == len(set(keys))
        return dict(zip(keys, amplitudes[point_slice]))

    lower = stencil_map(0)
    upper = stencil_map(1)
    lower_phase = np.exp(-2j * np.pi * k_point.x * simulation.cell_size.x)
    global_entry_count = int(mp.sum_to_all(len(upper)))
    local_norm_squared = float(
        sum(abs(amplitude) ** 2 for amplitude in upper.values())
    )
    global_norm_squared = float(mp.sum_to_all(local_norm_squared))

    assert lower.keys() == upper.keys()
    assert global_entry_count > 0
    assert global_norm_squared > 1e-24
    for key in lower:
        np.testing.assert_allclose(
            lower[key],
            lower_phase * upper[key],
            rtol=2e-14,
            atol=2e-14,
        )


@pytest.mark.mpi2
def test_periodic_field_grid_matches_meep_across_cell_boundary(
    periodic_simulation,
):
    simulation, _ = periodic_simulation
    coords_x = (-1.0, -0.99, 0.31, 1.0)
    coords_y = (-1.0, -0.97, 0.17, 1.0)
    expected = np.asarray(
        [
            [
                simulation.get_field_point(mp.Ez, mp.Vector3(x, y))
                for y in coords_y
            ]
            for x in coords_x
        ]
    )
    field = FastFieldGrid(simulation, mp.Ez, coords_x, coords_y)
    raw_sampled = native_sampler.sample_component_grid_allreduced(
        int(simulation.fields.this), coords_x, coords_y, int(mp.Ez)
    )

    local_indices, boundary_indices = field.history_sampling_indices()
    history_indices = np.concatenate((local_indices, boundary_indices))
    history_values = field.sample_history_values(
        local_indices, boundary_indices
    )

    if mp.count_processors() == 1:
        assert np.array_equal(
            np.sort(history_indices), np.arange(expected.size)
        )
    np.testing.assert_allclose(
        field.sample(), expected, rtol=2e-13, atol=1e-15
    )
    np.testing.assert_allclose(
        raw_sampled, expected, rtol=2e-13, atol=1e-15
    )
    np.testing.assert_allclose(
        history_values,
        expected.reshape(-1)[history_indices],
        rtol=2e-13,
        atol=1e-15,
    )


@pytest.mark.mpi2
def test_cartesian_3d_point_monitor_matches_bloch_periodic_boundaries():
    mp.verbosity(0)
    k_point = mp.Vector3(0.11, -0.07, 0.19)
    simulation = mp.Simulation(
        cell_size=mp.Vector3(1.0, 1.25, 0.75),
        resolution=8,
        k_point=k_point,
        force_complex_fields=True,
        sources=[
            mp.Source(
                mp.GaussianSource(frequency=0.8, fwidth=0.5),
                component=mp.Ex,
                center=mp.Vector3(0.13, -0.19, 0.11),
            )
        ],
    )
    simulation.run(until=2.0)
    try:
        positions = (
            mp.Vector3(0.17, -0.13, -0.375),
            mp.Vector3(0.17, -0.13, 0.375),
            mp.Vector3(-0.5, 0.17, -0.11),
            mp.Vector3(0.5, 0.17, -0.11),
            mp.Vector3(-0.49, -0.61, -0.37),
        )
        expected = np.asarray(
            [
                simulation.get_field_point(mp.Ex, position)
                for position in positions
            ]
        )
        sampled = FastPointMonitor(simulation, mp.Ex, positions).sample()

        assert np.linalg.norm(expected) > 1e-12
        np.testing.assert_allclose(
            sampled, expected, rtol=2e-13, atol=1e-15
        )
        np.testing.assert_allclose(
            expected[0],
            np.exp(-2j * np.pi * k_point.z * simulation.cell_size.z)
            * expected[1],
            rtol=2e-13,
            atol=1e-15,
        )
    finally:
        simulation.reset_meep()


@pytest.mark.mpi2
def test_cylindrical_point_monitor_matches_bloch_periodic_z_boundary():
    mp.verbosity(0)
    k_point = mp.Vector3(0.0, 0.0, 0.21)
    simulation = mp.Simulation(
        cell_size=mp.Vector3(1.2, 0.0, 1.4),
        dimensions=mp.CYLINDRICAL,
        m=1,
        resolution=10,
        boundary_layers=[mp.PML(0.2, direction=mp.R)],
        k_point=k_point,
        force_all_components=True,
        force_complex_fields=True,
        sources=[
            mp.Source(
                mp.GaussianSource(frequency=0.8, fwidth=0.5),
                component=mp.Er,
                center=mp.Vector3(0.43, 0.0, -0.17),
            )
        ],
    )
    simulation.run(until=2.0)
    try:
        positions = (
            mp.Vector3(0.0, 0.0, -0.7),
            mp.Vector3(0.0, 0.0, 0.7),
            mp.Vector3(0.02, 0.0, -0.7),
            mp.Vector3(0.02, 0.0, 0.7),
        )
        expected = np.asarray(
            [
                simulation.get_field_point(mp.Ep, position)
                for position in positions
            ]
        )
        sampled = FastPointMonitor(simulation, mp.Ep, positions).sample()
        lower_phase = np.exp(
            -2j * np.pi * k_point.z * simulation.cell_size.z
        )

        assert np.linalg.norm(expected) > 1e-12
        np.testing.assert_allclose(
            sampled, expected, rtol=2e-13, atol=1e-15
        )
        np.testing.assert_allclose(
            expected[[0, 2]],
            lower_phase * expected[[1, 3]],
            rtol=2e-13,
            atol=1e-15,
        )
    finally:
        simulation.reset_meep()
