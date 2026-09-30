import meep as mp
import numpy as np
import pytest

from tama import native_sampler
from tama.objectives import _adjoint_point_sources
from tama.sampling_grid import FastPointMonitor


@pytest.mark.mpi2
def test_cartesian_3d_point_plan_matches_meep():
    mp.verbosity(0)
    simulation = mp.Simulation(
        cell_size=mp.Vector3(1.2, 1.0, 0.8),
        sources=[
            mp.Source(
                mp.GaussianSource(frequency=0.8, fwidth=0.5),
                component=mp.Ex,
                center=mp.Vector3(),
            )
        ],
        resolution=8,
        force_complex_fields=True,
    )
    simulation.run(until=1.0)
    try:
        positions = [
            mp.Vector3(0.0, 0.23, -0.13),
            mp.Vector3(0.13, -0.17, 0.11),
            mp.Vector3(-0.21, 0.19, -0.07),
        ]
        monitor = FastPointMonitor(simulation, mp.Ex, positions)
        local_values = np.empty(len(positions), dtype=np.complex128)
        monitor.sample_history_into(local_values)
        deferred_values = monitor.reduce_history(local_values[np.newaxis, :])[0]
        expected = [
            simulation.get_field_point(mp.Ex, position)
            for position in positions
        ]

        offsets, source_data, amplitudes = monitor.indexed_transpose_stencil()

        assert native_sampler.API_VERSION == 12
        assert np.allclose(monitor.sample(), expected, rtol=2e-14, atol=2e-14)
        assert np.allclose(deferred_values, expected, rtol=2e-14, atol=2e-14)
        assert offsets.shape == (len(positions) + 1,)
        assert offsets[0] == 0
        assert offsets[-1] == len(source_data) == amplitudes.size
        assert int(mp.sum_to_all(len(source_data))) > 0
        assert np.all(np.isfinite(amplitudes))

        zero_source = mp.CustomSource(src_func=lambda _time: 0j)
        batched_sources = []
        for point_index, position in enumerate(positions):
            point_slice = slice(
                int(offsets[point_index]),
                int(offsets[point_index + 1]),
            )
            batched_sources.extend(
                _adjoint_point_sources(
                    zero_source,
                    mp.Ex,
                    position,
                    mp.Vector3(),
                    1.0,
                    (
                        source_data[point_slice],
                        amplitudes[point_slice],
                    ),
                )
            )
        assert sum(source.amp_arr.size for source in batched_sources) == len(
            source_data
        )
        assert int(mp.sum_to_all(len(batched_sources))) < int(
            mp.sum_to_all(len(source_data))
        )
        for source in batched_sources:
            source.add_source(simulation)
        simulation.fields.require_source_components()
    finally:
        simulation.reset_meep()


@pytest.mark.mpi2
def test_cartesian_2d_x_zero_point_has_indexed_transpose_support():
    simulation = mp.Simulation(
        cell_size=mp.Vector3(1.0, 1.0),
        sources=[
            mp.Source(
                mp.GaussianSource(frequency=0.8, fwidth=0.5),
                component=mp.Ex,
                center=mp.Vector3(),
            )
        ],
        resolution=8,
        force_complex_fields=True,
    )
    simulation.run(until=0.1)
    try:
        monitor = FastPointMonitor(
            simulation,
            mp.Ex,
            [mp.Vector3(0.0, 0.13)],
        )
        offsets, source_data, amplitudes = monitor.indexed_transpose_stencil()

        assert offsets.shape == (2,)
        assert offsets[0] == 0
        assert offsets[-1] == len(source_data) == amplitudes.size
        assert int(mp.sum_to_all(len(source_data))) > 0
        assert np.all(np.isfinite(amplitudes))
    finally:
        simulation.reset_meep()


@pytest.mark.mpi2
def test_cartesian_2d_point_plan_rejects_z_coordinates():
    simulation = mp.Simulation(
        cell_size=mp.Vector3(1.0, 1.0),
        resolution=8,
    )
    simulation.init_sim()
    try:
        with pytest.raises(
            ValueError,
            match="only for Cartesian 3D point plans",
        ):
            native_sampler.create_component_point_plan(
                int(simulation.fields.this),
                [0.0],
                [0.0],
                [0.0],
                int(mp.Ez),
            )
    finally:
        simulation.reset_meep()
