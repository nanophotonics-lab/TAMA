import meep as mp
import numpy as np
import pytest

from tama.sampling_grid import FastPointMonitor


def _make_simulation(m=1):
    return mp.Simulation(
        cell_size=mp.Vector3(2.0, 0.0, 2.4),
        boundary_layers=[
            mp.PML(0.2, direction=mp.R),
            mp.PML(0.2, direction=mp.Z),
        ],
        sources=[
            mp.Source(
                mp.GaussianSource(frequency=0.8, fwidth=0.5),
                component=mp.Er,
                center=mp.Vector3(0.6, 0.0, -0.7),
            )
        ],
        dimensions=mp.CYLINDRICAL,
        m=m,
        resolution=10,
        force_all_components=True,
        force_complex_fields=True,
    )


@pytest.mark.mpi2
@pytest.mark.parametrize("component", [mp.Er, mp.Ep, mp.Ez])
def test_cylindrical_point_plan_matches_meep_and_builds_indexed_transpose(
    component,
):
    mp.verbosity(0)
    simulation = _make_simulation()
    simulation.run(until=3.0)
    try:
        positions = [
            mp.Vector3(0.63, 0.0, 0.17),
            mp.Vector3(1.07, 0.0, -0.23),
        ]
        monitor = FastPointMonitor(simulation, component, positions)

        assert monitor.ensure_plan() is not None
        assert np.allclose(
            monitor.sample(),
            [simulation.get_field_point(component, position) for position in positions],
            rtol=2e-14,
            atol=2e-14,
        )

        indexed_offsets, source_data, indexed_amplitudes = (
            monitor.indexed_transpose_stencil()
        )
        assert indexed_offsets.shape == (len(positions) + 1,)
        assert indexed_offsets[0] == 0
        assert indexed_offsets[-1] == len(source_data)
        assert indexed_amplitudes.shape == (len(source_data),)
        assert np.all(indexed_amplitudes.real > 0.0)
        assert np.allclose(indexed_amplitudes.imag, 0.0)
        for data in source_data:
            assert data.near_fd_comp == component
            assert data.fc_idx >= 0

        if mp.count_processors() == 1:
            assert np.all(np.diff(indexed_offsets) > 0)

        zero_source = mp.CustomSource(src_func=lambda _time: 0j)
        for data, amplitude in zip(source_data, indexed_amplitudes):
            mp.IndexedSource(
                zero_source,
                data,
                np.asarray([amplitude], dtype=np.complex128),
            ).add_source(simulation)
        simulation.fields.require_source_components()
    finally:
        simulation.reset_meep()


@pytest.mark.parametrize(
    ("m", "component"),
    [
        (0, mp.Ez),
        (0, mp.Hz),
        (1, mp.Er),
        (1, mp.Ep),
        (1, mp.Hr),
        (1, mp.Hp),
        (-1, mp.Er),
        (-1, mp.Ep),
        (-1, mp.Hr),
        (-1, mp.Hp),
    ],
)
def test_cylindrical_axis_point_plan_matches_meep_and_builds_transpose(
    m,
    component,
):
    mp.verbosity(0)
    simulation = _make_simulation(m)
    simulation.run(until=3.0)
    try:
        position = mp.Vector3(0.0, 0.0, 0.13)
        monitor = FastPointMonitor(
            simulation,
            component,
            [position],
        )
        sampled = monitor.sample()
        assert monitor.ensure_plan() is not None
        assert np.allclose(
            sampled,
            [simulation.get_field_point(component, position)],
            rtol=2e-14,
            atol=2e-14,
        )

        offsets, source_data, amplitudes = monitor.indexed_transpose_stencil()
        assert offsets.shape == (2,)
        assert offsets[0] == 0
        assert offsets[1] == len(source_data)
        if mp.count_processors() == 1:
            assert len(source_data) > 0
        else:
            assert mp.sum_to_all(len(source_data)) > 0
        assert amplitudes.shape == (len(source_data),)
        assert np.all(np.isfinite(amplitudes))
        expected_source_component = {
            mp.Ep: mp.Er,
            mp.Hr: mp.Hp,
        }.get(component, component)
        for data in source_data:
            assert data.near_fd_comp == expected_source_component
            assert data.fc_idx >= 0

        zero_source = mp.CustomSource(src_func=lambda _time: 0j)
        for data, amplitude in zip(source_data, amplitudes):
            mp.IndexedSource(
                zero_source,
                data,
                np.asarray([amplitude], dtype=np.complex128),
            ).add_source(simulation)
        simulation.fields.require_source_components()
    finally:
        simulation.reset_meep()


@pytest.mark.parametrize(
    ("m", "component", "independent_component"),
    [
        (1, mp.Ep, mp.Er),
        (1, mp.Hr, mp.Hp),
        (-1, mp.Ep, mp.Er),
        (-1, mp.Hr, mp.Hp),
    ],
)
def test_cylindrical_near_axis_point_plan_expands_dependent_axis_entry(
    m,
    component,
    independent_component,
):
    mp.verbosity(0)
    simulation = _make_simulation(m)
    simulation.run(until=3.0)
    try:
        position = mp.Vector3(0.02, 0.0, 0.13)
        monitor = FastPointMonitor(simulation, component, [position])

        assert np.allclose(
            monitor.sample(),
            [simulation.get_field_point(component, position)],
            rtol=2e-14,
            atol=2e-14,
        )

        _, source_data, amplitudes = monitor.indexed_transpose_stencil()
        if mp.count_processors() == 1:
            assert len(source_data) > 0
        else:
            assert mp.sum_to_all(len(source_data)) > 0
        assert np.all(np.isfinite(amplitudes))
        if mp.count_processors() == 1:
            assert {data.near_fd_comp for data in source_data} == {
                component,
                independent_component,
            }
    finally:
        simulation.reset_meep()
