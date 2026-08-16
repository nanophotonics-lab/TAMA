import meep as mp
import numpy as np
import pytest

from tama.eigenmode import (
    _coefficient_history_from_overlaps,
    _coefficient_overlap_pullback,
    _coefficient_projection_weights,
    coefficient_history,
    coefficient_history_pullback,
    mode_power,
)
from tama.flux import flux_component_pairs
from tama.multi_tda_objective import temporal_convolve_signal
from tama.sampling_grid import FastPointMonitor


def _complex_positive_power_mode(normal, rng, n_points):
    mode_fields = {}
    for electric, magnetic, sign in flux_component_pairs(normal):
        values = (
            rng.standard_normal(n_points)
            + 1j * rng.standard_normal(n_points)
        )
        mode_fields[electric] = values
        mode_fields[magnetic] = sign * values
    return mode_fields


def _contract_histories(histories, electric_weights, magnetic_weights):
    n_samples = next(iter(histories.values())).shape[0]
    overlaps = np.zeros((n_samples, 2), dtype=np.complex128)
    for component, weights in electric_weights.items():
        overlaps[:, 0] += histories[component] @ weights
    for component, weights in magnetic_weights.items():
        overlaps[:, 1] += histories[component] @ weights
    return overlaps


def _dense_coefficient_reference(
    histories,
    mode_fields,
    normal,
    spatial_weights,
):
    normalization = 1.0 / np.sqrt(
        abs(mode_power(mode_fields, normal, spatial_weights))
    )
    n_samples = next(iter(histories.values())).shape[0]
    coefficient = np.zeros(n_samples - 1, dtype=np.complex128)
    for electric, magnetic, sign in flux_component_pairs(normal):
        centered_magnetic = 0.5 * (
            histories[magnetic][:-1] + histories[magnetic][1:]
        )
        coefficient += (
            0.5
            * normalization
            * sign
            * np.sum(
                spatial_weights
                * (
                    histories[electric][:-1]
                    * np.conjugate(mode_fields[magnetic])
                    + np.conjugate(mode_fields[electric])
                    * centered_magnetic
                ),
                axis=1,
            )
        )
    return coefficient


def _dense_pullback_reference(
    mode_fields,
    normal,
    spatial_weights,
    coefficient_covector,
    sample_count,
):
    normalization = 1.0 / np.sqrt(
        abs(mode_power(mode_fields, normal, spatial_weights))
    )
    n_points = len(spatial_weights)
    pulled_back = {
        component: np.zeros(
            (sample_count, n_points),
            dtype=np.complex128,
        )
        for component in mode_fields
    }
    time_covector = coefficient_covector[:, np.newaxis]
    for electric, magnetic, sign in flux_component_pairs(normal):
        common = (
            0.5
            * normalization
            * sign
            * time_covector
            * spatial_weights
        )
        pulled_back[electric][:-1] += common * np.conjugate(
            mode_fields[magnetic]
        )
        centered_magnetic = common * np.conjugate(mode_fields[electric])
        pulled_back[magnetic][:-1] += 0.5 * centered_magnetic
        pulled_back[magnetic][1:] += 0.5 * centered_magnetic
    return pulled_back


def test_contracted_history_matches_dense_projection_after_fir_filtering():
    rng = np.random.default_rng(20260810)
    normal = mp.Vector3(1.0)
    n_samples = 12
    n_points = 5
    spatial_weights = rng.uniform(0.2, 1.3, n_points)
    mode_fields = _complex_positive_power_mode(normal, rng, n_points)
    histories = {
        component: rng.standard_normal((n_samples, n_points))
        for component in mode_fields
    }
    electric_weights, magnetic_weights = _coefficient_projection_weights(
        mode_fields,
        normal,
        spatial_weights,
    )

    overlap_history = _contract_histories(
        histories,
        electric_weights,
        magnetic_weights,
    )
    expected = _dense_coefficient_reference(
        histories,
        mode_fields,
        normal,
        spatial_weights,
    )
    np.testing.assert_allclose(
        coefficient_history(histories, mode_fields, normal, spatial_weights),
        expected,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        _coefficient_history_from_overlaps(overlap_history),
        expected,
        rtol=1.0e-13,
        atol=1.0e-13,
    )

    kernel = np.asarray([0.3, -0.2, 0.7, 0.1, -0.4])
    dt = 0.125
    filtered_histories = {
        component: temporal_convolve_signal(history, kernel, dt)
        for component, history in histories.items()
    }
    filtered_overlaps = temporal_convolve_signal(
        overlap_history,
        kernel,
        dt,
    )

    expected_filtered = _dense_coefficient_reference(
        filtered_histories,
        mode_fields,
        normal,
        spatial_weights,
    )
    np.testing.assert_allclose(
        coefficient_history(
            filtered_histories,
            mode_fields,
            normal,
            spatial_weights,
        ),
        expected_filtered,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        _coefficient_history_from_overlaps(filtered_overlaps),
        expected_filtered,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_contracted_pullback_matches_dense_projection_pullback():
    rng = np.random.default_rng(20260811)
    normal = mp.Vector3(0.0, 0.0, -1.0)
    n_samples = 10
    n_points = 4
    spatial_weights = rng.uniform(0.1, 1.0, n_points)
    mode_fields = _complex_positive_power_mode(normal, rng, n_points)
    histories = {
        component: rng.standard_normal((n_samples, n_points))
        for component in mode_fields
    }
    coefficient_covector = (
        rng.standard_normal(n_samples - 1)
        + 1j * rng.standard_normal(n_samples - 1)
    )
    electric_weights, magnetic_weights = _coefficient_projection_weights(
        mode_fields,
        normal,
        spatial_weights,
    )
    overlap_pullback = _coefficient_overlap_pullback(
        coefficient_covector,
        n_samples,
    )
    expanded_pullback = {
        component: overlap_pullback[:, 0, np.newaxis] * weights
        for component, weights in electric_weights.items()
    }
    expanded_pullback.update(
        {
            component: overlap_pullback[:, 1, np.newaxis] * weights
            for component, weights in magnetic_weights.items()
        }
    )
    dense_pullback = coefficient_history_pullback(
        histories,
        mode_fields,
        normal,
        spatial_weights,
        coefficient_covector,
    )
    expected_pullback = _dense_pullback_reference(
        mode_fields,
        normal,
        spatial_weights,
        coefficient_covector,
        n_samples,
    )

    assert overlap_pullback.shape == (n_samples, 2)
    np.testing.assert_array_equal(
        overlap_pullback[[0, -1], 1],
        0.5 * coefficient_covector[[0, -1]],
    )
    for component in dense_pullback:
        np.testing.assert_allclose(
            dense_pullback[component],
            expected_pullback[component],
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            expanded_pullback[component],
            expected_pullback[component],
            rtol=1.0e-13,
            atol=1.0e-13,
        )


@pytest.mark.mpi2
def test_native_overlap_monitor_matches_real_raw_history_with_complex_mode():
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")
    from tama.sampling_grid import FastEigenmodeOverlapMonitor

    source_position = mp.Vector3(-0.7, -0.2)
    positions = (
        mp.Vector3(0.05, -0.43),
        mp.Vector3(0.05, -0.11),
        mp.Vector3(0.05, 0.19),
        mp.Vector3(0.05, 0.47),
    )
    layout = mp.BinaryPartition(data=[(mp.X, 0.0), 0, 1])
    simulation = mp.Simulation(
        cell_size=mp.Vector3(3.0, 2.0),
        resolution=10,
        chunk_layout=layout,
        sources=[
            mp.Source(
                mp.GaussianSource(frequency=1.0, fwidth=0.8),
                component=mp.Hz,
                center=source_position,
            ),
            mp.Source(
                mp.GaussianSource(frequency=1.2, fwidth=0.9),
                component=mp.Ez,
                center=source_position + mp.Vector3(0.0, 0.3),
            ),
        ],
    )
    simulation.init_sim()
    assert simulation.fields.is_real

    mode_values = np.asarray(
        [
            1.0 + 0.4j,
            -0.3 + 1.2j,
            0.7 - 0.9j,
            -1.1 - 0.2j,
        ]
    )
    second_mode_values = np.asarray(
        [
            -0.6 + 0.8j,
            1.3 + 0.1j,
            -0.2 - 1.1j,
            0.5 - 0.7j,
        ]
    )
    mode_fields = {
        mp.Ey: mode_values,
        mp.Hz: mode_values,
        mp.Ez: second_mode_values,
        mp.Hy: -second_mode_values,
    }
    spatial_weights = np.asarray([0.6, 1.1, 0.8, 1.4])
    electric_weights, magnetic_weights = _coefficient_projection_weights(
        mode_fields,
        mp.Vector3(1.0),
        spatial_weights,
    )
    overlap_monitor = FastEigenmodeOverlapMonitor(
        simulation,
        positions,
        electric_weights,
        magnetic_weights,
    )
    with pytest.raises(ValueError, match="must be complex"):
        overlap_monitor.sample_history_into(np.empty(2, dtype=np.float64))
    raw_monitors = {
        component: FastPointMonitor(simulation, component, positions)
        for component in mode_fields
    }
    local_overlap_rows = []
    local_history_rows = {component: [] for component in mode_fields}

    def record(active_simulation):
        overlap_row = np.empty(2, dtype=np.complex128)
        overlap_monitor.sample_history_into(overlap_row)
        local_overlap_rows.append(overlap_row)
        for component, monitor in raw_monitors.items():
            history_row = np.empty(len(positions), dtype=np.float64)
            monitor.sample_history_into(history_row)
            local_history_rows[component].append(history_row)

    try:
        simulation.run(record, until=0.8)
        local_overlaps = np.asarray(local_overlap_rows)
        assert local_overlaps.shape[0] >= 2
        overlap_history = overlap_monitor.reduce_history(local_overlaps)
        histories = {
            component: monitor.reduce_history(
                np.asarray(local_history_rows[component])
            )
            for component, monitor in raw_monitors.items()
        }

        assert np.any(np.abs(overlap_history.imag) > 0.0)
        np.testing.assert_allclose(
            _coefficient_history_from_overlaps(overlap_history),
            coefficient_history(
                histories,
                mode_fields,
                mp.Vector3(1.0),
                spatial_weights,
            ),
            rtol=2.0e-12,
            atol=2.0e-13,
        )
    finally:
        simulation.reset_meep()
