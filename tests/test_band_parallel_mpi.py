import meep as mp
import numpy as np
import pytest

import tama as tm
from tama.multi_tda_objective import (
    temporal_convolve_signal,
    temporal_convolve_signal_transpose,
)


@pytest.mark.mpi2
def test_distributed_band_filters_match_local_reference_exactly():
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    dt = 0.05
    bands = [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8)]
    weights = np.array([1.0, 0.75, 1.25, 0.5])
    objective = tm.MultiTDAObjective(
        update_design=lambda _: None,
        sim_factory=lambda sources=None: None,
        t_final=1.0,
        monitor_positions=[mp.Vector3() for _ in bands],
        component=mp.Ez,
        wavelength_bands=bands,
        weights=weights,
        kernel_length=9,
        pixel_chunk=1,
        dt=dt,
    )
    time = np.arange(37) * dt
    signals = np.column_stack([
        np.exp((0.1j + 0.03 * band_index) * time)
        for band_index in range(len(bands))
    ]).astype(np.complex128)

    expected_filtered = np.column_stack([
        temporal_convolve_signal(
            signals[:, band_index],
            weights[band_index] * objective.kernels[band_index],
            dt,
        )
        for band_index in range(len(bands))
    ])
    distributed_filtered = objective._distributed_filter_monitor_signals(signals)
    assert np.array_equal(distributed_filtered, expected_filtered)

    coefficients = np.array([0.5, 1.5, 0.25, 2.0])
    expected_transpose = np.column_stack([
        temporal_convolve_signal_transpose(
            expected_filtered[:, band_index] * coefficients[band_index],
            objective.weighted_kernels[band_index],
            dt,
        )
        for band_index in range(len(bands))
    ])
    distributed_transpose = objective._distributed_target_transform(
        expected_filtered,
        lambda band_index: temporal_convolve_signal_transpose(
            expected_filtered[:, band_index:band_index + 1]
            * coefficients[band_index],
            objective.weighted_kernels[band_index],
            dt,
        ),
        output_time_count=signals.shape[0],
    )
    assert np.array_equal(distributed_transpose, expected_transpose)

    filtered_adjoint_signals = (
        np.conjugate(expected_filtered) + (0.25 - 0.1j)
    )
    expected_custom_transpose = np.column_stack([
        np.conjugate(
            temporal_convolve_signal_transpose(
                np.conjugate(
                    filtered_adjoint_signals[:, band_index]
                    * coefficients[band_index]
                ),
                objective.weighted_kernels[band_index],
                dt,
            )
        )
        for band_index in range(len(bands))
    ])
    distributed_custom_transpose = (
        objective._filter_transpose_adjoint_signals(
            filtered_adjoint_signals,
            coefficients,
        )
    )
    assert np.array_equal(
        distributed_custom_transpose,
        expected_custom_transpose,
    )


@pytest.mark.mpi2
def test_non_deduplicated_target_transform_failure_is_synchronized():
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    objective = tm.MultiTDAObjective(
        update_design=lambda _: None,
        sim_factory=lambda sources=None: None,
        t_final=1.0,
        monitor_positions=[mp.Vector3(), mp.Vector3(0.1)],
        component=mp.Ez,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        weights=[1.0, 1.0],
        kernel_length=9,
        pixel_chunk=1,
        dt=0.05,
    )
    assert not objective._uses_deduplicated_monitors
    signals = np.zeros((3, 2))

    def transform(target_index):
        if target_index == 1:
            raise ValueError("intentional target transform failure")
        return objective._raw_target_history(signals, target_index)

    with pytest.raises(
        (ValueError, RuntimeError),
        match="intentional target transform failure|another active Meep rank",
    ):
        objective._distributed_target_transform(signals, transform)


@pytest.mark.mpi2
def test_band_transform_failure_is_synchronized():
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    signals = np.zeros((3, 2))

    def transform(band_index):
        if band_index == 1:
            raise ValueError("intentional band transform failure")
        return signals[:, band_index]

    with pytest.raises(
        (ValueError, RuntimeError),
        match="intentional band transform failure|another active Meep rank",
    ):
        tm.MultiTDAObjective._distributed_band_transform(
            signals,
            transform,
        )


if __name__ == "__main__":
    test_distributed_band_filters_match_local_reference_exactly()
