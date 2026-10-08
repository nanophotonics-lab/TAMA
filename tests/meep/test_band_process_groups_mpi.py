import meep as mp
import numpy as np
import pytest

from tama.multi_tda_objective import MultiTDAObjective


@pytest.mark.mpi4
def test_band_reduction_stays_within_the_active_meep_process_group():
    if mp.comm.Get_size() != 4:
        pytest.skip("requires exactly four MPI ranks")

    group_index = mp.divide_parallel_processes(2)
    try:
        group_value = complex(group_index + 1)
        signals = np.zeros((3, 2), dtype=np.complex128)

        result = MultiTDAObjective._distributed_band_transform(
            signals,
            lambda band_index: np.full(
                signals.shape[0],
                group_value * (band_index + 1),
                dtype=np.complex128,
            ),
        )
        expected = np.column_stack(
            [
                np.full(signals.shape[0], group_value, dtype=np.complex128),
                np.full(signals.shape[0], 2 * group_value, dtype=np.complex128),
            ]
        )

        assert np.array_equal(result, expected)

        objective = object.__new__(MultiTDAObjective)
        objective._flat_target_positions = (None, None, None)
        objective._target_slices = (slice(0, 1), slice(1, 3))
        objective.target_sample_shapes = ((), (2,))
        regional_signals = np.zeros((3, 3), dtype=np.complex128)

        regional_result = objective._distributed_target_transform(
            regional_signals,
            lambda target_index: np.full(
                objective._raw_target_history(
                    regional_signals,
                    target_index,
                ).shape,
                group_value * (target_index + 1),
                dtype=np.complex128,
            ),
        )
        regional_expected = np.column_stack(
            [
                np.full(3, group_value, dtype=np.complex128),
                np.full(3, 2 * group_value, dtype=np.complex128),
                np.full(3, 2 * group_value, dtype=np.complex128),
            ]
        )
        assert np.array_equal(regional_result, regional_expected)
    finally:
        mp.end_divide_parallel()


if __name__ == "__main__":
    test_band_reduction_stays_within_the_active_meep_process_group()
