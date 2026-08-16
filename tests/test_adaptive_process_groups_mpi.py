import meep as mp
import numpy as np
import pytest

from tama.adaptive_chunking import AdaptiveAdjointChunkBalancer


@pytest.mark.mpi4
def test_adaptive_timing_gather_stays_within_the_active_meep_process_group():
    if mp.comm.Get_size() != 4:
        pytest.skip("requires exactly four MPI ranks")

    group_index = mp.divide_parallel_processes(2)
    try:
        local_seconds = 100.0 * group_index + mp.my_rank() + 1.0
        gathered = AdaptiveAdjointChunkBalancer._gather_rank_times(
            local_seconds
        )
        expected = 100.0 * group_index + np.arange(1.0, 3.0)

        assert np.array_equal(gathered, expected)
    finally:
        mp.end_divide_parallel()


if __name__ == "__main__":
    test_adaptive_timing_gather_stays_within_the_active_meep_process_group()
